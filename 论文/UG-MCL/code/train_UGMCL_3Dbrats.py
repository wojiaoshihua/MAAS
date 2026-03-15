import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/zhangmingxiu/dataset/BraTS2019/MICCAI_BraTS_2019_Data_Training', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BraTs2019_UGMCL', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D_dt', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=50,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2
    labeled_bs = args.labeled_bs
    consistency_criterion = losses.softmax_mse_loss

    def create_model(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model,
                             in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    #save_mode_path = "/home/zhangmingxiu/UG-MCL-main/model/BraTs2019_UGMCL_25/unet_3D_dt/iter_22400_dice_0.843.pth"
    #model.load_state_dict(torch.load(save_mode_path))
    #ema_model.load_state_dict(torch.load(save_mode_path))

    db_train = BraTS2019(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 250))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise
            outdis, outputs,out_conf8, out_conf7, out_conf6, out_conf5= model(volume_batch)
            with torch.no_grad():
                ema_dis, ema_output, emaout_conf8, emaout_conf7, emaout_conf6, emaout_conf5 = ema_model(ema_inputs)
            T = 8
            volume_batch_r = volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, 96, 96, 96]).cuda()
            for i in range(T//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    p_dis, preds[2 * stride * i:2 * stride * (i + 1)],_ ,_ ,_ ,_ = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, 96, 96, 96)
            preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
            if iter_num <= 2000:
                uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)
            else:
                uncertainty = torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)

            ## calculate the loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            #loss_seg_conf8 = F.cross_entropy(out_conf8[:labeled_bs], label_batch[:labeled_bs])
            #loss_seg_conf7 = F.cross_entropy(out_conf7[:labeled_bs], label_batch[:labeled_bs])
            #loss_seg_conf6 = F.cross_entropy(out_conf6[:labeled_bs], label_batch[:labeled_bs])
            #loss_seg_conf5 = F.cross_entropy(out_conf5[:labeled_bs], label_batch[:labeled_bs])

            ema_output_soft = F.softmax(ema_output,dim=1)
            outputs_soft = F.softmax(outputs, dim=1)
            #out_conf8_soft = F.softmax(out_conf8, dim=1)
            #out_conf7_soft = F.softmax(out_conf7, dim=1)
            #out_conf6_soft = F.softmax(out_conf6, dim=1)
            #out_conf5_soft = F.softmax(out_conf5, dim=1)

            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            #loss_seg_dice_conf8 = losses.dice_loss(out_conf8_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            #loss_seg_dice_conf7 = losses.dice_loss(out_conf7_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            #loss_seg_dice_conf6 = losses.dice_loss(out_conf6_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            #loss_seg_dice_conf5 = losses.dice_loss(out_conf5_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

            #supervised_loss = 0.1*(loss_seg + loss_seg_conf8 + loss_seg_conf7 + loss_seg_conf6 + loss_seg_conf5
            #                 + loss_seg_dice + loss_seg_dice_conf8 + loss_seg_dice_conf7 + loss_seg_dice_conf6 + loss_seg_dice_conf5)

            supervised_loss = 0.1*(loss_seg + loss_seg_dice)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            # consistency_dist = consistency_criterion(outputs, ema_output) #(batch, 2, 112,112,80)
            # dual uncertainty
            consistency_dist = consistency_criterion(outputs, ema_output) #(batch, 2, 112,112,80)
            #consistency_dist_conf8 = consistency_criterion(outputs, emaout_conf8)
            #consistency_dist_conf7 = consistency_criterion(outputs, emaout_conf7)
            #consistency_dist_conf6 = consistency_criterion(outputs, emaout_conf6)
            #consistency_dist_conf5 = consistency_criterion(outputs, emaout_conf5)
            #############
            
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(2)
            mask = (uncertainty<threshold).float()
            consistency_dist = 0.5 * torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
            #consistency_dist_conf8 = torch.sum(mask*consistency_dist_conf8)/(2*torch.sum(mask)+1e-16)
            #consistency_dist_conf7 = torch.sum(mask*consistency_dist_conf7)/(2*torch.sum(mask)+1e-16)
            #consistency_dist_conf6 = torch.sum(mask*consistency_dist_conf6)/(2*torch.sum(mask)+1e-16)
            #consistency_dist_conf5 = torch.sum(mask*consistency_dist_conf5)/(2*torch.sum(mask)+1e-16)

            
            #consistency_loss = consistency_weight * (consistency_dist + consistency_dist_conf8 + consistency_dist_conf7
            #                         + consistency_dist_conf6 + consistency_dist_conf5) / 5
            consistency_loss = consistency_weight * consistency_dist
            ### cross_task
            dis_to_mask = torch.sigmoid(-1500*outdis)
            loss_dis_dice = losses.dice_loss(dis_to_mask[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            
            #cross_task_dist = torch.sum(mask*torch.mean((dis_to_mask - outputs_soft) ** 2))/(2*torch.sum(mask)+1e-16)      
            cross_task_dist = torch.mean((dis_to_mask - outputs_soft) ** 2)      
            cross_task_loss = consistency_weight * cross_task_dist
            
            loss = supervised_loss + consistency_loss + cross_task_loss + 0.5*loss_dis_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            


            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_seg, iter_num)
            writer.add_scalar('info/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_seg.item(), loss_seg_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
