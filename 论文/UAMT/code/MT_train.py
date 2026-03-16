import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
import h5py
import math

from medpy import metric
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

#from networks.vnet import VNet
from networks.VNet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='MT_unlabel16', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)


def test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    _, y, _, _, _, _ = model(test_patch)
                    if len(y) > 1:
                        y = y[0]
                    y = F.softmax(y, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(np.int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def var_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, dataset_name="LA"):
    if dataset_name == "LA":
        with open('/home/zhangmingxiu/dataset/2018LA_Seg_Training Set/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["/home/zhangmingxiu/dataset/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    elif dataset_name == "Pancreas_CT":
        with open('./data/Pancreas/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["./data/Pancreas/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    
    save_mode_path2 = "/home/zhangmingxiu/UA-MT-master/model/MT_unlabel8/MT(multi_initial)/iter_1300_dice_0.885308338341655.pth"
    model.load_state_dict(torch.load(save_mode_path2))
    save_mode_path2 = "/home/zhangmingxiu/UA-MT-master/model/MT_unlabel8/MT(multi_initial)/ema_iter_1300_dice_0.885308338341655.pth"
    ema_model.load_state_dict(torch.load(save_mode_path2))
    
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_test = LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    labeled_idxs = list(range(16))
    unlabeled_idxs = list(range(16, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch, name_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['name']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            #print(volume_batch.shape)
            #print(label_batch.shape)
            namestr = str(name_batch)
            #print(namestr)
            #print(name_batch.aaa())
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            
            outdis, outputs, out_conf8, out_conf7, out_conf6, out_conf5 = model(volume_batch)
            
            with torch.no_grad():
                ema_dis,ema_output, emaout_conf8, emaout_conf7, emaout_conf6, emaout_conf5 = ema_model(ema_inputs)
            
            
            #nib.save(nib.Nifti1Image(ema_output.astype(np.float32), np.eye(4)), "../model/prediction/teacher/" + epoch_num + "_pred.nii.gz")
            #nib.save(nib.Nifti1Image(label_batch.astype(np.float32), np.eye(4)), "../model/prediction/teacher/" + epoch_num +name+ "_img.nii.gz")
            
            '''
            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, 112, 112, 80]).cuda()
            
            for i in range(T//2):
                #randn_like:返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充
                #clamp,将输入的张量每个元素值压缩到min，max中即[-0.2,0.2]，具体做法：小于min的元素=min，大于max的元素=max
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    _,preds[2 * stride * i:2 * stride * (i + 1)], _, _, _, _ = ema_model(ema_inputs)
            
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, 112, 112, 80)
            preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
            if iter_num <= 2000:
                uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)
            else:
                uncertainty = torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)
            '''
            ## calculate the loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            loss_seg_conf8 = F.cross_entropy(out_conf8[:labeled_bs], label_batch[:labeled_bs])
            loss_seg_conf7 = F.cross_entropy(out_conf7[:labeled_bs], label_batch[:labeled_bs])
            loss_seg_conf6 = F.cross_entropy(out_conf6[:labeled_bs], label_batch[:labeled_bs])
            loss_seg_conf5 = F.cross_entropy(out_conf5[:labeled_bs], label_batch[:labeled_bs])

            ema_output_soft = F.softmax(ema_output,dim=1)
            outputs_soft = F.softmax(outputs, dim=1)
            out_conf8_soft = F.softmax(out_conf8, dim=1)
            out_conf7_soft = F.softmax(out_conf7, dim=1)
            out_conf6_soft = F.softmax(out_conf6, dim=1)
            out_conf5_soft = F.softmax(out_conf5, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_dice_conf8 = losses.dice_loss(out_conf8_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_dice_conf7 = losses.dice_loss(out_conf7_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_dice_conf6 = losses.dice_loss(out_conf6_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg_dice_conf5 = losses.dice_loss(out_conf5_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

            supervised_loss = 0.5*(loss_seg + loss_seg_conf8 + loss_seg_conf7 + loss_seg_conf6 + loss_seg_conf5
                             + loss_seg_dice + loss_seg_dice_conf8 + loss_seg_dice_conf7 + loss_seg_dice_conf6 + loss_seg_dice_conf5)
            #Temporal Ensembling for Semi-Supervised Learning
            consistency_weight = get_current_consistency_weight(iter_num//150)

            
            consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output) #(batch, 2, 112,112,80)
            consistency_dist_conf8 = consistency_criterion(outputs[labeled_bs:], emaout_conf8)
            consistency_dist_conf7 = consistency_criterion(outputs[labeled_bs:], emaout_conf7)
            consistency_dist_conf6 = consistency_criterion(outputs[labeled_bs:], emaout_conf6)
            consistency_dist_conf5 = consistency_criterion(outputs[labeled_bs:], emaout_conf5)

            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(2)
            

#            mask = (uncertainty<threshold).float()
#            consistency_dist = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
#            consistency_dist_conf8 = torch.sum(mask*consistency_dist_conf8)/(2*torch.sum(mask)+1e-16)
#            consistency_dist_conf7 = torch.sum(mask*consistency_dist_conf7)/(2*torch.sum(mask)+1e-16)
#            consistency_dist_conf6 = torch.sum(mask*consistency_dist_conf6)/(2*torch.sum(mask)+1e-16)
#            consistency_dist_conf5 = torch.sum(mask*consistency_dist_conf5)/(2*torch.sum(mask)+1e-16)

            consistency_dist = torch.sum(consistency_dist)/(torch.sum(ema_output)+1e-16)
            consistency_dist_conf8 = torch.sum(consistency_dist_conf8)/(torch.sum(ema_output)+1e-16)
            consistency_dist_conf7 = torch.sum(consistency_dist_conf7)/(torch.sum(ema_output)+1e-16)
            consistency_dist_conf6 = torch.sum(consistency_dist_conf6)/(torch.sum(ema_output)+1e-16)
            consistency_dist_conf5 = torch.sum(consistency_dist_conf5)/(torch.sum(ema_output)+1e-16)
            #consistency_dist = torch.sum(consistency_dist)/(torch.sum(ema_output)+1e-16)
            
            consistency_loss = consistency_weight * (consistency_dist + consistency_dist_conf8 + consistency_dist_conf7
                                     + consistency_dist_conf6 + consistency_dist_conf5)

            loss_dis_dice = 0
            cross_task_loss = 0
            ### cross_task
            dis_to_mask = torch.sigmoid(-1500*outdis)
            #loss_dis_dice = losses.dice_loss(dis_to_mask[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
#cross_task_dist = torch.sum(mask*torch.mean((dis_to_mask - outputs_soft) ** 2))/(2*torch.sum(mask)+1e-16)      
            cross_task_dist = torch.mean((dis_to_mask - outputs_soft) ** 2)      
            #cross_task_loss = consistency_weight * cross_task_dist

            loss = 0.5*supervised_loss + consistency_loss + cross_task_loss + 0.5*loss_dis_dice


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            '''
            writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            '''
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
            
            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f, loss_dis: %f, loss_task: %f, loss_supervised: %f, loss_consitency: %f' %
                         (iter_num, loss.item(), consistency_dist.item(), consistency_weight,cross_task_loss,loss_dis_dice,supervised_loss,consistency_loss))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = dis_to_mask[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Dis2Mask', grid_image, iter_num)

                image = outdis[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/DistMap', grid_image, iter_num)


                image = torch.max(ema_output_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/ema_label', grid_image, iter_num)
                
                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)
                
                '''
                image = uncertainty[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/uncertainty', grid_image, iter_num)

                mask2 = (uncertainty > threshold).float()
                image = mask2[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/mask', grid_image, iter_num)
                '''
                
                #####
                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))


            if iter_num % 100 == 0:
                model.eval()
                
                dice_sample = var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4, dataset_name = 'LA')
                
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_best_path = os.path.join(snapshot_path,'./MT(multi_initial)/my_best_model.pth')
                    save_bestema_path = os.path.join(snapshot_path,'./MT(multi_initial)/my_best_emamodel.pth')
                    torch.save(model.state_dict(), save_best_path)
                    torch.save(ema_model.state_dict(), save_bestema_path)
                    logging.info("save best model to {}".format(save_best_path))
                save_mode_path = os.path.join(snapshot_path,  './MT(multi_initial)/iter_{}_dice_{}.pth'.format(iter_num, dice_sample))
                save_modeema_path = os.path.join(snapshot_path,  './MT(multi_initial)/ema_iter_{}_dice_{}.pth'.format(iter_num, dice_sample))
                torch.save(model.state_dict(), save_mode_path)
                torch.save(ema_model.state_dict(), save_modeema_path)
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()
                


            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, './MTmulti/iter_'+str(max_iterations)+'.pth')
    save_modeema_path = os.path.join(snapshot_path, './MTmulti/iter_'+str(max_iterations)+'ema.pth')
    torch.save(model.state_dict(), save_mode_path)
    torch.save(ema_model.state_dict(), save_modeema_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
