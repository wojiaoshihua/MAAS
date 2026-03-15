import argparse
import os
import shutil
from glob import glob

import torch
from networks.net_factory_3d import net_factory_3d
from inference_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/zhangmingxiu/dataset/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA_ACMT_SErr_clv2_16', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id')
FLAGS = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]

def Inference(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "../model/{}/{}_Prediction".format(FLAGS.exp, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes).cuda()
    ema_net = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes).cuda()
    save_mode_path = os.path.join(
        snapshot_path, 'multi_initial/iter_10000_dice_0.9097.pth')
    save_emamode_path = os.path.join(
        snapshot_path, 'multi_initial/emaiter_10000_dice_0.9097.pth')
    net.load_state_dict(torch.load(save_mode_path))
    ema_net.load_state_dict(torch.load(save_emamode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    ema_net.eval()
    avg_metric = test_all_case(net, ema_net, image_list=image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4, test_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':
    
    

    metric = Inference(FLAGS)
    print('dice, jc, hd, asd:', metric)
