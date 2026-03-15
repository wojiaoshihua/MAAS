import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, nms=0):
    total_metric = 0.0
    cnt = 0
    sample_entropy = torch.zeros(64).cuda()
    for image_path in tqdm(image_list):
        cnt += 1
        id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map, prediction_conf8, prediction_conf7, prediction_conf6, prediction_conf5, sample_entropy_single, confidence_single, diversity_single, edgeInformation_single = test_single_case(cnt, net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, test_save_path=test_save_path)
        a = sample_entropy_single
        b = confidence_single
        c = diversity_single
        d = edgeInformation_single

        if nms:
            prediction = getLargestCC(prediction)

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            print(single_metric)
        total_metric += np.asarray(single_metric)

        if save_result:
            #nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "entropy" + str(a) + id + "_pred.nii.gz")
            #nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "confidence" + str(b) + id + "_pred.nii.gz")
            #nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "diversity" + str(c) + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "edge" + str(d) + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")

            nib.save(nib.Nifti1Image(prediction_conf8.astype(np.float32), np.eye(4)), test_save_path + id + "_pred_conf8.nii.gz")
            nib.save(nib.Nifti1Image(prediction_conf7.astype(np.float32), np.eye(4)), test_save_path + id + "_pred_conf7.nii.gz")
            nib.save(nib.Nifti1Image(prediction_conf6.astype(np.float32), np.eye(4)), test_save_path + id + "_pred_conf6.nii.gz")
            nib.save(nib.Nifti1Image(prediction_conf5.astype(np.float32), np.eye(4)), test_save_path + id + "_pred_conf5.nii.gz")
            
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test_single_case(cnt_epoch, net, image, stride_xy, stride_z, patch_size,test_save_path, num_classes=1):
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
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    score_map_dis = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    score_map_conf8 = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    score_map_conf7 = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    score_map_conf6 = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    score_map_conf5 = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)
    cnt_dis = np.zeros(image.shape).astype(np.float32)
    cnt_conf8 = np.zeros(image.shape).astype(np.float32)
    cnt_conf7 = np.zeros(image.shape).astype(np.float32)
    cnt_conf6 = np.zeros(image.shape).astype(np.float32)
    cnt_conf5 = np.zeros(image.shape).astype(np.float32)
    entropy = 0
    diversity = 0
    diversity_conf8 = 0
    diversity_conf7 = 0
    diversity_conf6 = 0
    diversity_conf5 = 0
    total_diversity = 0
    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1_dis, y1, y1_conf8, y1_conf7, y1_conf6, y1_conf5 = net(test_patch)
                #print(y1_conf8.shape)
                #print(y1_conf7.shape)
                #print(y1_conf6.shape)
                #print(y1_conf5.shape)
                y = F.softmax(y1, dim=1)
                y_conf8 = F.softmax(y1_conf8, dim=1)
                y_conf7 = F.softmax(y1_conf7, dim=1)
                y_conf6 = F.softmax(y1_conf6, dim=1)
                y_conf5 = F.softmax(y1_conf5, dim=1)
                y_dis = F.softmax(-1500*y1_dis, dim=1)

                total_diversity += 1
                #print(y.shape)
                #entropy += -torch.sum(y * torch.log2(y + 1e-10), dim=1)
                #sample_entropy = torch.mean(pixel_entropy, dim=(1, 2, 3, 4))
                #print(entropy.shape)
                #print(entropy)
                y = y.cpu().data.numpy()
                y_conf8 = y_conf8.cpu().data.numpy()
                y_conf7 = y_conf7.cpu().data.numpy()
                y_conf6 = y_conf6.cpu().data.numpy()
                y_conf5 = y_conf5.cpu().data.numpy()
                y_dis = y_dis.cpu().data.numpy()

                y = y[0,:,:,:,:]
                y_conf8 = y_conf8[0,:,:,:,:]
                y_conf7 = y_conf7[0,:,:,:,:]
                y_conf6 = y_conf6[0,:,:,:,:]
                y_conf5 = y_conf5[0,:,:,:,:]
                y_dis = y_dis[0,:,:,:,:]


                y_label_map = np.argmax(y, axis = 0)
                y_label_map_conf8 = np.argmax(y_conf8, axis = 0)
                y_label_map_conf8 = 1 - y_label_map_conf8
                y_label_map_conf7 = np.argmax(y_conf7, axis = 0)
                y_label_map_conf6 = np.argmax(y_conf6, axis = 0)
                y_label_map_conf5 = np.argmax(y_conf5, axis = 0)
                #nib.save(nib.Nifti1Image(y_label_map_conf8.astype(np.float32), np.eye(4)), test_save_path + str(total_diversity) + "_pred_conf8.nii.gz")
                #nib.save(nib.Nifti1Image(y_label_map_conf7.astype(np.float32), np.eye(4)), test_save_path + str(total_diversity) + "_pred_conf7.nii.gz")
                #nib.save(nib.Nifti1Image(y_label_map_conf6.astype(np.float32), np.eye(4)), test_save_path + str(total_diversity) + "_pred_conf6.nii.gz")
                #nib.save(nib.Nifti1Image(y_label_map_conf5.astype(np.float32), np.eye(4)), test_save_path + str(total_diversity) + "_pred_conf5.nii.gz")
                #nib.save(nib.Nifti1Image(y_label_map.astype(np.float32), np.eye(4)), test_save_path + str(total_diversity) + "_pred.nii.gz")
                #print(y_dis.aaa())
                #print(y_label_map.shape)
                
                y_label_map_min = np.min(y_label_map)
                y_label_map_max = np.max(y_label_map)
                count_min = (y_label_map == y_label_map_min).sum()
                count_max = (y_label_map == y_label_map_max).sum()
                count_total = y_label_map.size

                y_label_map_max_conf8 = np.max(y_label_map_conf8)
                count_max_conf8 = (y_label_map_conf8 == y_label_map_max_conf8).sum()
                y_label_map_max_conf7 = np.max(y_label_map_conf7)
                count_max_conf7 = (y_label_map_conf7 == y_label_map_max_conf7).sum()
                y_label_map_max_conf6 = np.max(y_label_map_conf6)
                count_max_conf6 = (y_label_map_conf6 == y_label_map_max_conf6).sum()
                y_label_map_max_conf5 = np.max(y_label_map_conf5)
                count_max_conf5 = (y_label_map_conf5 == y_label_map_max_conf5).sum()
                #print(count_min,count_max,count_total)
                if count_max < diversity_const:
                    diversity += 1
                if count_max_conf8 < diversity_const:
                  diversity_conf8 += 1
                if count_max_conf7 < diversity_const:
                  diversity_conf7 += 1
                if count_max_conf6 < diversity_const:
                  diversity_conf6 += 1
                if count_max_conf5 < diversity_const:
                  diversity_conf5 += 1




                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                score_map_dis[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map_dis[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y_dis
                score_map_conf8[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map_conf8[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y_conf8
                score_map_conf7[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map_conf7[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y_conf7
                score_map_conf6[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map_conf6[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y_conf6
                score_map_conf5[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map_conf5[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y_conf5
                
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
                cnt_dis[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt_dis[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

                cnt_conf8[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt_conf8[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
                cnt_conf7[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt_conf7[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
                cnt_conf6[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt_conf6[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
                cnt_conf5[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt_conf5[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    #print(score_map.aaa())            
    score_map = score_map/np.expand_dims(cnt,axis=0)
    score_map_dis = score_map_dis/np.expand_dims(cnt_dis,axis=0)
    score_map_conf8 = score_map_conf8/np.expand_dims(cnt_conf8,axis=0)
    score_map_conf7 = score_map_conf7/np.expand_dims(cnt_conf7,axis=0)
    score_map_conf6 = score_map_conf6/np.expand_dims(cnt_conf6,axis=0)
    score_map_conf5 = score_map_conf5/np.expand_dims(cnt_conf5,axis=0)

    tensor_score_map = torch.from_numpy(score_map)
    tensor_score_map_dis = torch.from_numpy(score_map_dis)
    tensor_score_map_conf8 = torch.from_numpy(score_map_conf8)
    tensor_score_map_conf7 = torch.from_numpy(score_map_conf7)
    tensor_score_map_conf6 = torch.from_numpy(score_map_conf6)
    tensor_score_map_conf5 = torch.from_numpy(score_map_conf5)

    edgeInformation = torch.mean((tensor_score_map_dis - tensor_score_map) ** 2)
    edgeInformation_conf8 = torch.mean((tensor_score_map_conf8 - tensor_score_map_dis) ** 2)
    edgeInformation_conf7 = torch.mean((tensor_score_map_conf7 - tensor_score_map_dis) ** 2)
    edgeInformation_conf6 = torch.mean((tensor_score_map_conf6 - tensor_score_map_dis) ** 2)
    edgeInformation_conf5 = torch.mean((tensor_score_map_conf5 - tensor_score_map_dis) ** 2)

    tensor_score_map = tensor_score_map.permute(1,2,3,0)
    tensor_score_map_conf8 = tensor_score_map_conf8.permute(1,2,3,0)
    tensor_score_map_conf7 = tensor_score_map_conf7.permute(1,2,3,0)
    tensor_score_map_conf6 = tensor_score_map_conf6.permute(1,2,3,0)
    tensor_score_map_conf5 = tensor_score_map_conf5.permute(1,2,3,0)
    
    #print(tensor_score_map.shape)
    confidence = np.zeros((tensor_score_map.shape[0], tensor_score_map.shape[1], tensor_score_map.shape[2]))
    confidence_conf8 = np.zeros((tensor_score_map_conf8.shape[0], tensor_score_map_conf8.shape[1], tensor_score_map_conf8.shape[2]))
    confidence_conf7 = np.zeros((tensor_score_map_conf7.shape[0], tensor_score_map_conf7.shape[1], tensor_score_map_conf7.shape[2]))
    confidence_conf6 = np.zeros((tensor_score_map_conf6.shape[0], tensor_score_map_conf6.shape[1], tensor_score_map_conf6.shape[2]))
    confidence_conf5 = np.zeros((tensor_score_map_conf5.shape[0], tensor_score_map_conf5.shape[1], tensor_score_map_conf5.shape[2]))
    
    entropy = -torch.sum(tensor_score_map * torch.log2(tensor_score_map + 1e-10), dim=3)  # 计算像素点的熵
    entropy_conf8 = -torch.sum(tensor_score_map_conf8 * torch.log2(tensor_score_map_conf8 + 1e-10), dim=3)
    entropy_conf7 = -torch.sum(tensor_score_map_conf7 * torch.log2(tensor_score_map_conf7 + 1e-10), dim=3)
    entropy_conf6 = -torch.sum(tensor_score_map_conf6 * torch.log2(tensor_score_map_conf6 + 1e-10), dim=3)
    entropy_conf5 = -torch.sum(tensor_score_map_conf5 * torch.log2(tensor_score_map_conf5 + 1e-10), dim=3)

    sample_entropy = torch.mean(entropy, dim=(0, 1, 2))
    sample_entropy_conf8 = torch.mean(entropy_conf8, dim=(0, 1, 2))
    sample_entropy_conf7 = torch.mean(entropy_conf7, dim=(0, 1, 2))
    sample_entropy_conf6 = torch.mean(entropy_conf6, dim=(0, 1, 2))
    sample_entropy_conf5 = torch.mean(entropy_conf5, dim=(0, 1, 2))
    #print(entropy)
    #print(entropy.shape)
    print("样本熵",sample_entropy)
    #print(sample_entropy.shape)
    #print(score_map)
    #confidence=torch.abs(tensor_score_map[...,1]-tensor_score_map[...,0])
    #confidence_conf8=torch.abs(tensor_score_map_conf8[...,1]-tensor_score_map_conf8[...,0])
    count = ((tensor_score_map > 0.3) & (tensor_score_map < 0.7)).sum()
    count_conf8 = ((tensor_score_map_conf8 > 0.3) & (tensor_score_map_conf8 < 0.7)).sum()
    count_conf7 = ((tensor_score_map_conf7 > 0.3) & (tensor_score_map_conf7 < 0.7)).sum()
    count_conf6 = ((tensor_score_map_conf6 > 0.3) & (tensor_score_map_conf6 < 0.7)).sum()
    count_conf5 = ((tensor_score_map_conf5 > 0.3) & (tensor_score_map_conf5 < 0.7)).sum()
    #total_count = confidence.numel()
    total_count = tensor_score_map.numel()
    print("不确定体素个数",count)
    print("总体素个数",total_count)
    print("不确定体素/总体素",count/total_count)
    print("多样性",diversity)
    print("总块数",total_diversity)
    print("多样性百分比",diversity/total_diversity)
    print("边界信息",edgeInformation)
    file_name = "a.txt"
    with open(file_name, 'w') as file:
      file.write(str(cnt_epoch) + '\t')
      file.write(str(sample_entropy) + '\t')
      file.write(str(sample_entropy_conf8) + '\t')
      file.write(str(sample_entropy_conf7) + '\t')
      file.write(str(sample_entropy_conf6) + '\t')
      file.write(str(sample_entropy_conf5) + '\t')
      file.write(str(count) + '\t')
      file.write(str(count_conf8) + '\t')
      file.write(str(count_conf7) + '\t')
      file.write(str(count_conf6) + '\t')
      file.write(str(count_conf5) + '\t')
      file.write(str(total_count) + '\t')
      file.write(str(count/total_count) + '\t')
      file.write(str(count_conf8/total_count) + '\t')
      file.write(str(count_conf7/total_count) + '\t')
      file.write(str(count_conf6/total_count) + '\t')
      file.write(str(count_conf5/total_count) + '\t')
      file.write(str(diversity_conf8) + '\t')
      file.write(str(diversity_conf7) + '\t')
      file.write(str(diversity_conf6) + '\t')
      file.write(str(diversity_conf5) + '\t')
      file.write(str(total_diversity) + '\t')
      file.write(str(diversity/total_diversity) + '\t')
      file.write(str(diversity_conf8/total_diversity) + '\t')
      file.write(str(diversity_conf7/total_diversity) + '\t')
      file.write(str(diversity_conf6/total_diversity) + '\t')
      file.write(str(diversity_conf5/total_diversity) + '\t')
      file.write(str(edgeInformation_conf8) + '\t')
      file.write(str(edgeInformation_conf7) + '\t')
      file.write(str(edgeInformation_conf6) + '\t')
      file.write(str(edgeInformation_conf5) + '\t')
      file.write('\r\n')
      
    # 将张量的数值转换为 NumPy 数组，然后迭代写入文件
    #for row in tensor.numpy():
    #    for value in row:
    #        file.write(str(value) + '\t')
    #    file.write('\n')
    score_map_conf8 = 1 - score_map_conf8
    label_map = np.argmax(score_map, axis = 0)
    label_map_conf8 = np.argmax(score_map_conf8, axis = 0)
    label_map_conf7 = np.argmax(score_map_conf7, axis = 0)
    label_map_conf6 = np.argmax(score_map_conf6, axis = 0)
    label_map_conf5 = np.argmax(score_map_conf5, axis = 0)
    
    #print(label_map.shape)
    #print(label_map)
    
    #print(score_map.aaa())
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map_conf8 = score_map_conf8[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map_conf7 = score_map_conf7[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map_conf6 = score_map_conf6[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map_conf5 = score_map_conf5[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map, label_map_conf8, label_map_conf7, label_map_conf6, label_map_conf5, sample_entropy, count/total_count, diversity/total_diversity, edgeInformation

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
