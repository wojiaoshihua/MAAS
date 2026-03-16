import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm


def test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    cnt = 0
    #sample_entropy = torch.zeros(64).cuda()
    A = {}
    B = {}
    C = {}
    D = {}
    for image_path in tqdm(image_list):
        cnt += 1
        id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, prediction_fusion, score_map, prediction_conf8, prediction_conf7, prediction_conf6, prediction_conf5, sample_entropy_single, confidence_single, diversity_single, edgeInformation_single = test_single_case(cnt, net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, test_save_path=test_save_path)
        a = sample_entropy_single
        b = confidence_single
        c = diversity_single
        d = edgeInformation_single
        A[id] = a
        B[id] = b
        C[id] = c
        D[id] = d
        
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            print(single_metric)
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "entropy" + str(a) + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "confidence" + str(b) + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "diversity" + str(c) + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "edge" + str(d) + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(prediction_fusion.astype(np.float32), np.eye(4)), test_save_path + "fusion" + str(d) + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")

            nib.save(nib.Nifti1Image(prediction_conf8.astype(np.float32), np.eye(4)), test_save_path + id + "_pred_conf8.nii.gz")
            nib.save(nib.Nifti1Image(prediction_conf7.astype(np.float32), np.eye(4)), test_save_path + id + "_pred_conf7.nii.gz")
            nib.save(nib.Nifti1Image(prediction_conf6.astype(np.float32), np.eye(4)), test_save_path + id + "_pred_conf6.nii.gz")
            nib.save(nib.Nifti1Image(prediction_conf5.astype(np.float32), np.eye(4)), test_save_path + id + "_pred_conf5.nii.gz")
        #print(A)
        #if cnt == 5:
        #  break    
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    A = dict(sorted(A.items(), key=lambda item: item[1], reverse=True))
    B = dict(sorted(B.items(), key=lambda item: item[1], reverse=True))
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    D = dict(sorted(D.items(), key=lambda item: item[1]))
    select = []
    A_keys = list(A.keys())
    B_keys = list(B.keys())
    C_keys = list(C.keys())
    D_keys = list(D.keys())
    A_value = list(A.values())
    B_value = list(B.values())
    C_value = list(C.values())
    D_value = list(D.values())
    i = 0
    while len(select) < 8:
      if A_keys[i] not in select:
        select.append(A_keys[i])
      if B_keys[i] not in select:
        select.append(B_keys[i])
      if C_keys[i] not in select:
        select.append(C_keys[i])
      if D_keys[i] not in select:
        select.append(D_keys[i])
      i = i + 1
    #print(select)
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
    diversity_const = 100000
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
    tensor_score_map_stack = torch.stack([tensor_score_map,tensor_score_map_conf8,tensor_score_map_conf7,tensor_score_map_conf6,tensor_score_map_conf5])
    

    #模糊融合
    #print(tensor_score_map_stack.shape)#5 2 171 146 88
    CF = tensor_score_map_stack.permute(0,2,3,4,1).cuda()
    #print(CF)
    #print(CF.shape)#5,171,146,88,2
    #print(CF[0][0][0][0][0])
    #print(CF[0][0][0][0][1])
    #print(ww.aaa())
    #print(CF[0][0][0][0][2])
    #print(tensor_score_map_conf5)
    #print(tensor_score_map.aaa())
    
    R_L = fuzzy_rank(CF, 2)
    #print(R_L)
    #print(R_L.shape)
    RS = torch.sum(R_L,dim=0)
    CFS = CFS_func(CF, R_L)
    FS = (RS*CFS)
    score_prediction = torch.softmax(FS, dim=-1).cpu()
    score_prediction = torch.flip(score_prediction, dims=(-1,))
    #print(score_prediction)
    #print(score_prediction.shape)#171 146 88 2
    #print(score_prediction.aaa())
    FS = FS.cpu().numpy()
    
    #print(CFS)
    #print(CFS.shape) #171 146 88 2
    #print(FS)
    #print(FS.shape)  #171 146 88 2
    
    predictions = np.zeros(shape = (FS.shape[0],FS.shape[1],FS.shape[2]))
    for i in range(FS.shape[0]):
      for j in range(FS.shape[1]):
        for k in range(FS.shape[2]):
          predictions[i][j][k] = np.argmin(FS[i][j][k], axis=-1)
    
    #print(predictions)
    #print(predictions.shape)#171 146 88

    #print(ww.aaa())
    edgeInformation = torch.mean((tensor_score_map_dis - tensor_score_map) ** 2)
    edgeInformation_conf8 = torch.mean((tensor_score_map_conf8 - tensor_score_map_dis) ** 2)
    edgeInformation_conf7 = torch.mean((tensor_score_map_conf7 - tensor_score_map_dis) ** 2)
    edgeInformation_conf6 = torch.mean((tensor_score_map_conf6 - tensor_score_map_dis) ** 2)
    edgeInformation_conf5 = torch.mean((tensor_score_map_conf5 - tensor_score_map_dis) ** 2)
    edgeInformation_stack = torch.stack([edgeInformation,edgeInformation_conf8,edgeInformation_conf7,edgeInformation_conf6,edgeInformation_conf5])
    #平均
    #edgeInformation_final = (edgeInformation + edgeInformation_conf8 + edgeInformation_conf7 + edgeInformation_conf6 + edgeInformation_conf5) / 5
    #最大
    #edgeInformation_final = torch.max(edgeInformation_stack)
    #最小
    #edgeInformation_final = torch.min(edgeInformation_stack)
    #模糊融合
    score_prediction = score_prediction.permute(3,0,1,2)
    edgeInformation_final = torch.mean((tensor_score_map_dis - score_prediction) ** 2)
    score_prediction = score_prediction.permute(1,2,3,0)
    #print(tensor_score_map.shape) #2,171,146,88
    
    tensor_score_map = tensor_score_map.permute(1,2,3,0)
    #print(tensor_score_map.shape) #171,146,88,2
    
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
    entropy_fusion = -torch.sum(score_prediction * torch.log2(score_prediction + 1e-10), dim=3)

    sample_entropy = torch.mean(entropy, dim=(0, 1, 2))
    sample_entropy_conf8 = torch.mean(entropy_conf8, dim=(0, 1, 2))
    sample_entropy_conf7 = torch.mean(entropy_conf7, dim=(0, 1, 2))
    sample_entropy_conf6 = torch.mean(entropy_conf6, dim=(0, 1, 2))
    sample_entropy_conf5 = torch.mean(entropy_conf5, dim=(0, 1, 2))
    sample_entropy_fusion = torch.mean(entropy_fusion, dim=(0, 1, 2))
    sample_entropy_stack = torch.stack([sample_entropy,sample_entropy_conf8,sample_entropy_conf7,sample_entropy_conf6,sample_entropy_conf5])
    #平均
    #sample_entropy_final = (sample_entropy + sample_entropy_conf8 + sample_entropy_conf7 + sample_entropy_conf6 + sample_entropy_conf5) / 5
    #最大
    #sample_entropy_final = torch.max(sample_entropy_stack)
    #最小
    #sample_entropy_final = torch.min(sample_entropy_stack)
    #模糊融合
    sample_entropy_final = sample_entropy_fusion
    #print(sample_entropy_stack.shape) 
    #print(sample_entropy_stack[1].shape)
    #print(sample_entropy_stack)
    '''
    CF = sample_entropy_stack.view(5,1,1).cpu().numpy()
    print(CF)
    print(CF.shape)
    #print(entropy.aaa())
    R_L = fuzzy_rank(CF, 2)
    print(R_L)
    print(R_L.shape)
    RS = np.sum(R_L,axis=0)
    print(RS)
    print(RS.shape)
    print(entropy.aaa())
    CFS = CFS_func(CF, R_L)
    FS = RS*CFS
    print(CFS)
    print(CFS.shape)
    print(FS)
    print(FS.shape)
    predictions = np.argmin(FS,axis=-1)
'''
    
    #print(entropy)
    #print(entropy.shape)
    #print("样本熵",sample_entropy)
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
    count_fusion = ((score_prediction > 0.3) & (score_prediction < 0.7)).sum()
    total_count = tensor_score_map.numel()
    #平均
    #confidence_single_final = (count + count_conf8 + count_conf7 + count_conf6 + count_conf5) / (5 * total_count)
    #最大
    #confidence_single_stack = torch.stack([count,count_conf8,count_conf7,count_conf6,count_conf5])
    #confidence_single_final = torch.max(confidence_single_stack) / total_count
    #最小
    #confidence_single_final = torch.min(confidence_single_stack) / total_count
    #模糊融合
    confidence_single_final = count_fusion / total_count
    #平均
    diversity_final = (diversity + diversity_conf8 + diversity_conf7 + diversity_conf6 + diversity_conf5) / (5*total_diversity)
    #最大
    #diversity_final = max(diversity,diversity_conf8,diversity_conf7,diversity_conf6,diversity_conf5) / total_diversity
    #最小
    #diversity_final = min(diversity,diversity_conf8,diversity_conf7,diversity_conf6,diversity_conf5) / total_diversity    
    #print("不确定体素个数",count)
    #print("总体素个数",total_count)
    #print("不确定体素/总体素",count/total_count)
    #print("多样性",diversity)
    #print("总块数",total_diversity)
    #print("多样性百分比",diversity/total_diversity)
    #print("边界信息",edgeInformation)
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
    #score_map_conf8 = 1 - score_map_conf8
    
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
    return label_map, predictions, score_map, label_map_conf8, label_map_conf7, label_map_conf6, label_map_conf5, sample_entropy, count/total_count, diversity/total_diversity, edgeInformation
#    return label_map, score_map, label_map_conf8, label_map_conf7, label_map_conf6, label_map_conf5, sample_entropy_final, confidence_single_final, diversity_final, edgeInformation_final

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


def fuzzy_rank(CF, top):
    #print(CF.shape) #5,171,146,88,2   3,745,2
    R_L = torch.zeros_like(CF)
    ''''
    if CF.device.type == 'cuda':
      #print('张量在GPU上')
    else:
      #print('张量在CPU上')
    if R_L.device.type == 'cuda':
      #print('张量在GPU上')
    else:
      #print('张量在CPU上')
    #print("0")
    
    for i in torch.arange(CF.shape[0], device='cuda'):
      for j1 in torch.arange(CF.shape[1], device='cuda'):
        for j2 in torch.arange(CF.shape[2], device='cuda'):
            for j3 in torch.arange(CF.shape[3], device='cuda'):
                for k in torch.arange(CF.shape[4], device='cuda'):
                    if k.device.type == 'cuda':
                      print('张量在GPU上')
                    else:
                      print('张量在CPU上')
                    R_L[i][j1][j2][j3][k] = 1 - torch.exp(-torch.exp(-2.0 * CF[i][j1][j2][j3][k]))  #Gompertz Function
    '''
    R_L = 1 - torch.exp(-torch.exp(-2.0 * CF))
    #print("1")

    K_L = 0.632 * torch.ones_like(R_L, device='cuda') #initiate all values as penalty values
    _, top_indices = torch.topk(R_L, top, largest=True)
    K_L.scatter_(-1, top_indices, R_L.gather(-1, top_indices))
    '''
    for i in range(R_L.shape[0]):
        for sample1 in range(R_L.shape[1]):
          for sample2 in range(R_L.shape[2]):
            for sample3 in range(R_L.shape[3]):
              for k in range(top):
                a = R_L[i][sample1][sample2][sample3]
                idx = np.where(a==np.partition(a, k)[k])
                #if sample belongs to top 'k' classes, R_L =R_L, else R_L = penalty value
                K_L[i][sample1][sample2][sample3][idx] = R_L[i][sample1][sample2][sample3][idx]#将满足条件的idx元组全部直接赋值
    '''
    #print("2")
    return K_L


def CFS_func(CF, K_L): #把对应为0.632部分的原始数据置为0
    H = CF.shape[0]  # no. of classifiers

    # 对 CF 进行替换操作
    replace_value = 0
    CF[K_L == 0.632] = replace_value

    # 沿着第一个维度求和
    CFS = 1 - torch.sum(CF, dim=0) / H
    return CFS

