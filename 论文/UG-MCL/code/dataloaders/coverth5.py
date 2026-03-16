import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import SimpleITK as sitk
listt = glob('/home/zhangmingxiu/dataset/BraTS2019/MICCAI_BraTS_2019_Data_Training/label/*.nii.gz')
for item in tqdm(listt):
    label = sitk.GetArrayFromImage(sitk.ReadImage(item))
    image = sitk.GetArrayFromImage(sitk.ReadImage(item.replace("label", "flair")))
    f = h5py.File(item.replace('_flair.nii.gz', '.h5'), 'w')
    f.create_dataset('image', data=image, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()