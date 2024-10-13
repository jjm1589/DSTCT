import os

import h5py
import numpy as np
import SimpleITK as sitk

label_dir = "/root/autodl-tmp/SSL4MIS/data/FHPS/label_mha/"
image_dir = "/root/autodl-tmp/SSL4MIS/data/FHPS/image_mha/"
label_list = os.listdir(label_dir)
image_list = os.listdir(image_dir)
slice_num = 0
for case in image_list:
    # print(os.path.join(image_dir, case))
    img_itk = sitk.ReadImage(os.path.join(image_dir, case))
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)
    image = np.mean(image, axis=0)
   
    label_itk = sitk.ReadImage(os.path.join(label_dir, case))
    mask = sitk.GetArrayFromImage(label_itk)
    # mask = np.tile(mask, (3, 1, 1))
    # print(mask.shape)
    image = (image - image.min()) / (image.max() - image.min())
    image = image.astype(np.float32)
    item = case.split(".")[0]
    if image.shape != mask.shape:
            print("Error")
    f = h5py.File("/root/autodl-tmp/SSL4MIS/data/FHPS/data/slices/{}_slice_{}.h5".format(item, 0), 'w')
    f.create_dataset('image', data=image, compression="gzip")
    f.create_dataset('label', data=mask, compression="gzip")
    f.close()
    slice_num += 1

print("Total {} slices".format(slice_num))
