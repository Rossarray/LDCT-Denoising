import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from fp import FP
def load_scan(path):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
   image = (image - MIN_B) / (MAX_B - MIN_B)
   return torch.from_numpy(image)
def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)

def crop(ds,patch_size=64,stride=64):
    patches = []
    for left in range(0, ds.shape[0] - patch_size + 1, stride):
        for top in range(0, ds.shape[1] - patch_size + 1, stride):
            patches.append(ds[left: left + patch_size, top: top + patch_size])
    return patches

def save_dataset(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    patients_list = sorted([d for d in os.listdir(args.data_path) if 'zip' not in d])
    fp=FP(360,512,1).cuda()
    for p_ind, patient in enumerate(patients_list):
        patient_input_path = os.path.join(args.data_path, patient,
                                          "quarter_{}mm".format(args.mm))
        patient_target_path = os.path.join(args.data_path, patient,
                                           "full_{}mm".format(args.mm))

        for path_ in [patient_input_path, patient_target_path]:
            full_pixels = get_pixels_hu(load_scan(path_))
            for pi in range(len(full_pixels)):
                io = 'input_sino' if 'quarter' in path_ else 'target_sino'
                f =  normalize_(full_pixels[pi], args.norm_range_min, args.norm_range_max).unsqueeze(0).unsqueeze(0).float().cuda()
                x = fp(f).squeeze(0).squeeze(0).cpu()
                f_name = '{}_{}_{}.npy'.format(patient, pi, io)
                np.save(os.path.join(args.save_path, f_name), x)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/media/Data/lyl/code/D45_3mm/ori_d45_3mm/ori_d45_3mm')
    parser.add_argument('--save_path', type=str, default=r'/media/Data/lyl/code/D45_3mm/ori_d45_3mm/npy_d45_3mm')

    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--mm', type=int, default=3)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)

    args = parser.parse_args()
    save_dataset(args)