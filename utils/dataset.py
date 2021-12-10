from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os
import cv2
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        img_id = []
        #for file in listdir(imgs_dir):
         # if np.where(np.load(os.path.join(imgs_dir+file))["label"] == 5,1,0).sum() > 200:
          #  img_id.append(file)
        #print(img_id)
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        #print(pil_img.shape)
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        #print(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        #print(self.masks_dir + idx + self.mask_suffix + '.*')
        #print(mask_file)
        assert len(mask_file) == 1, \
            f'{len(mask_file)}{self.masks_dir + idx + self.mask_suffix}Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        #print(mask_file[0])
        mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)#np.load(mask_file[0])#["label"]
        #print(mask)
        #mask = np.where(mask.astype("float32")==5,1.0,0)
        #print(mask.sum())
        mask = Image.fromarray(mask)
        img = Image.fromarray(cv2.imread(img_file[0]))#np.load(img_file[0]))#["image"]

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        #print("img=",img)
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        #print(mask.sum())
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
