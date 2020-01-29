import numpy as np
import pandas as pd
import cv2

import os
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from math import sin, cos
import random


camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

camera_matrix_inv = np.linalg.inv(camera_matrix)
IMG_SHAPE = (2710, 3384, 3)
DISTANCE_THRESH_CLEAR = 2
IMG_WIDTH = 1536
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 4


def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img


def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords


def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x


def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image
        ys: y coordinates in the image
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys


# ######################################## Label processing ##########################################
def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict


def _regr_back(regr_dict):
    # {x,y,z,yaw, p_sin,p_cos, roll}
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)

    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict


# ######################################## Image processing ##########################################
def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:, ::-1]
    return (img / 255).astype('float32')


def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)
    for x, y, regr_dict in zip(xs, ys, coords):
        x, y = y, x
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4 / 3) / MODEL_SCALE
        y = np.round(y).astype('int')

        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1
            regr_dict = _regr_preprocess(regr_dict, flip)
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
    if flip:
        mask = np.array(mask[:, ::-1])
        regr = np.array(regr[:, ::-1])
    return mask, regr


# ######################################## Decode ##########################################


def convert_3d_to_2d(x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    return x * fx / z + cx, y * fy / z + cy


def optimize_xy(r, c, x0, y0, z0):
    # R,C coordinate in image plane
    # x0,y0,z0 coordinate in world
    def distance_fn(xyz):
        x, y, z = xyz
        x, y = convert_3d_to_2d(x, y, z0)   # world coordinate to image plane coordinate
        y, x = x, y
        # change prediction world
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / MODEL_SCALE
        y = np.round(y).astype('int')

        return (x - r) ** 2 + (y - c) ** 2

    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z0


def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2) ** 2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]


def extract_coords(prediction):
    """
    :param prediction: size:(1+7, W, H)
    :return:
    """
    logits = prediction[0]          # size:(W, H)
    regr_output = prediction[1:]    # size:(7, W, H)
    points = np.argwhere(logits > 0)
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))   # {x,y,z,yaw, p_sin,p_cos, roll}
        coords.append(_regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))  # sigmoid
        # use image plane coordinate (r,c) to refine predict world coordinate (x,y,z)
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = optimize_xy(r, c, coords[-1]['x'], coords[-1]['y'],
                                                                        coords[-1]['z'])
    coords = clear_duplicates(coords)
    return coords


def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)


# ######################################## Pytorch Dataset ##########################################

def brightness_change(images):
    images = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)(images)
    return images


class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, aug=True, test=False):
        self.df = dataframe
        self.root_dir = root_dir
        self.aug = aug
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name    ################################################
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        # Read image        ################################################
        img0 = imread(img_name, True)

        # Load mask (if have) ##############
        mask_name = img_name.replace('_images', '_masks')
        if os.path.exists(mask_name):
            img_mask = cv2.imread(mask_name, 0)
            imagemaskinv = cv2.bitwise_not(img_mask)
            img0 = cv2.bitwise_and(img0, img0, mask=imagemaskinv)

        flip = False
        if self.aug:    # augment on cv2 image
            # augment brightness, saturation, contrast, hue
            img0 = Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))   # cv2 to PIL
            img0 = brightness_change(img0)
            img0 = cv2.cvtColor(np.array(img0), cv2.COLOR_RGB2BGR)          # PIL to cv2

            # Flip
            flip = np.random.randint(10) == 1

        img = preprocess_image(img0, flip=flip)
        img = torch.from_numpy(np.rollaxis(img, 2, 0))      # RGB, (3,H,W)

        # add gaussian noise on Tensor image
        if self.aug:
            scale = random.randint(0, 10)/100
            img = img + scale * torch.randn(img.shape)

        # normalize image
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)      # normalize image

        # ******************** For training and validation dataset ***********************
        if not self.test:           # Get mask and regression maps
            mask, regr = get_mask_and_regr(img0, labels, flip=flip)
            regr = np.rollaxis(regr, 2, 0)
            mask = torch.from_numpy(mask)
            regr = torch.from_numpy(regr)
            return img, mask, regr, img_name
        else:
            return img, img_name


if __name__ == "__main__":
    PATH = '../input/pku-autonomous-driving/'
    os.listdir(PATH)

    train = pd.read_csv(PATH + 'train.csv')
    test = pd.read_csv(PATH + 'sample_submission.csv')

    train_images_dir = PATH + 'train_images/{}.jpg'
    test_images_dir = PATH + 'test_images/{}.jpg'

    df_train, df_dev = train_test_split(train, test_size=0.1, random_state=42)
    df_test = test

    # Create dataset objects
    train_dataset = CarDataset(df_train, train_images_dir, aug=True)
    dev_dataset = CarDataset(df_dev, train_images_dir, aug=False)
    test_dataset = CarDataset(df_test, test_images_dir, aug=False)








