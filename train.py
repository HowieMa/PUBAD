import torch
import torch.nn as nn
import numpy as np
import json
import os
import pandas as pd
import cv2

from torch.utils.data import DataLoader


from utils.loss import centernet_loss
from utils.visual import visualize
from utils.metric import get_mAP
from dataset import CarDataset, train_test_split, extract_coords, coords2str, imread
from dla34_dcn import DLA34_DCN


learning_rate = 1e-4
lambda_size = 10
down_ratio = 1

pretrain = False
PATH = '../pku-autonomous-driving/'
BATCH_SIZE = 8
model_save_path = 'ckpt/'
epochs = 80

cuda = torch.cuda.is_available()
torch.cuda.set_device(0)

# ####################### PUBD data prepare #######################
train_data = pd.read_csv(PATH + 'train.csv')
test_data = pd.read_csv(PATH + 'sample_submission.csv')

train_images_dir = PATH + 'train_images/{}.jpg'
test_images_dir = PATH + 'test_images/{}.jpg'

df_train, df_dev = train_test_split(train_data, test_size=0.1, random_state=0)
df_test = test_data

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)


def train(model, learning_rate, epochs, train_loader, val_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    down_ratio = 1
    if pretrain:
        print('Load pretrain model ...')
        checkpoints = torch.load(os.path.join(model_save_path, 'best_model_map.tar'))
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        loss = checkpoints['loss']
        eval_loss, eval_map = eval_model(model, val_loader)
        print('Pretrain model eval mAP: {}'.format(eval_map))

    curr_lr = learning_rate
    best_epo_m = -1
    best_map = -1

    total_step = len(train_loader)

    for epoch in range(epochs):
        # ################## One Epoch ####################################
        print('.......... EPOCH [{}/{}] ..........'.format(epoch, epochs))
        model.train()
        for step, (img_batch, mask_batch, reg_batch, img_name) in enumerate(train_loader):
            if cuda:
                img_batch = img_batch.cuda()        # Tensor size:(B,3,H,W)
                mask_batch = mask_batch.cuda()      # Tensor size:(B,H,W))    mask,w,h
                reg_batch = reg_batch.cuda()        # Tensor size:(B,7,H,W)    mask,w,h

            optimizer.zero_grad()
            output = model(img_batch)  # Tensor size:(B,1+2,72,128)    mask,w,h

            mask_loss, pose_loss = centernet_loss(output, mask_batch, reg_batch)
            loss = down_ratio * mask_loss + lambda_size * pose_loss
            loss.backward()

            optimizer.step()

            if step % 50 == 0:
                print('STEP [{}/{}] ....'.format(step, total_step, ))
                print('Mask Loss: {:.6f}\t Pose Loss: {:.6f}\t'.format(mask_loss.item(),
                                                                       pose_loss.item()))
        # Evaluate model after one epoch #################################################
        print('Start eval ....')
        if epoch < 3:
            print('Continue ...')
            continue

        eval_loss, eval_map = eval_model(model, val_loader)
        print('Epoch {} Validation Loss: {}  mAP : {}'.format(epoch, eval_loss, eval_map))

        if eval_map > best_map:
            best_map = eval_map
            best_epo_m = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            },
            os.path.join(model_save_path, 'best_model_map.tar'))
        print("So far Best mAP epoch:{}, best validation mAP:{}\n".format(best_epo_m, best_map))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        },
            os.path.join(model_save_path, 'checkpoints_{}.tar'.format(epoch)))
        # Update Learning rate ###########################################################
        if epoch == 40:
            curr_lr = curr_lr * 0.1
            update_lr(optimizer, curr_lr)

        if epoch % 30 == 0:
            down_ratio = down_ratio * 0.1


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def eval_model(model, eval_loader, epoch=0):
    total_loss = 0.0
    total_step = len(eval_loader)
    predictions = []

    os.makedirs('ckpt/epoch_{}'.format(epoch), exist_ok=True)
    model.eval()
    with torch.no_grad():
        for step, (img_batch, mask_batch, reg_batch, img_name) in enumerate(eval_loader):
            if cuda:
                img_batch = img_batch.cuda()        # Tensor size:(B,3,H,W)
                mask_batch = mask_batch.cuda()      # Tensor size:(B,H,W))    mask,w,h
                reg_batch = reg_batch.cuda()        # Tensor size:(B,7,H,W)    mask,w,h

            output = model(img_batch)  # Tensor size:(B,1+2,72,128)    mask,w,h

            # ################### calculate validation loss ####################
            mask_loss, pose_loss = centernet_loss(output, mask_batch, reg_batch)
            loss = down_ratio * mask_loss + lambda_size * pose_loss
            total_loss += loss.item()

            # ################### calculate predict coords ####################
            output = output.cpu().numpy()
            for b in range(output.shape[0]):
                out = output[b]
                coords = extract_coords(out)
                s = coords2str(coords)
                predictions.append(s)

                # ################## save visualization image ##################
                # img_0 = imread(img_name[b])
                # visual_img = visualize(img_0, coords)
                # save_dir = 'ckpt/epoch_{}/{}'.format(epoch, img_name[b].split('/')[-1])
                # cv2.imwrite(save_dir, visual_img)

    # get mAP
    valid_res = df_dev.copy()
    valid_res['PredictionString'] = predictions
    valid_res.to_csv('ckpt/valid_epoch{}_predictions.csv'.format(epoch), index=False)

    map_score = get_mAP(PATH + 'train.csv', 'ckpt/valid_epoch{}_predictions.csv'.format(epoch))

    return total_loss/total_step, map_score


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def submission(model):
    predictions = []
    model.eval()
    coords_res = {}
    for (img_batch, img_name) in test_loader:
        with torch.no_grad():
            if cuda:
                img_batch = img_batch.cuda()        # Tensor size:(B,3,H,W)
            output = model(img_batch)
        output = output.cpu().numpy()

        for b in range(output.shape[0]):
            out = output[b]
            coords = extract_coords(out)

            coords_res[img_name[b]] = coords

            s = coords2str(coords)
            predictions.append(s)

            # # ################## save visualization image ##################
            # img_0 = imread(img_name[b])
            # visual_img = visualize(img_0, coords)
            # os.makedirs('ckpt/test', exist_ok=True)
            # save_dir = 'ckpt/test/{}'.format(img_name[b].split('/')[-1])
            # cv2.imwrite(save_dir, visual_img)

    test = pd.read_csv(PATH + 'sample_submission.csv')
    test['PredictionString'] = predictions
    test.to_csv('predictions.csv', index=False)

    # For each prediction, save one json profile to ensemble
    json.dump(coords_res, open('pred.json'), cls=MyEncoder, sort_keys=True, indent=4)


if __name__ == "__main__":

    # Create dataset objects
    train_dataset = CarDataset(df_train, train_images_dir, aug=True)
    dev_dataset = CarDataset(df_dev, train_images_dir, aug=True)
    test_dataset = CarDataset(df_test, test_images_dir, aug=False, test=True)

    print('Total image in train data : {}'.format(len(train_dataset)))
    print('Total image in valid data : {}'.format(len(dev_dataset)))
    print('Total image in test data : {}'.format(len(test_dataset)))

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = DLA34_DCN(num_layers=34, heads={'hm': 1, 'wh': 7})
    if cuda:
        model = model.cuda()

    train(model, learning_rate, epochs, train_loader, dev_loader)

    print('TESTING ============================>')
    checkpoints = torch.load(os.path.join(model_save_path, 'best_model_map.tar'))
    model.load_state_dict(checkpoints['model_state_dict'])
    submission(model)











