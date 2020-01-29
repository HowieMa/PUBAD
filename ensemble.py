import pandas as pd
import numpy as np
import json
import os
from dataset import imread
from utils.visual import visualize
import cv2


DISTANCE_THRESH_CLEAR = 2
PATH = '../pku-autonomous-driving/'

#
tmp_data1 = json.load(open('tmp-large-0114-077.json'))
tmp_data2 = json.load(open('pred.json'))


def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)


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


coords_res = {}
predictions = []

test = pd.read_csv(PATH + 'sample_submission.csv')
names_id = list(test['ImageId'])

for im_name in names_id:
    im_name = '../pku-autonomous-driving/test_images/' + im_name + '.jpg'
    # get result from each dict #####################
    coords1 = tmp_data1[im_name]
    coords2 = tmp_data2[im_name]

    # merge result
    coords = coords1 + coords2
    coords = clear_duplicates(coords)

    coords_res[im_name] = coords        # final

    s = coords2str(coords)
    predictions.append(s)

    # ################## save visualization image ##################
    img_0 = imread(im_name)
    visual_img = visualize(img_0, coords)
    os.makedirs('ckpt/test', exist_ok=True)
    save_dir = 'ckpt/test/{}'.format(im_name.split('/')[-1])
    cv2.imwrite(save_dir, visual_img)


json.dump(coords_res, open('cmb.json','w'), cls=MyEncoder, sort_keys=True, indent=4)
test = pd.read_csv(PATH + 'sample_submission.csv')
test['PredictionString'] = predictions
test.to_csv('predictions-sample.csv', index=False)

