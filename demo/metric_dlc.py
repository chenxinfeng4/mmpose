# Copyright (c) OpenMMLab. All rights reserved.
# %%
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools.coco import COCO

basedir = '/home/liying_lab/chenxinfeng/deeplabcut-project/bwrat_28kpt-cxf-2022-02-25/labeled-data/2021-11-02-bwrat_side6-kp_trainval/val'
coco_json = '/home/liying_lab/chenxinfeng/DATA/mmpose/data/rat_keypoints/2021-11-02-bwrat_side6-kp_trainval.json'

gt_file = osp.join(basedir, 'CollectedData_cxf_val.h5')
det_file = osp.join(basedir,
                    'valDLC_resnet101_bwrat_28kptFeb25shuffle1_1030000.h5')
imageindex_file = osp.join(basedir, 'filenames.txt')
# %%
df_gt = pd.read_hdf(gt_file, 'df_with_missing')
df_det = pd.read_hdf(det_file, 'df_with_missing')
# %%
with open(imageindex_file, 'r') as f:
    imageindex = f.readlines()

df_gt_index = df_gt.index.values
imgs_full = [osp.dirname(df_gt.index[0]) + '/' + x.strip() for x in imageindex]

df_gt_resort = df_gt.reindex(imgs_full)

df_gt_array = df_gt_resort.values
df_det_array = df_det.values
(n_gt, n_gtpt) = df_gt_array.shape
(n_det, n_detpt) = df_det_array.shape
assert n_gt == n_det
assert n_detpt / n_gtpt == 3 / 2

# %% remove the columns of df_det_array every the third column
df_det_x = df_det_array[:, ::3]
df_det_y = df_det_array[:, 1::3]
df_det_p = df_det_array[:, 2::3]
df_gt_x = df_gt_array[:, ::2]
df_gt_y = df_gt_array[:, 1::2]

# %% load coco
coco = COCO(coco_json)
imgs = coco.imgs
imgs_dict = {x['file_name']: x['id'] for x in imgs.values()}
imgs_nake_base = [
    osp.splitext(osp.basename(img_full))[0] for img_full in imgs_full
]

torsor_b = []
torsor_w = []
torsor_min = 100
for img_nake_base in imgs_nake_base:
    rat_img = img_nake_base + '_rat_black.jpg'
    if rat_img in imgs_dict:
        bbox = coco.loadAnns(coco.getAnnIds(imgs_dict[rat_img]))[0]['bbox']
        torsor = np.linalg.norm(bbox[2:])
        torsor_b.append(torsor)
    else:
        torsor_b.append(0)

    rat_img = img_nake_base + '_rat_white.jpg'
    if rat_img in imgs_dict:
        bbox = coco.loadAnns(coco.getAnnIds(imgs_dict[rat_img]))[0]['bbox']
        torsor = np.linalg.norm(bbox[2:])
        torsor_w.append(torsor)
    else:
        torsor_w.append(0)

torsor_b = np.array(torsor_b)
torsor_w = np.array(torsor_w)
torsor_b[torsor_b < torsor_min] = torsor_min
torsor_w[torsor_w < torsor_min] = torsor_min

# %%
torsor_b14 = np.repeat(torsor_b[:, np.newaxis], 14, axis=1)
torsor_w14 = np.repeat(torsor_w[:, np.newaxis], 14, axis=1)
torsor_bw = np.concatenate((torsor_b14, torsor_w14), axis=1)
# %%
kpt_dx = df_det_x - df_gt_x
kpt_dy = df_det_y - df_gt_y
kpt_d = np.sqrt(kpt_dx**2 + kpt_dy**2)

dist_list = np.concatenate((kpt_d[:, 0:14], kpt_d[:, 14:28]), axis=0)
torsor_bw_list = np.concatenate((torsor_bw[:, 0:14], torsor_bw[:, 14:28]),
                                axis=0)
dist_list_vis = dist_list[~np.isnan(dist_list)]
dist_list_normalized = dist_list / torsor_bw_list
dist_list_normalized_vis = dist_list_normalized[~np.isnan(dist_list_normalized
                                                          )]

# %%
# %% histogram
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.hist(dist_list_vis, range=(0, 20), bins=50)
plt.ylabel('Frequency')
plt.xlabel('Pixel distance')
plt.title('DEEPLABCUT on b/w rats, train(n)=533, test(n)=179')
plt.subplot(212)
plt.hist(dist_list_normalized_vis, range=(0, 0.5), bins=50)
plt.ylabel('Frequency')
plt.xlabel('Normalized distance to body length')

# %%
THR = [0.05, 0.1, 0.2]
for thr in THR:
    presion = np.mean(dist_list_normalized_vis[:] < thr)
    print('PCK@{:.2f}, presion: {:.2f}'.format(thr, presion))

dist_list_median = np.median(dist_list_vis)
print('distance pixel median: {:.2f}'.format(dist_list_median))

dist_list_normalized_median = np.median(dist_list_normalized_vis)
print('distance normalized median: {:.2f}'.format(dist_list_normalized_median))

dist_list_mean = np.mean(dist_list_vis)
print('distance pixel mean: {:.2f}'.format(dist_list_mean))

dist_list_normalized_mean = np.mean(dist_list_normalized_vis)
print('distance normalized mean: {:.2f}'.format(dist_list_normalized_mean))
# %%
body_mosaic = [['head', 'ear L', 'ear R', 'neck'],
               ['back', 'tail', 'Fore-shouder L', 'Fore-shouder R'],
               [
                   'Back-shouder L', 'Back-shouder R', 'Fore-paw L',
                   'Fore-paw R'
               ], ['Back-paw L', 'Back-paw R', 'A', 'B']]

body_quary_ind = [['head', 0], ['ear L', 1], ['ear R', 2], ['neck', 3],
                  ['back', 4], ['tail', 5], ['Fore-shouder L', [6]],
                  ['Fore-shouder R', [8]], ['Back-shouder L', [10]],
                  ['Back-shouder R', [12]], ['Fore-paw L', [7]],
                  ['Fore-paw R', [9]], ['Back-paw L', [11]],
                  ['Back-paw R', [13]]]

# pck@0.05 precision
fig = plt.figure(constrained_layout=True, figsize=(10, 10))
ax_dict = fig.subplot_mosaic(body_mosaic)

for bodyname, ind in body_quary_ind:
    d = dist_list_normalized[:, ind].flatten()
    d = d[~np.isnan(d)]
    pck_05 = np.mean(d <= 0.05).round(2)
    ax = ax_dict[bodyname]
    ax.hist(d, range=(0, 0.2), bins=20)
    ax.set_title(bodyname)
    ax.plot([0.05, 0.05], [0, 80], 'k--')
    ax.text(
        0.04,
        70,
        pck_05,
        horizontalalignment='right',
        verticalalignment='center',
        fontsize=12)
