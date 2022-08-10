# Copyright (c) OpenMMLab. All rights reserved.
# %%
from typing import OrderedDict

import matplotlib.pylab as plt
import numpy as np
from pycocotools.coco import COCO

# %%
cocoGt = '/home/liying_lab/chenxinfeng/DATA/mmpose/data/rat_keypoints/2021-11-02-bwrat_side6-kp_val.json'
cocoDt = '/home/liying_lab/chenxinfeng/DATA/mmpose/out_work_dir/result_keypoints.json'
# cocoDt = '/home/liying_lab/chenxinfeng/DATA/mmpose/data/rat_keypoints/extract_2021-11-02-bwrat_side6-kp_val/result_keypoints.json'
cocoGt = COCO(cocoGt)

cocoDt = cocoGt.loadRes(cocoDt)

# %% one image, one class, one instance for cocoDt
imgIdsDt = list(cocoDt.imgs.keys())
imgIdsGt = list(cocoGt.imgs.keys())
assert set(imgIdsDt) == set(imgIdsGt), 'imgsId not equal'

annIdsDt = []
for imgId in imgIdsDt:
    annIds = cocoDt.getAnnIds(imgIds=imgId)
    assert len(annIds) >= 1, 'no annIds for imgId: {}'.format(imgId)
    if len(annIds) > 1:
        anns = cocoDt.loadAnns(annIds)
        scores = [ann['score'] for ann in anns]
        max_score_id = np.argmax(scores)
        annIds = [annIds[max_score_id]]
    annIdsDt.append(annIds[0])

    annIdsGt = cocoGt.getAnnIds(imgIds=imgId)
    assert len(annIdsGt) == 1, 'no or more annIds for imgId: {}'.format(imgId)

# %%
DEFAULT_PCK_THRESHOLD = 0.1


def pck_eval(cocoGt, cocoDt, imgIdsDt, annIdsDt):
    # one image one class one instance
    dist_list = []
    dist_list_normalized = []
    p_pos = []
    p_neg = []
    for imgIdDt, annIdDt in zip(imgIdsDt, annIdsDt):
        annDt = cocoDt.loadAnns(annIdDt)
        annGt = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=imgIdDt))
        assert len(annDt) == len(
            annGt) == 1, 'one image one class one instance'
        annDt = annDt[0]
        annGt = annGt[0]
        assert annDt['category_id'] == annGt[
            'category_id'], 'category_id not equal'

        prediction_keypoints = np.array(annDt['keypoints']).reshape(-1, 3)
        annotation_keypoints = np.array(annGt['keypoints']).reshape(-1, 3)

        ignoreGt = annotation_keypoints[:, 2] == 0
        annotation_keypoints[ignoreGt, :2] = np.nan
        p_pos.append(prediction_keypoints[~ignoreGt, 2])
        p_neg.append(prediction_keypoints[ignoreGt, 2])
        bboxGt = annGt['bbox']
        bboxw, bboxh = bboxGt[2], bboxGt[3]
        torso = np.linalg.norm([bboxw, bboxh])
        torso = np.max([torso, 100])  #min size is 100

        dist_keypoints = np.linalg.norm(
            prediction_keypoints[:, :2] - annotation_keypoints[:, :2], axis=1)
        dist_normalized = dist_keypoints / torso
        dist_list.append(dist_keypoints)
        dist_list_normalized.append(dist_normalized)

    dist_list = np.array(dist_list)
    dist_list_normalized = np.array(dist_list_normalized)
    # concat p_pos, p_neg along axis=0
    p_pos = np.concatenate(p_pos, axis=0)
    p_neg = np.concatenate(p_neg, axis=0)
    return dist_list, dist_list_normalized, p_pos, p_neg


def roc_curve(p_pos, p_neg):
    thr = np.linspace(0, 1.01, 100)
    tp = np.zeros_like(thr)
    fp = np.zeros_like(thr)
    tn = np.zeros_like(thr)
    fn = np.zeros_like(thr)
    for i, t in enumerate(thr):
        tp[i] = (p_pos >= t).sum()
        fp[i] = (p_neg >= t).sum()
        tn[i] = (p_neg < t).sum()
        fn[i] = (p_pos < t).sum()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    return fpr, tpr


dist_list, dist_list_normalized, p_pos, p_neg = pck_eval(
    cocoGt, cocoDt, imgIdsDt, annIdsDt)
fpr, tpr = roc_curve(p_pos, p_neg)
indVis = ~np.isnan(dist_list)
dist_list_vis = dist_list[indVis]
dist_list_normalized_vis = dist_list_normalized[indVis]
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
plt.figure(figsize=(8, 8))
plt.hist(p_pos, range=(0, 1), bins=50)
plt.hist(p_neg, range=(0, 1), bins=50)
plt.ylabel('Frequency')
plt.legend(['positive', 'negative'])

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
auc_area = abs(np.trapz(tpr, x=fpr))
plt.text(
    0.6,
    0.1,
    'AUC = {:.3f}'.format(auc_area),
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=12)
plt.xlim([0, 1])
plt.ylim([0, 1])

# %%
THR = [0.05, 0.1, 0.2]
for thr in THR:
    presion = np.mean(dist_list_normalized_vis[:] < thr)
    print('PCK@{:.2f}, presion: {:.2f}'.format(thr, presion))

dist_list_median = np.median(dist_list_vis)
print('distance pixel median: {:.2f}'.format(dist_list_median))

dist_list_normalized_median = np.median(dist_list_normalized_vis)
print('distance normalized median: {:.2f}'.format(dist_list_normalized_median))
# %%
body_mosaic = [['head', 'ears', 'neck'], ['back', 'tail', 'Fore-shouders'],
               ['Back-shouders', 'Fore-paws', 'Back-paws']]
fig = plt.figure(constrained_layout=True, figsize=(10, 10))
ax_dict = fig.subplot_mosaic(body_mosaic)

body_quary_ind = [['head', 0], ['ears', [1, 2]], ['neck', 3], ['back', 4],
                  ['tail', 5], ['Fore-shouders', [6, 8]],
                  ['Back-shouders', [10, 12]], ['Fore-paws', [7, 9]],
                  ['Back-paws', [11, 13]]]

# mean distance error
for bodyname, ind in body_quary_ind:
    d = dist_list[:, ind].flatten()
    d_mean = np.nanmean(d)
    ax = ax_dict[bodyname]
    ax.hist(d, range=(0, 40), bins=20)
    ax.set_title(bodyname)
    ax.plot([d_mean, d_mean], [0, 80], 'k--')

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

# %%
body_mosaic = [['head', 'ear L', 'ear R', 'neck'],
               ['back', 'tail', 'Fore-shouder L', 'Fore-shouder R'],
               [
                   'Back-shouder L', 'Back-shouder R', 'Fore-paw L',
                   'Fore-paw R'
               ], ['Back-paw L', 'Back-paw R', 'A', 'B']]
fig = plt.figure(constrained_layout=True, figsize=(10, 10))
ax_dict = fig.subplot_mosaic(body_mosaic)

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

# %%
