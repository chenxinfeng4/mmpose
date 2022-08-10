# Copyright (c) OpenMMLab. All rights reserved.
"""
data = pickle.load(f)
nframe = len(data)
iframe_keypoints = data[0]
njoint = len(iframe_keypoints)
x, y, p = iframe_keypoints[:, 0], iframe_keypoints[:, 1], iframe_keypoints[:, 2]
"""

import argparse
import glob
import os
import warnings

import mmcv
import numpy as np

from mmpose.apis import inference_bottom_up_pose_model, init_pose_model
from mmpose.datasets import DatasetInfo


def parse_args():
    parser = argparse.ArgumentParser(description='MMSegementation video demo')
    parser.add_argument('videos', help='Video folder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args_out = args.videos
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.config, args.checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
    else:
        dataset_info = DatasetInfo(dataset_info)

    num_joint = len(dataset_info.keypoint_info)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    pose_nms_thr = 0.9
    # loop for each video
    videos = glob.glob(args.videos + '/*.mp4') + glob.glob(args.videos +
                                                           '/*.avi')
    out_pkls = [
        os.path.join(args_out,
                     os.path.split(f)[-1][:-4] + '.pkl') for f in videos
    ]

    for j, (video, out_pkl) in enumerate(zip(videos, out_pkls)):
        print('[{} in {}]: {}'.format(j + 1, len(videos),
                                      os.path.split(video)[-1]))
        video_reader = mmcv.VideoReader(video)
        outputs = []
        # loop each video frame
        for img in mmcv.track_iter_progress(video_reader):
            pose_results, returned_outputs = inference_bottom_up_pose_model(
                pose_model,
                img,
                dataset=dataset,
                dataset_info=dataset_info,
                pose_nms_thr=pose_nms_thr,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            if len(pose_results) == 0:
                keypoints = np.zeros((num_joint, 3))
            elif len(pose_results) == 1:
                keypoints = pose_results[0]['keypoints']
            else:
                #sort the pose_results by the max score
                pose_results = sorted(
                    pose_results, key=lambda x: x['score'], reverse=True)
                keypoints = pose_results[0]['keypoints']
            outputs.append(keypoints)
        # save to pkl
        mmcv.dump(outputs, out_pkl)


if __name__ == '__main__':
    main()
