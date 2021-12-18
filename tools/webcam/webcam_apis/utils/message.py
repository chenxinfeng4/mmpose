# Copyright (c) OpenMMLab. All rights reserved.
import time
import uuid
import warnings
from typing import Dict, List, Optional

import numpy as np


class Message():
    """Message base class.

    All message class should inherit this class
    """

    def __init__(self, msg: str = '', data: Optional[Dict] = None):
        self.msg = msg
        self.data = data if data else {}
        self.route_info = []
        self.timestamp = time.time()
        self.id = uuid.uuid4()

    def update_route_info(self,
                          node=None,
                          node_name=None,
                          node_type=None,
                          info: Optional[Dict] = None):
        if node is not None:
            if node_name is not None or node_type is not None:
                warnings.warn(
                    '`node_name` and `node_type` will be overridden if node'
                    'is provided.')
            node_name = node.name
            node_type = node.__class__.__name__

        node_info = {'node': node_name, 'node_type': node_type, 'info': info}
        self.route_info.append(node_info)

    def set_route_info(self, route_info: List):
        self.route_info = route_info

    def merge_route_info(self, route_info: List):
        self.route_info += route_info
        self.route_info.sort(key=lambda x: x.get('timestamp', np.inf))

    def get_route_info(self) -> List:
        return self.route_info.copy()


class VideoEndingMessage(Message):
    """A special message to indicate the input video is ending."""


class FrameMessage(Message):
    """The message to store information of a video frame."""

    def __init__(self, img):
        super().__init__(data=dict(image=img))

    def get_image(self):
        return self.data.get('image', None)

    def set_image(self, img):
        self.data['image'] = img

    def add_detection_result(self, result, tag=None):
        if 'detection_results' not in self.data:
            self.data['detection_results'] = []
        self.data['detection_results'].append((tag, result))

    def get_detection_results(self, tag=None):
        if 'detection_results' not in self.data:
            return None
        if tag is None:
            results = [res for _, res in self.data['detection_results']]
        else:
            results = [
                res for _tag, res in self.data['detection_results']
                if _tag == tag
            ]
        return results

    def add_pose_result(self, result, tag=None):
        if 'pose_results' not in self.data:
            self.data['pose_results'] = []
        self.data['pose_results'].append((tag, result))

    def get_pose_results(self, tag=None):
        if 'pose_results' not in self.data:
            return None
        if tag is None:
            results = [res for _, res in self.data['pose_results']]
        else:
            results = [
                res for _tag, res in self.data['pose_results'] if _tag == tag
            ]
        return results

    def get_full_results(self):
        result_keys = ['detection_results', 'pose_results']
        results = {k: self.data[k] for k in result_keys}
        return results

    def set_full_results(self, results):
        self.data.update(results)
