# tools/dist_train.sh res50_coco_ball_512x320_cam9.py 4
# python tools/train.py res50_coco_ball_512x320_cam9.py
_base_ = [
    'configs/_base_/default_runtime.py',
    'configs/_base_/datasets/coco_caliball.py'
]

load_from = 'work_dirs/res50_coco_ball_512x320_cam9/latest.pth'
total_epochs = 180

# params
evaluation = dict(interval=30, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
num_joints = 1
channel_cfg = dict(
    num_output_channels=num_joints,
    dataset_joints=num_joints,
    dataset_channel=[list(range(num_joints))],
    inference_channel=list(range(num_joints)))

# model settings
model = dict(
    type='TopDown',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=2048,
        num_deconv_layers=5,
        num_deconv_filters=(256, 128, 64, 32, 16),
        num_deconv_kernels=(4, 4, 4, 4, 4),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[512, 320],
    heatmap_size=[512, 320],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='ConcatDataset',
        datasets=[
            # dict(
            #     type='TopDownCocoDatasetTight',
            #     ann_file='data/ball/ball_1280x800x9_20221011_trainval.json',
            #     img_prefix='data/ball/ball_1280x800x9_20221011/',
            #     data_cfg=data_cfg,
            #     pipeline=train_pipeline,
            #     dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='TopDownCocoDatasetTight',
            #     ann_file='data/ball/ball_1280x800x9_20221012_trainval.json',
            #     img_prefix='data/ball/ball_1280x800x9_20221012/',
            #     data_cfg=data_cfg,
            #     pipeline=train_pipeline,
            #     dataset_info={{_base_.dataset_info}}),
            # dict(
            #     type='TopDownCocoDatasetTight',
            #     ann_file='data/ball/ball_1280x800x9_20230201_trainval.json',
            #     img_prefix='data/ball/ball_1280x800x9_20230201/',
            #     data_cfg=data_cfg,
            #     pipeline=train_pipeline,
            #     dataset_info={{_base_.dataset_info}}),
            dict(
                type='TopDownCocoDatasetTight',
                ann_file=
                'data/ball/ball_1280x800x9_20230504_checkerboard_trainval.json',
                img_prefix='data/ball/ball_1280x800x9_20230504_checkerboard/',
                data_cfg=data_cfg,
                pipeline=train_pipeline,
                dataset_info={{_base_.dataset_info}}),
        ],
    ),
    val=dict(
        type='TopDownCocoDatasetTight',
        ann_file='data/ball/ball_val.json',
        img_prefix='data/ball/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDatasetTight',
        ann_file='data/ball/ball_val.json',
        img_prefix='data/ball/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
