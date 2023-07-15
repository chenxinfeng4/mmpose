# tools/dist_train.sh res50_coco_bwcom2d_384x256_cam9.py 4
# python tools/train.py res50_coco_bwcom2d_384x256_cam9.py
# python -m lilab.mmpose_dev.a2_convert_mmpose2onnx res50_coco_bwcom2d_384x256_cam9.py --full --dynamic --monocolor
_base_ = [
    'configs/_base_/default_runtime.py',
    'configs/_base_/datasets/coco_bwcom2d.py'
]

load_from = 'work_dirs/res50_coco_bwcom2d_384x256_cam9/latest.pth'
# load_from = None
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
num_joints = 2
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
    image_size=[384, 256],
    heatmap_size=[384, 256],
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
        type='TopDownGetRandomScaleRotation', rot_factor=20, scale_factor=0.1),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=8),
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

mmdet_imgdir = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats/'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='TopDownCocoDatasetTight',
                ann_file=
                'data/bw_com2d/bw_rat_1280x800_20220509_OXTR_trainval_com2d.json',
                img_prefix=mmdet_imgdir + 'bw_rat_1280x800_20220509_OXTR/',
                data_cfg=data_cfg,
                pipeline=train_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='TopDownCocoDatasetTight',
                ann_file=
                'data/bw_com2d/bw_rat_1280x800_20230525_LTZ_trainval_com2d.json',
                img_prefix=mmdet_imgdir + 'bw_rat_1280x800_20230525_LTZ/',
                data_cfg=data_cfg,
                pipeline=train_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='TopDownCocoDatasetTight',
                ann_file=
                'data/bw_com2d/bw_rat_1280x800_20230625_WT_trainval_com2d.json',
                img_prefix=mmdet_imgdir + 'bw_rat_1280x800_20230625_WT/',
                data_cfg=data_cfg,
                pipeline=train_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='TopDownCocoDatasetTight',
                ann_file=
                'data/bw_com2d/bw_rat_1280x800x9_20230209_VPA_trainval_com2d.json',
                img_prefix=mmdet_imgdir + 'bw_rat_1280x800x9_20230209_VPA/',
                data_cfg=data_cfg,
                pipeline=train_pipeline,
                dataset_info={{_base_.dataset_info}}),
            dict(
                type='TopDownCocoDatasetTight',
                ann_file=
                'data/bw_com2d/bw_rat_1280x800x9_20230608_zyq_trainval_com2d.json',
                img_prefix=mmdet_imgdir + 'bw_rat_1280x800x9_20230608_zyq/',
                data_cfg=data_cfg,
                pipeline=train_pipeline,
                dataset_info={{_base_.dataset_info}}),
        ],
    ),
    val=dict(
        type='TopDownCocoDatasetTight',
        ann_file=
        'data/bw_com2d/bw_rat_1280x800_20230625_WT_trainval_com2d.json',
        img_prefix=mmdet_imgdir + 'bw_rat_1280x800_20230625_WT/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDatasetTight',
        ann_file=
        'data/bw_com2d/bw_rat_1280x800_20230625_WT_trainval_com2d.json',
        img_prefix=mmdet_imgdir + 'bw_rat_1280x800_20230625_WT/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
