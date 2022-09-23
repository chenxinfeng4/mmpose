keypoint_info = dict({
    0:
    dict(name='Nose', id=0, color=[51, 153, 255], type='upper', swap=''),
    1:
    dict(name='EarL', id=1, color=[51, 153, 255], type='upper', swap='EarR'),
    2:
    dict(name='EarR', id=2, color=[51, 153, 255], type='upper', swap='EarL'),
    3:
    dict(name='Neck', id=3, color=[51, 153, 255], type='upper', swap=''),
    4:
    dict(name='Back', id=4, color=[51, 153, 255], type='upper', swap=''),
    5:
    dict(name='Tail', id=5, color=[51, 153, 255], type='upper', swap=''),
    6:
    dict(
        name='ForeShoulderL',
        id=6,
        color=[51, 153, 255],
        type='upper',
        swap='ForeShoulderR'),
    7:
    dict(
        name='ForePowL',
        id=7,
        color=[51, 153, 255],
        type='upper',
        swap='ForePowR'),
    8:
    dict(
        name='ForeShoulderR',
        id=8,
        color=[51, 153, 255],
        type='upper',
        swap='ForeShoulderL'),
    9:
    dict(
        name='ForePowR',
        id=9,
        color=[51, 153, 255],
        type='upper',
        swap='ForePowL'),
    10:
    dict(
        name='BackShoulderL',
        id=10,
        color=[51, 153, 255],
        type='upper',
        swap='BackShoulderR'),
    11:
    dict(
        name='BackPowL',
        id=11,
        color=[51, 153, 255],
        type='upper',
        swap='BackPowR'),
    12:
    dict(
        name='BackShoulderR',
        id=12,
        color=[51, 153, 255],
        type='upper',
        swap='BackShoulderL'),
    13:
    dict(
        name='BackPowR',
        id=13,
        color=[51, 153, 255],
        type='upper',
        swap='BackPowL')
})
skeleton_info = dict({
    0:
    dict(link=('Nose', 'EarL'), id=0, color=[0, 255, 0]),
    1:
    dict(link=('Nose', 'EarR'), id=1, color=[0, 255, 0]),
    2:
    dict(link=('Neck', 'EarL'), id=2, color=[255, 128, 0]),
    3:
    dict(link=('Neck', 'EarR'), id=3, color=[255, 128, 0]),
    4:
    dict(link=('BackShoulderL', 'BackPowL'), id=4, color=[51, 153, 255]),
    5:
    dict(link=('BackShoulderR', 'BackPowR'), id=5, color=[51, 153, 255]),
    6:
    dict(link=('ForeShoulderL', 'ForePowL'), id=6, color=[51, 153, 255]),
    7:
    dict(link=('ForeShoulderR', 'ForePowR'), id=7, color=[51, 153, 255]),
    8:
    dict(link=('ForeShoulderL', 'BackShoulderL'), id=8, color=[0, 255, 0]),
    9:
    dict(link=('ForeShoulderR', 'BackShoulderR'), id=9, color=[255, 128, 0]),
    10:
    dict(link=('Back', 'ForeShoulderL'), id=10, color=[0, 255, 0]),
    11:
    dict(link=('Back', 'ForeShoulderR'), id=11, color=[255, 128, 0]),
    12:
    dict(link=('Back', 'BackShoulderL'), id=12, color=[51, 153, 255]),
    13:
    dict(link=('Back', 'BackShoulderR'), id=13, color=[51, 153, 255]),
    14:
    dict(link=('Tail', 'BackShoulderL'), id=14, color=[51, 153, 255]),
    15:
    dict(link=('Tail', 'BackShoulderR'), id=15, color=[51, 153, 255]),
})
joint_weights = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
],
sigmas = [
    0.026, 0.025, 0.025, 0.035, 0.055, 0.025, 0.050, 0.050, 0.050, 0.050,
    0.050, 0.050, 0.050, 0.050
]
dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author=
        'Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Dollar, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/'),
    keypoint_info=keypoint_info,
    skeleton_info=skeleton_info,
    joint_weights=joint_weights,
    sigmas=sigmas)
