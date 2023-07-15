# the calibration ball
keypoint_info = dict({
    0:
    dict(name='ball', id=0, color=[51, 153, 255], type='upper', swap=''),
})
skeleton_info = dict()
joint_weights = [1.0],
sigmas = [0.014]
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
