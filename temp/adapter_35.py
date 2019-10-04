import os
import pcl
import pickle

path = ''
file_name = os.listdir(path)

GLOBAL_PATH = '/home/cyrus/Research/Waymo_Kitti_Adapter'
PYTHON_PATH = '/usr/bin/python3.6'
# path to waymo dataset "folder" (all the file in that folder will be converted)
DATA_PATH = GLOBAL_PATH + '/waymo_dataset'
# path to save kitti dataset
KITTI_PATH = GLOBAL_PATH + '/kitti_dataset'

# do not change
LABEL_PATH = KITTI_PATH + '/label'
IMAGE_PATH = KITTI_PATH + '/image_'
CALIB_PATH = KITTI_PATH + '/calib'
LIDAR_PATH = KITTI_PATH + '/lidar'
INDEX_LENGTH = 6    # max indexing length
IMAGE_FORMAT = 'png'

for i in file_name:
    with open(file_name, 'rb') as f:
        point_cloud = pickle.load(f)
    pc_1 = pcl.PointCloud()
    pc_1.from_array(point_cloud)
    pc_path_1 = LIDAR_PATH + '/' + i.split('/')[-1].split('.')[0] + '.pcd'
    pcl.save(pc_1, pc_path_1)
