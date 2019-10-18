# Waymo_Kitti_Adapter
This is a tool converting [Waymo open dataset](https://waymo.com/open/) format to [Kitti dataset](http://www.cvlibs.net/datasets/kitti/) format.
> Author: **Yao Shao**
>
> Contact: **yshao998@gmail.com**
## Instruction
0. Follow the instructons in [QuickStart.md](https://github.com/Yao-Shao/Waymo_Kitti_Adapter/blob/master/QuickStart.md), clone the [waymo open dataset repo](https://github.com/waymo-research/waymo-open-dataset), build and test it. 
1. Clone this repo to your computer, then copy the files in `protocol buffer` folder and paste them into `waymo open dataset` folder.
2. Copy adapter.py to `waymo-od` folder. Open adapter.py and change the configurations at the top so that it suits to your own computer's path.
3. The folder tree may look like this, the downloaded waymo dataset should be in the folder named `waymo_dataset`, and the generated kitti dataset should be in the folder `kitti_dataset/`. Feel free to change them to your preferred path by rewriting the configurations in `adapter.py`.
``` 
.
├── adapter.py
├── waymo_open_dataset
│   ├── label_pb2.py
│   ├── label.proto
│   └── ...
├── waymo_dataset
│   └── frames
├── kitti_dataset
│   ├── calib
│   ├── image_0
│   ├── image_1
│   ├── image_2
│   ├── image_3
│   ├── image_4
│   ├── lidar
│   └── label
├── configure.sh
├── CONTRIBUTING.md
├── docs
├── LICENSE
├── QuickStart.md
├── README.md
├── tf
├── third_party
├── tutorial
└── WORKSPACE
```

4. Run adapter.py.

## Data specification

### Cameras

Waymo dataset contains five cameras:

``` 
FRONT = 0;
FRONT_LEFT = 1;
FRONT_RIGHT = 2;
SIDE_LEFT = 3;
SIDE_RIGHT = 4;
```

all the names below with post-fix 0-4 is corresponding to these five cameras.  

### Label

label_0 to label_4 contains label data for each camera and label_all fonder contain all the labels.

All in vehicle frame.

For each frame, here is the data specification:

```
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1   	camera_num	 only exist in label_all, the camera number which the object 
   					belongs to  
```

### Calib

```
P0-P4 : intrinsic matrix for each camera
R0_rect : rectify matrix
Tr_velo_to_cam_0 - Tr_velo_to_cam_4 : transformation matrix from vehicle frame to camera frame
```

### Image

```
image_0 - image_4 : images for each 
```

### Lidar

Point cloud in vehicle frame.

```
x y z intensity
```

For more details, see [readme.txt](https://github.com/Yao-Shao/Waymo_Kitti_Adapter/blob/master/KITTI/readme.txt) by KITTI.

## References

1. [Waymo open dataset](https://github.com/waymo-research/waymo-open-dataset)
2. [argoverse kitti adapter](https://github.com/yzhou377/argoverse-kitti-adapter)
