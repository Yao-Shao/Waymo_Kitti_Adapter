# Waymo_Kitti_Adapter
This is a tool converting [Waymo open dataset](https://waymo.com/open/) format to [Kitti dataset](http://www.cvlibs.net/datasets/kitti/) format.
> Author: **Yao Shao**
>
> Contact: **yshao998@gmail.com**
## Instruction
0. Follow the instructons in [waymo's tutorial](https://colab.sandbox.google.com/github/waymo-research/waymo-open-dataset/blob/r1.0/tutorial/tutorial.ipynb), clone the [waymo open dataset repo](https://github.com/waymo-research/waymo-open-dataset), build and test it. 
1. Clone this repo to your computer, then copy the files in `protocol buffer` folder and put them into `waymo open dataset` folder.
2. Copy adapter.py to waymo's repo. Open adapter.py and change the configurations at the top so that it suits to your own computer's path.
3. The folder tree may look like this, the downloaded waymo dataset should be in the folder named `waymo_dataset`, and the generated kitti dataset should be in the folder `kitti_dataset/label_2`. Feel free to change them to your preferred path by rewriting the configurations in `adapter.py`.
```
.
├── waymo-od
│   ├── adapter.py
│   ├── docs
│   ├── kitti_dataset
│   │   └── label_2
│   ├── README.md
│   ├── tf
│   ├── third_party
│   ├── tutorial
│   ├── waymo_dataset
│   │   └── frames
│   ├── waymo_open_dataset
│   │   ├── dataset_pb2.py
│   │   ├── label_pb2.py
│   │   └── ...
│   └── WORKSPACE
```
4. Run adapter.py.

## References
1. [Waymo open dataset](https://github.com/waymo-research/waymo-open-dataset)
2. [argoverse kitti adapter](https://github.com/yzhou377/argoverse-kitti-adapter)
