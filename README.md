# Waymo_Kitti_Adapter
This is a tool converting [Waymo open dataset](https://waymo.com/open/) format to [Kitti dataset](http://www.cvlibs.net/datasets/kitti/) format.
- Author: Yao Shao
- Contact: yshao998@gmail.com 
## Instruction
0. follow the instructons in [waymo's tutorial](https://colab.sandbox.google.com/github/waymo-research/waymo-open-dataset/blob/r1.0/tutorial/tutorial.ipynb), clone the [waymo open dataset repo](https://github.com/waymo-research/waymo-open-dataset), build and test it. 
1. clone this repo to your computer. copy the files in $protocol buffer$ folder and put them into $waymo open dataset$ folder.
2. copy adapter.py to waymo's repo, open adapter.py, change the configurations at the top so that it suits to your own computer's path.
3. run adapter.py.
## References
1. [Waymo open dataset](https://github.com/waymo-research/waymo-open-dataset)
2. [argoverse kitti adapter](https://github.com/yzhou377/argoverse-kitti-adapter)
