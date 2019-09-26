import os
import imp
import tensorflow as tf
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

############Config#############################################################
GLOBAL_PATH = '/home/cyrus/Research/Waymo_Kitti_Adapter'
# path to waymo dataset "folder" (all the file in that folder will be converted)
DATA_PATH = GLOBAL_PATH + '/waymo_dataset/'
# path to save kitti dataset
LABEL_PATH = GLOBAL_PATH + '/kitti_dataset/label_2/'
IMAGE_PATH = GLOBAL_PATH + '/kitti_dataset/image_2/'
INDEX_LENGTH = 6

os.environ['PYTHONPATH'] = GLOBAL_PATH
m = imp.find_module('waymo_open_dataset', [GLOBAL_PATH])
imp.load_module('waymo_open_dataset', m[0], m[1], m[2])
###############################################################################

class Adapter:
    def __init__(self):
        # get all segment file name
        self.__file_names = [(DATA_PATH + i) for i in os.listdir(DATA_PATH)]
        # if not os.path.exists(LABEL_PATH):
        #     os.mkdir(GLOBAL_PATH + LABEL_PATH)

        self.__lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']
        self.__type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

    def cvt(self):
        tf.enable_eager_execution()
        frame_num = 0
        print("start converting ...")
        for file_name in self.__file_names:
            # read one frame
            dataset = tf.data.TFRecordDataset(file_name, compression_type='')
            for data in dataset:
                fp_label = open(LABEL_PATH + '/' + str(frame_num).zfill(INDEX_LENGTH) + '.txt', 'w+')
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))

                # store the image:
                img_path = IMAGE_PATH + '/' + str(frame_num).zfill(INDEX_LENGTH) + '.png'
                img = cv2.imdecode(np.frombuffer(frame.images[0].image, np.uint8), cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                plt.imsave(img_path, rgb_img,format='png')

                # parse the label
                # # context is the same for all frames in the same segment(file)
                # context = frame.context

                # preprocess bounding box data
                id_to_bbox = dict()
                for labels in frame.projected_lidar_labels:
                    for label in labels.labels:
                        bbox = [label.box.center_x - label.box.length / 2, label.box.center_y + label.box.width / 2,
                                label.box.center_x + label.box.length / 2, label.box.center_y - label.box.width / 2]
                        id_to_bbox[label.id] = bbox

                for obj in frame.laser_labels:
                    # caculate bounding box
                    bounding_box = None
                    id = obj.id
                    for lidar in self.__lidar_list:
                        if id + lidar in id_to_bbox:
                            bounding_box = id_to_bbox.get(id + lidar)
                    if bounding_box == None:
                        continue

                    my_type = self.__type_list[obj.type]
                    truncated = 0
                    occluded = 0
                    height = obj.box.height
                    width = obj.box.width
                    length = obj.box.length
                    x = obj.box.center_x
                    y = obj.box.center_y
                    z = obj.box.center_z
                    rotation_y = obj.box.heading
                    beta = math.atan2(x, z)
                    alpha = (rotation_y + beta - math.pi / 2) % (2 * math.pi)

                    # save the labels
                    line = my_type + ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(round(truncated, 2),
                                                                                            occluded,
                                                                                            round(alpha, 2),
                                                                                            round(bounding_box[0], 2),
                                                                                            round(bounding_box[1], 2),
                                                                                            round(bounding_box[2], 2),
                                                                                            round(bounding_box[3], 2),
                                                                                            round(height, 2),
                                                                                            round(width, 2),
                                                                                            round(length, 2),
                                                                                            round(x, 2),
                                                                                            round(y, 2),
                                                                                            round(z, 2),
                                                                                            round(rotation_y, 2))
                    # print(line)
                    # store the label
                    fp_label.write(line)
                frame_num += 1
                fp_label.close()
        print("finished ...")

    def image_show(self, data, name, layout, cmap=None):
        """Show an image."""
        plt.subplot(*layout)
        plt.imshow(tf.image.decode_jpeg(data), cmap=cmap)
        plt.title(name)
        plt.grid(False)
        plt.axis('off')

    def parse_range_image_and_camera_projection(self, frame):
        """Parse range images and camera projections given a frame.
        Args:
           frame: open dataset frame proto
        Returns:
           range_images: A dict of {laser_name,
             [range_image_first_return, range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
              camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
        """
        self.__range_images = {}
        camera_projections = {}
        range_image_top_pose = None
        for laser in frame.lasers:
            if len(laser.ri_return1.range_image_compressed) > 0:
                range_image_str_tensor = tf.decode_compressed(
                    laser.ri_return1.range_image_compressed, 'ZLIB')
                ri = open_dataset.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                self.__range_images[laser.name] = [ri]

                if laser.name == open_dataset.LaserName.TOP:
                    range_image_top_pose_str_tensor = tf.decode_compressed(
                        laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                    range_image_top_pose = open_dataset.MatrixFloat()
                    range_image_top_pose.ParseFromString(
                        bytearray(range_image_top_pose_str_tensor.numpy()))

                camera_projection_str_tensor = tf.decode_compressed(
                    laser.ri_return1.camera_projection_compressed, 'ZLIB')
                cp = open_dataset.MatrixInt32()
                cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
                camera_projections[laser.name] = [cp]
            if len(laser.ri_return2.range_image_compressed) > 0:
                range_image_str_tensor = tf.decode_compressed(
                    laser.ri_return2.range_image_compressed, 'ZLIB')
                ri = open_dataset.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                self.__range_images[laser.name].append(ri)

                camera_projection_str_tensor = tf.decode_compressed(
                    laser.ri_return2.camera_projection_compressed, 'ZLIB')
                cp = open_dataset.MatrixInt32()
                cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
                camera_projections[laser.name].append(cp)
        return self.__range_images, camera_projections, range_image_top_pose

    def plot_range_image_helper(self, data, name, layout, vmin=0, vmax=1, cmap='gray'):
        """Plots range image.
        Args:
          data: range image data
          name: the image title
          layout: plt layout
          vmin: minimum value of the passed data
          vmax: maximum value of the passed data
          cmap: color map
        """
        plt.subplot(*layout)
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(name)
        plt.grid(False)
        plt.axis('off')

    def get_range_image(self, laser_name, return_index):
        """Returns range image given a laser name and its return index."""
        return self.__range_images[laser_name][return_index]

    def show_range_image(self, range_image, layout_index_start=1):
        """Shows range image.
        Args:
          range_image: the range image data from a given lidar of type MatrixFloat.
          layout_index_start: layout offset
        """
        range_image_tensor = tf.convert_to_tensor(range_image.data)
        range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
        lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
        range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                      tf.ones_like(range_image_tensor) * 1e10)
        range_image_range = range_image_tensor[..., 0]
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]
        self.plot_range_image_helper(range_image_range.numpy(), 'range',
                                [8, 1, layout_index_start], vmax=75, cmap='gray')
        self.plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                                [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
        self.plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                                [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')

    def convert_range_image_to_point_cloud(self, frame, range_images, camera_projections, range_image_top_pose, ri_index=0):
        """Convert range images to point cloud.
        Args:
          frame: open dataset frame
           range_images: A dict of {laser_name,
             [range_image_first_return, range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
              camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
          ri_index: 0 for the first return, 1 for the second return.
        Returns:
          points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
          cp_points: {[N, 6]} list of camera projections of length 5
            (number of lidars).
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        lasers = sorted(frame.lasers, key=lambda laser: laser.name)
        points = []
        cp_points = []

        frame_pose = tf.convert_to_tensor(
            np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == open_dataset.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))

            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

        return points, cp_points

    def rgba(self, r):
        """Generates a color based on range.
        Args:
          r: the range value of a given point.
        Returns:
          The color for a given range
        """
        c = plt.get_cmap('jet')((r % 20.0) / 20.0)
        c = list(c)
        c[-1] = 0.5  # alpha
        return c

    def plot_image(self, camera_image):
        """Plot a cmaera image."""
        plt.figure(figsize=(20, 12))
        plt.imshow(tf.image.decode_jpeg(camera_image.image))
        plt.grid("off")

    def plot_points_on_image(self, projected_points, camera_image, rgba_func, point_size=5.0):
        """Plots points on a camera image.
        Args:
          projected_points: [N, 3] numpy array. The inner dims are
            [camera_x, camera_y, range].
          camera_image: jpeg encoded camera image.
          rgba_func: a function that generates a color from a range value.
          point_size: the point size.
        """
        self.plot_image(camera_image)

        xs = []
        ys = []
        colors = []

        for point in projected_points:
            xs.append(point[0])  # width, col
            ys.append(point[1])  # height, row
            colors.append(rgba_func(point[2]))

        plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")

if __name__ == '__main__':
    adapter = Adapter()
    adapter.cvt()