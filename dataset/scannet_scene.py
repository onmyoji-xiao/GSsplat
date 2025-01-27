import os
import numpy as np
import cv2
import glob
from skimage.io import imread
from PIL import Image
from natsort import natsorted

# 320,240
INTRINSIC = np.array([[288.7955, 0.0, 159.4525],
                      [0.0, 289.365, 121.342],
                      [0.0, 0.0, 1.0]])


class ScannetScene(object):
    def __init__(self, root_dir, scene_name, use_depth=True,is_train=True):
        self.scene_name = scene_name
        self.root_dir = f'{root_dir}/{scene_name}'
        self.intrinsic = INTRINSIC
        self.use_depth = use_depth

        self.h, self.w = 240, 320

        rgb_paths = [x for x in glob.glob(os.path.join(self.root_dir, "color", "*")) if
                     (x.endswith(".jpg") or x.endswith(".png"))]
        self.rgb_paths = natsorted(rgb_paths)

        depth_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "depth", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        self.depth_paths = natsorted(depth_paths)

        label_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "label-filt", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        self.label_paths = natsorted(label_paths)

        pose_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "pose", "*")) if x.endswith(".txt")]
        self.pose_paths = natsorted(pose_paths)

        self.img_ids = []
        skip_num = 0
        for i, rgb_path in enumerate(self.rgb_paths):
            if is_train and np.isinf(np.sum(self.get_pose(i))):
                skip_num += 1
                continue
            self.img_ids.append(i)
        # if skip_num > 0:
        #     print(f'{scene_name} skip inf num {skip_num}')

    def get_image(self, img_id):
        img = cv2.imread(self.rgb_paths[img_id])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_depth(self, img_id):
        img = Image.open(self.depth_paths[img_id])
        depth = np.array(img, dtype=np.float32) / 1000.0  # mm to m
        # depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_mask(self, lbimg):
        mask = np.ones([self.h, self.w], bool)
        mask = mask * (lbimg != 0) * (lbimg != 1)
        return mask

    def get_depth_range(self):
        return np.asarray((0.1, 10.0), np.float32)

    def get_pred_depth(self, img_id):
        depth_path = self.depth_paths[img_id]
        pred_path = depth_path.replace("Scannet_Dataset", "Scannet_Add")
        img = Image.open(pred_path)
        depth = np.array(img, dtype=np.float32) / 1000  # mm to m
        depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_label(self, img_id, label_mapping, scan2nyu):
        img = Image.open(self.label_paths[img_id])
        label = np.array(img)
        # label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        label = scan2nyu[label]
        return label_mapping(label)

    def get_pose(self, img_id):
        try:
            pose = np.loadtxt(self.pose_paths[img_id]).reshape([4, 4])
        except:
            print(len(self.pose_paths),img_id)
            print(self.scene_name)
        return pose.copy()

    def compute_nearest_camera_indices(self, que_ids, ref_ids=None):
        if ref_ids is None: ref_ids = que_ids
        ref_poses = [self.get_pose(img_id) for img_id in ref_ids]
        ref_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in ref_poses])
        que_poses = [self.get_pose(que_id) for que_id in que_ids]
        que_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in que_poses])
        dists = np.linalg.norm(ref_cam_pts[None, :, :] - que_cam_pts[:, None, :], 2, 2)
        dists_idx = np.argsort(dists, 1)
        return dists_idx

    def select_working_views(self, que_id, ref_ids, view_num):
        ref_ids = np.array(ref_ids)
        dist_idx = self.compute_nearest_camera_indices([que_id], ref_ids)[0]
        dist_idx = dist_idx[ref_ids[dist_idx] != que_id]
        dist_idx = dist_idx[:view_num]
        ref_ids = ref_ids[dist_idx]

        ref_cv_ids = self.compute_nearest_camera_indices(ref_ids)[:, 0:4]
        return ref_ids, ref_cv_ids

    def build_imgs_info(self, img_ids, label_mapping, scan2nyu):
        ref_imgs = (np.array([self.get_image(img_id) for img_id in img_ids], dtype=np.float32) / 255).transpose(
            [0, 3, 1, 2])
        ref_labels = np.array([self.get_label(img_id, label_mapping, scan2nyu) for img_id in img_ids])
        ref_masks = np.array([self.get_mask(lbimg) for lbimg in ref_labels], dtype=np.float32)

        if self.use_depth:
            ref_depths = [self.get_depth(img_id) for img_id in img_ids]
        else:
            ref_depths = [self.get_pred_depth(img_id) for img_id in img_ids]
        ref_depths = np.array(ref_depths, dtype=np.float32)

        ref_poses = np.array([self.get_pose(img_id) for img_id in img_ids], dtype=np.float32)
        ref_intrinsics = np.array([self.intrinsic for _ in img_ids], dtype=np.float32)

        ref_depth_range = np.asarray([self.get_depth_range() for _ in img_ids], dtype=np.float32)

        ref_imgs_info = {'imgs': ref_imgs, 'poses': ref_poses, 'intrinsics': ref_intrinsics, 'depths': ref_depths,
                         'near_fars': ref_depth_range, 'fmasks': ref_masks, 'labels': ref_labels}

        return ref_imgs_info
