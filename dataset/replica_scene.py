from PIL import Image
import cv2
import numpy as np
import glob
import os
from natsort import natsorted


def compute_nearest_camera_indices(poses, que_ids, ref_ids=None):
    if ref_ids is None: ref_ids = que_ids
    ref_poses = [poses[ref_id] for ref_id in ref_ids]
    ref_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in ref_poses])
    que_poses = [poses[que_id] for que_id in que_ids]
    que_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in que_poses])

    dists = np.linalg.norm(ref_cam_pts[None, :, :] - que_cam_pts[:, None, :], 2, 2)
    dists_idx = np.argsort(dists, 1)
    return dists_idx


class ReplicaScene(object):
    def __init__(self, root_dir, scene_name, w, use_depth=True):
        self.root_dir = root_dir
        _, self.scene_name, self.seq_id = scene_name.split('/')

        self.root_dir = f'{root_dir}/{self.scene_name}/{self.seq_id}'
        self.use_depth = use_depth
        self.ratio = w / 640
        self.h, self.w = int(self.ratio * 480), int(w)

        rgb_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "rgb", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        self.rgb_paths = natsorted(rgb_paths)

        depth_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "depth", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        self.depth_paths = natsorted(depth_paths)

        label_paths = [x for x in glob.glob(os.path.join(
            self.root_dir, "semantic_class", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        self.label_paths = natsorted(label_paths)

        # Replica camera intrinsics
        # Pinhole Camera Model
        fx, fy, cx, cy, s = 320.0, 320.0, 319.5, 229.5, 0.0
        if self.ratio != 1.0:
            fx, fy, cx, cy = fx * self.ratio, fy * self.ratio, cx * self.ratio, cy * self.ratio
        # fx, fy, cx, cy = fx / 640, fy / 480, cx / 640, cy / 480
        self.intrinsic = np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.poses = np.loadtxt(f'{self.root_dir}/traj_w_c.txt', delimiter=' ').reshape(-1, 4, 4).astype(np.float32)
        # c2w

        self.img_ids = []
        for i, rgb_path in enumerate(self.rgb_paths):
            self.img_ids.append(i)

    def get_image(self, img_id):
        img = cv2.imread(self.rgb_paths[img_id])
        if self.w != 640:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_depth(self, img_id):
        img = Image.open(self.depth_paths[img_id])
        depth = np.array(img, dtype=np.float32) / 1000.0  # mm to m
        depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        return depth

    def get_pred_depth(self, img_id):
        depth_path = self.depth_paths[img_id]
        pred_path = depth_path.replace("Replica_Dataset", "Replica_Add")
        img = Image.open(pred_path)
        depth = np.array(img, dtype=np.float32) / 1000  # mm to m
        depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_mask(self, lbimg):
        mask = np.ones([self.h, self.w], bool)
        mask = mask * (lbimg != 6) * (lbimg != 16)
        return mask

    def get_depth_range(self):
        return np.array([0.1, 6.0], np.float32)

    def get_label(self, img_id, label_mapping):
        img = Image.open(self.label_paths[img_id])
        label = np.array(img)
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        return label_mapping(label)

    def get_pose(self, img_id):
        pose = self.poses[img_id]
        return pose.copy()

    def select_working_views(self, que_id, ref_ids, view_num):
        ref_ids = np.array(ref_ids)
        dist_idx = compute_nearest_camera_indices(self.poses, [que_id], ref_ids)[0]
        dist_idx = dist_idx[ref_ids[dist_idx] != que_id]
        dist_idx = dist_idx[:view_num]
        ref_ids = ref_ids[dist_idx]

        ref_cv_ids = compute_nearest_camera_indices(self.poses, ref_ids)[:, 0:4]
        return ref_ids, ref_cv_ids

    def build_imgs_info(self, img_ids, label_mapping):
        ref_imgs = (np.array([self.get_image(img_id) for img_id in img_ids], dtype=np.float32) / 255).transpose(
            [0, 3, 1, 2])
        ref_labels = np.array([self.get_label(img_id, label_mapping) for img_id in img_ids])
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
