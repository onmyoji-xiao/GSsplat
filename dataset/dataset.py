import glob
import os
import random

import torch
import numpy as np
from torch.utils.data import Dataset
from .replica_scene import ReplicaScene
from .scannet_scene import ScannetScene
import cv2
import pandas as pd


class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.
    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    Args:
        valid_cat_ids (list[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater  than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx
        if max_cat_id > 40:  # replica
            for i in range(self.cat_id2class.shape[0]):
                value = self.cat_id2class[i]
                if value == 19:
                    self.cat_id2class[i] = 6
                elif value == 20:
                    self.cat_id2class[i] = 19

    def __call__(self, seg_label):
        """Call function to map original semantic class to valid category ids.
        Args:
            results (dict): Result dict containing point semantic masks.
        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.
                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        seg_label = np.clip(seg_label, 0, self.max_cat_id)
        return self.cat_id2class[seg_label]


# def compute_projmats(imgs_info):
#     c2ws = imgs_info['poses']
#     intrinsics = imgs_info['intrinsics']
#     w2cs = []
#     affine_mats, affine_mats_inv = [], []
#     project_mats = []
#
#     for i in range(c2ws.shape[0]):
#         w2cs.append(np.linalg.inv(c2ws[i]))
#
#         proj_mat = np.eye(4)
#         intrinsic_temp = intrinsics[i].copy()
#         proj_mat[:3, :4] = intrinsic_temp @ w2cs[i][:3, :4]
#         affine_mats.append(proj_mat)
#         affine_mats_inv.append(np.linalg.inv(proj_mat))
#         # For unsupervised depth loss
#         proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
#         proj_mat[0, :4, :4] = w2cs[i][:4, :4]
#         proj_mat[1, :3, :3] = intrinsic_temp
#         project_mats.append(proj_mat)
#
#     imgs_info["affine_mats"] = np.stack(affine_mats)
#     imgs_info["affine_mats_inv"] = np.stack(affine_mats_inv)
#     imgs_info["project_mats"] = np.stack(project_mats)
#
#     return imgs_info


class ReplicaDataset(Dataset):
    def __init__(self, is_train, cfg):
        self.is_train = is_train
        self.cfg = cfg
        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[12, 17, 20, 22, 31, 37, 40, 44, 47, 56, 64, 79, 80, 87, 91, 92, 93, 95, 97],
            max_cat_id=101
        )
        # 0:blinds,1:camera,2:chair,3:clock,4:ceiling,5:door,6:floor,7:indoor-plant,8:lamp,9:panel,10:plate,
        # 11:switch,12:table,13:tv-screen,14:vase,15:vent,16:wall,17:wall-plug,18:window
        if is_train:
            scene_names = np.loadtxt(cfg.train_set_list, dtype=str)
        else:
            scene_names = np.loadtxt(cfg.val_set_list, dtype=str)
        self.samples, self.scenes = self.get_samples(scene_names)

    def get_samples(self, scene_names):
        samples = []
        scenes = {}
        if self.is_train:
            for scene_name in scene_names:
                rs = ReplicaScene(self.cfg.top_dir, scene_name, self.cfg.input_width, self.cfg.use_depth)
                scenes[scene_name] = rs
                train_ids = rs.img_ids
                for idx in train_ids:
                    samples.append((scene_name, idx))
        else:
            for scene_name in scene_names:
                rs = ReplicaScene(self.cfg.top_dir, scene_name, self.cfg.input_width, self.cfg.use_depth)
                scenes[scene_name] = rs
                val_ids = rs.img_ids[2:700:20]
                if len(val_ids) > 10:
                    val_ids = val_ids[:10]
                for idx in val_ids:
                    samples.append((scene_name, idx))
        return samples, scenes

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))  # random index
        return [lst[x] for x in indices]

    def multi_scale_depth(self, depth_h):
        '''
        This is the implementation of Klevr dataset and move here to make dataset format the same
        '''

        depth = {}
        for l in range(3):
            depth[f"level_{l}"] = cv2.resize(
                depth_h,
                None,
                fx=1.0 / (2 ** l),
                fy=1.0 / (2 ** l),
                interpolation=cv2.INTER_NEAREST,
            )
            # depth[f"level_{l}"][depth[f"level_{l}"] > far_bound * 0.95] = 0.0

        if self.is_train:
            cutout = np.ones_like(depth[f"level_2"])
            h0 = int(np.random.randint(0, high=cutout.shape[0] // 5, size=1))
            h1 = int(
                np.random.randint(
                    4 * cutout.shape[0] // 5, high=cutout.shape[0], size=1
                )
            )
            w0 = int(np.random.randint(0, high=cutout.shape[1] // 5, size=1))
            w1 = int(
                np.random.randint(
                    4 * cutout.shape[1] // 5, high=cutout.shape[1], size=1
                )
            )
            cutout[h0:h1, w0:w1] = 0
            depth_aug = depth[f"level_2"] * cutout
        else:
            depth_aug = depth[f"level_2"].copy()

        return depth, depth_aug

    def compute_projmats(self, sample):
        sample['w2cs'] = []
        affine_mats, affine_mats_inv, depths_aug = [], [], []
        project_mats = []
        depths = {"level_0": [], "level_1": [], "level_2": []}

        for i in range(sample['poses'].shape[0]):
            sample['w2cs'].append(np.linalg.inv(sample['poses'][i]))
            # sample['w2cs'].append(torch.asarray(np.linalg.inv(np.asarray(sample['c2ws'][i]))))

            aff = []
            aff_inv = []
            proj_matrices = []

            for l in range(3):
                proj_mat_l = np.eye(4)
                intrinsic_temp = sample['intrinsics'][i].copy()
                intrinsic_temp[:2] = intrinsic_temp[:2] / (2 ** l)
                proj_mat_l[:3, :4] = intrinsic_temp @ sample['w2cs'][i][:3, :4]
                aff.append(proj_mat_l)
                aff_inv.append(np.linalg.inv(proj_mat_l))
                # For unsupervised depth loss
                proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
                proj_mat[0, :4, :4] = sample['w2cs'][i][:4, :4]
                proj_mat[1, :3, :3] = intrinsic_temp
                proj_matrices.append(proj_mat)

            aff = np.stack(aff, axis=-1)
            aff_inv = np.stack(aff_inv, axis=-1)
            proj_matrices = np.stack(proj_matrices)

            affine_mats.append(aff)
            affine_mats_inv.append(aff_inv)
            project_mats.append(proj_matrices)

            depth, depth_aug = self.multi_scale_depth(np.asarray(sample['depths'][i]))
            depths["level_0"].append(depth["level_0"])
            depths["level_1"].append(depth["level_1"])
            depths["level_2"].append(depth["level_2"])
            depths_aug.append(depth_aug)

        affine_mats = np.stack(affine_mats)
        affine_mats_inv = np.stack(affine_mats_inv)
        project_mats = np.stack(project_mats)
        depths_aug = np.stack(depths_aug)
        depths["level_0"] = np.stack(depths["level_0"])
        depths["level_1"] = np.stack(depths["level_1"])
        depths["level_2"] = np.stack(depths["level_2"])

        sample['w2cs'] = np.stack(sample['w2cs'], 0)  # (1+nb_views, 4, 4)
        sample['affine_mats'] = affine_mats
        sample['affine_mats_inv'] = affine_mats_inv
        sample['depths_aug'] = depths_aug
        sample['depths_'] = depths
        sample['project_mats'] = project_mats

        return sample

    def __getitem__(self, index):
        # if self.is_train:
        #     inx = random.randint(0, len(self.samples))
        #     scene_name, que_id = self.samples[inx]
        # else:
        scene_name, que_id = self.samples[index]
        scene = self.scenes[scene_name]
        if self.is_train:
            ref_ids = scene.img_ids
        else:
            ref_ids = scene.img_ids[:700:5]
        view_ids, cv_ids = scene.select_working_views(que_id, ref_ids, self.cfg.view_num)
        view_ids = np.concatenate((view_ids, [que_id]), 0)
        imgs_info = scene.build_imgs_info(view_ids, self.label_mapping)

        imgs_info['scene_name'] = scene_name
        imgs_info['closest_idxs'] = cv_ids.astype(np.int64)
        imgs_info = self.compute_projmats(imgs_info)

        return imgs_info

    def __len__(self):
        return len(self.samples)


class ScannetDataset(Dataset):
    def __init__(self, is_train, cfg):
        self.is_train = is_train
        self.cfg = cfg

        mapping_file = os.path.join('./data/scannetv2-labels.combined.tsv')
        mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
        scan_ids = mapping_file['id'].values
        nyu40_ids = mapping_file['nyu40id'].values
        scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
        for i in range(len(scan_ids)):
            scan2nyu[scan_ids[i]] = nyu40_ids[i]
        self.scan2nyu = scan2nyu
        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
            max_cat_id=40
        )
        # 0:wall 1:floor 2:cabinet 3:bed 4:chair 5:sofa 6:table 7:door 8:window 9:bookshelf
        # 10:picture 11:counter 12:desk 13:curtain 14:refridgerator 15:shower curtain 16:toilet 17:sink 18:bathtub 19:otherfurniture

        if is_train:
            scene_names = np.loadtxt(cfg.train_set_list, dtype=str)
        else:
            scene_names = np.loadtxt(cfg.val_set_list, dtype=str)
        self.samples, self.scenes = self.get_samples(scene_names)

    def get_samples(self, scene_names):
        samples = []
        scenes = {}
        if self.is_train:
            for scene_name in scene_names:
                ss = ScannetScene(self.cfg.top_dir, scene_name, is_train=True)
                scenes[scene_name] = ss
                for idx in ss.img_ids:
                    samples.append((scene_name, idx))
        else:
            for scene_name in scene_names:
                ss = ScannetScene(self.cfg.top_dir, scene_name, is_train=False)
                scenes[scene_name] = ss
                val_ids = ss.img_ids[2:700:20]
                if len(val_ids) > 10:
                    val_ids = val_ids[:10]
                for idx in val_ids:
                    samples.append((scene_name, idx))
        return samples, scenes

    def multi_scale_depth(self, depth_h):
        '''
        This is the implementation of Klevr dataset and move here to make dataset format the same
        '''

        depth = {}
        for l in range(3):
            depth[f"level_{l}"] = cv2.resize(
                depth_h,
                None,
                fx=1.0 / (2 ** l),
                fy=1.0 / (2 ** l),
                interpolation=cv2.INTER_NEAREST,
            )
            # depth[f"level_{l}"][depth[f"level_{l}"] > far_bound * 0.95] = 0.0

        if self.is_train:
            cutout = np.ones_like(depth[f"level_2"])
            h0 = int(np.random.randint(0, high=cutout.shape[0] // 5, size=1))
            h1 = int(
                np.random.randint(
                    4 * cutout.shape[0] // 5, high=cutout.shape[0], size=1
                )
            )
            w0 = int(np.random.randint(0, high=cutout.shape[1] // 5, size=1))
            w1 = int(
                np.random.randint(
                    4 * cutout.shape[1] // 5, high=cutout.shape[1], size=1
                )
            )
            cutout[h0:h1, w0:w1] = 0
            depth_aug = depth[f"level_2"] * cutout
        else:
            depth_aug = depth[f"level_2"].copy()

        return depth, depth_aug

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))  # random index
        return [lst[x] for x in indices]

    def compute_projmats(self, sample):
        sample['w2cs'] = []
        affine_mats, affine_mats_inv, depths_aug = [], [], []
        project_mats = []
        depths = {"level_0": [], "level_1": [], "level_2": []}

        for i in range(sample['poses'].shape[0]):
            sample['w2cs'].append(np.linalg.inv(sample['poses'][i]))

            aff = []
            aff_inv = []
            proj_matrices = []

            for l in range(3):
                proj_mat_l = np.eye(4)
                intrinsic_temp = sample['intrinsics'][i].copy()
                intrinsic_temp[:2] = intrinsic_temp[:2] / (2 ** l)
                proj_mat_l[:3, :4] = intrinsic_temp @ sample['w2cs'][i][:3, :4]
                aff.append(proj_mat_l)
                aff_inv.append(np.linalg.inv(proj_mat_l))
                # For unsupervised depth loss
                proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
                proj_mat[0, :4, :4] = sample['w2cs'][i][:4, :4]
                proj_mat[1, :3, :3] = intrinsic_temp
                proj_matrices.append(proj_mat)

            aff = np.stack(aff, axis=-1)
            aff_inv = np.stack(aff_inv, axis=-1)
            proj_matrices = np.stack(proj_matrices)

            affine_mats.append(aff)
            affine_mats_inv.append(aff_inv)
            project_mats.append(proj_matrices)

            depth, depth_aug = self.multi_scale_depth(np.asarray(sample['depths'][i]))
            depths["level_0"].append(depth["level_0"])
            depths["level_1"].append(depth["level_1"])
            depths["level_2"].append(depth["level_2"])
            depths_aug.append(depth_aug)

        affine_mats = np.stack(affine_mats)
        affine_mats_inv = np.stack(affine_mats_inv)
        project_mats = np.stack(project_mats)
        depths_aug = np.stack(depths_aug)
        depths["level_0"] = np.stack(depths["level_0"])
        depths["level_1"] = np.stack(depths["level_1"])
        depths["level_2"] = np.stack(depths["level_2"])

        sample['w2cs'] = np.stack(sample['w2cs'], 0)  # (1+nb_views, 4, 4)
        sample['affine_mats'] = affine_mats
        sample['affine_mats_inv'] = affine_mats_inv
        sample['depths_aug'] = depths_aug
        sample['depths_'] = depths
        sample['project_mats'] = project_mats

        return sample

    def __getitem__(self, index):
        scene_name, que_id = self.samples[index]

        scene = self.scenes[scene_name]
        if self.is_train:
            ref_ids = scene.img_ids
        else:
            ref_ids = scene.img_ids[:700:5]

        view_ids, cv_ids = scene.select_working_views(que_id, ref_ids, self.cfg.view_num)
        view_ids = np.concatenate((view_ids, [que_id]), 0)
        imgs_info = scene.build_imgs_info(view_ids, self.label_mapping, self.scan2nyu)

        imgs_info['scene_name'] = scene_name
        imgs_info['closest_idxs'] = cv_ids.astype(np.int64)
        imgs_info = self.compute_projmats(imgs_info)

        return imgs_info

    def __len__(self):
        return len(self.samples)
