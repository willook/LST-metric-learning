# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
from typing import Protocol
import numpy as np
import pickle
import logging
import h5py
from sklearn.model_selection import train_test_split

root_path = './'
stat_path = osp.join(root_path, 'statistics')
setup_file = osp.join(stat_path, 'setup.txt')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
replication_file = osp.join(stat_path, 'replication.txt')
label_file = osp.join(stat_path, 'label.txt')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
examplar_txt_path = osp.join(stat_path, "examplar.txt")

denoised_path = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')

save_path = './'


if not osp.exists(save_path):
    os.mkdir(save_path)


def remove_nan_frames(ske_name, ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))

    return ske_joints[valid_frames]

def seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)

        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update

    return skes_joints


def frame_translation(skes_joints, skes_name, frames_cnt):
    nan_logger = logging.getLogger('nan_skes')
    nan_logger.setLevel(logging.INFO)
    nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
    nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        # Calculate the distance between spine base (joint-1) and spine (joint-21)
        j1 = ske_joints[:, 0:3]
        j21 = ske_joints[:, 60:63]
        dist = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

        for f in range(num_frames):
            origin = ske_joints[f, 3:6]  # new origin: middle of the spine (joint-2)
            if (ske_joints[f, 75:] == 0).all():
                ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / \
                                      dist[f] + np.tile(origin, 25)
            else:
                ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / \
                                 dist[f] + np.tile(origin, 50)

        ske_name = skes_name[idx]
        ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
        frames_cnt[idx] = num_frames  # update valid number of frames
        skes_joints[idx] = ske_joints

    return skes_joints, frames_cnt


def align_frames(skes_joints, frames_cnt):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()  # 300
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints,
                                                               np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 120))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


def split_dataset(skes_joints, label, performer, camera, skes_name, save_path):
    train_indices, test_indices, examplar_indices = get_indices(performer, label, skes_name)
    m = 'sklearn'  # 'sklearn' or 'numpy'
    # Select validation set from training set
    # train_indices, val_indices = split_train_val(train_indices, m)

    # Save labels and num_frames for each sequence of each data set
    train_labels = label[train_indices]
    test_labels = label[test_indices]
    examplar_labels = label[examplar_indices]
    
    train_one_hot_labels = one_hot_vector(train_labels)
    test_one_hot_labels = one_hot_vector(test_labels)
    examplar_one_hot_labels = one_hot_vector(examplar_labels)

    print("total number of data:", len(skes_joints))
    print("train number of data:", len(train_one_hot_labels))
    print("test number of data:", len(test_one_hot_labels))
    print("examplar number of data:", len(examplar_one_hot_labels))
    print(len(skes_joints) == len(train_one_hot_labels) + len(test_one_hot_labels) + len(examplar_one_hot_labels))
    print(len(skes_joints[train_indices]) == len(train_one_hot_labels))
    print(len(skes_joints[test_indices]) == len(test_one_hot_labels))
    print(len(skes_joints[examplar_indices]) == len(examplar_one_hot_labels))
    assert len(skes_joints) == len(train_one_hot_labels) + len(test_one_hot_labels) + len(examplar_one_hot_labels)
    assert len(skes_joints[train_indices]) == len(train_one_hot_labels)
    assert len(skes_joints[test_indices]) == len(test_one_hot_labels)
    assert len(skes_joints[examplar_indices]) == len(examplar_one_hot_labels)
    
    print("save data...")
    np.savez('NTU60_OS.npz', 
            x_train = skes_joints[train_indices],
            y_train = train_one_hot_labels,
            x_test = skes_joints[test_indices],
            y_test = test_one_hot_labels,
            x_examplar = skes_joints[examplar_indices],
            y_examplar = examplar_one_hot_labels,
            allow_pickle=False)

def extract_test_dict(examplar_txt_path):
    examplar_files = []
    test_classes = []
    with open(examplar_txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            examplar_files.append(line)
            test_classes.append(int(line[-3:])-1)
    return set(examplar_files), set(test_classes)

def get_indices(performer, labels, skes_name):
    examplar_files, test_ids = extract_test_dict(examplar_txt_path)
    train_ids = []
    for id in range(0, 60):
        if id not in test_ids:
            train_ids.append(id)
    
    # train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
    #                 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    # test_ids = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
    #             24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
    
    # Get indices of test data
    train_indices = np.empty(0)
    test_indices = []
    examplar_indices = []

    # Get indices of training data
    for train_id in train_ids:
        temp = np.where(labels == train_id)[0]  # 0-based index
        train_indices = np.hstack((train_indices, temp)).astype(np.int)
        
    for idx, (ske_name, label) in enumerate(zip(skes_name, labels)):
        ske_name = ske_name.decode('UTF-8')
        if ske_name in examplar_files:
            examplar_indices.append(idx)
        elif (label) in test_ids:
            test_indices.append(idx)
            
    examplar_indices = np.array(examplar_indices, dtype=np.int)
    return train_indices, test_indices, examplar_indices


if __name__ == '__main__':
    camera = np.loadtxt(camera_file, dtype=np.int)  # camera id: 1, 2, 3
    performer = np.loadtxt(performer_file, dtype=np.int)  # subject id: 1~40
    label = np.loadtxt(label_file, dtype=np.int) - 1  # action label: 0~59

    frames_cnt = np.loadtxt(frames_file, dtype=np.int)  # frames_cnt
    skes_name = np.loadtxt(skes_name_file, dtype=np.string_)

    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    skes_joints = seq_translation(skes_joints)

    skes_joints = align_frames(skes_joints, frames_cnt)  # aligned to the same frame length

    split_dataset(skes_joints, label, performer, camera, skes_name, save_path)
