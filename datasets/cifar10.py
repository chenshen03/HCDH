# -*- coding: utf-8 -*-

from datasets.transform import img_transform
from datasets.transform import Onehot

import numpy as np
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataloader import DataLoader


import os
import sys
import pickle


def load_data_gist(path, train=True):
    """加载对cifar10使用gist提取的数据

    Parameters
        path: str
        数据路径

        train: bools
        True，加载训练数据; False，加载测试数据

    Returns
        data: ndarray
        数据

        labels: ndarray
        标签
    """
    mat_data = sio.loadmat(path)

    if train:
        data = mat_data['traindata']
        labels = mat_data['traingnd'].astype(np.int)
    else:
        data = mat_data['testdata']
        labels = mat_data['testgnd'].astype(np.int)

    return data, labels


def load_data(opt, data_setting="s1"):
    """加载cifar10数据

    Parameters
        opt: Parser
        配置

    Returns
        query_dataloader, train_dataloader, database_dataloader: DataLoader
        数据加载器
    """
    if data_setting == 's1':
        CIFAR10.init(opt.data_path, 1000, 5000)
        R = 54000
    elif data_setting == 's2':
        CIFAR10.init(opt.data_path, 10000, 50000)
        R = 50000

    query_dataset = CIFAR10('query', transform=img_transform(opt.tencrop), target_transform=Onehot())
    train_dataset = CIFAR10('train', transform=img_transform(opt.tencrop), target_transform=Onehot())
    database_dataset = CIFAR10('database', transform=img_transform(opt.tencrop), target_transform=Onehot())

    query_dataloader = DataLoader(query_dataset,
                                  shuffle=False,
                                  batch_size=opt.batch_size,
                                  num_workers=4,
                                  )
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=opt.batch_size,
                                  num_workers=4,
                                  )
    database_dataloader = DataLoader(database_dataset,
                                     shuffle=False,
                                     batch_size=opt.batch_size,
                                     num_workers=4,
                                     )

    return query_dataloader, train_dataloader, database_dataloader, R


class CIFAR10(data.Dataset):
    """加载官网下载的CIFAR10数据集"""
    @staticmethod
    def init(root, num_query, num_train):
        cifar10_path = os.path.join(root, 'cifar10-s1.npy')
        if os.path.exists(cifar10_path) and num_query == 1000:
            print(f'loading cifar-s1 from {cifar10_path} ...')
            data = np.load(cifar10_path, allow_pickle=True).item()
            CIFAR10.QUERY_IMG = data['QUERY_IMG']
            CIFAR10.QUERY_TARGETS = data['QUERY_TARGETS']
            CIFAR10.TRAIN_IMG = data['TRAIN_IMG']
            CIFAR10.TRAIN_TARGETS = data['TRAIN_TARGETS']
            CIFAR10.DATABASE_IMG = data['DATABASE_IMG']
            CIFAR10.DATABASE_TARGETS = data['DATABASE_TARGETS']
        else:
            print(f'spliting cifar-10 ...')
            data_list = ['data_batch_1',
                        'data_batch_2',
                        'data_batch_3',
                        'data_batch_4',
                        'data_batch_5',
                        'test_batch',
                        ]
            base_folder = 'cifar-10-batches-py'

            data = []
            targets = []

            for file_name in data_list:
                file_path = os.path.join(root, base_folder, file_name)
                with open(file_path, 'rb') as f:
                    if sys.version_info[0] == 2:
                        entry = pickle.load(f)
                    else:
                        entry = pickle.load(f, encoding='latin1')
                    data.append(entry['data'])
                    if 'labels' in entry:
                        targets.extend(entry['labels'])
                    else:
                        targets.extend(entry['fine_labels'])

            data = np.vstack(data).reshape(-1, 3, 32, 32)
            data = data.transpose((0, 2, 3, 1))  # convert to HWC
            targets = np.array(targets)

            CIFAR10.ALL_IMG = data
            CIFAR10.ALL_TARGETS = targets

            # sort by class
            sort_index = CIFAR10.ALL_TARGETS.argsort()
            CIFAR10.ALL_IMG = CIFAR10.ALL_IMG[sort_index, :]
            CIFAR10.ALL_TARGETS = CIFAR10.ALL_TARGETS[sort_index]

            # (num_query / number of class) query images per class
            # (num_train / number of class) train images per class
            query_per_class = num_query // 10
            train_per_class = num_train // 10

            # permutate index (range 0 - 6000 per class)
            perm_index = np.random.permutation(CIFAR10.ALL_IMG.shape[0] // 10)
            query_index = perm_index[:query_per_class]
            train_index = perm_index[query_per_class: query_per_class + train_per_class]

            query_index = np.tile(query_index, 10)
            train_index = np.tile(train_index, 10)
            inc_index = np.array([i * (CIFAR10.ALL_IMG.shape[0] // 10) for i in range(10)])
            query_index = query_index + inc_index.repeat(query_per_class)
            train_index = train_index + inc_index.repeat(train_per_class)
            database_index = np.array([i for i in sort_index if i not in query_index and i not in train_index])

            # split data, tags
            CIFAR10.QUERY_IMG = CIFAR10.ALL_IMG[query_index, :]
            CIFAR10.QUERY_TARGETS = CIFAR10.ALL_TARGETS[query_index]
            CIFAR10.TRAIN_IMG = CIFAR10.ALL_IMG[train_index, :]
            CIFAR10.TRAIN_TARGETS = CIFAR10.ALL_TARGETS[train_index]
            if len(database_index) != 0:
                CIFAR10.DATABASE_IMG = CIFAR10.ALL_IMG[database_index, :]
                CIFAR10.DATABASE_TARGETS = CIFAR10.ALL_TARGETS[database_index]
            else:
                CIFAR10.DATABASE_IMG = CIFAR10.TRAIN_IMG
                CIFAR10.DATABASE_TARGETS = CIFAR10.TRAIN_TARGETS

    def __init__(self, mode='train',
                 transform=None, target_transform=None,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train':
            self.img = CIFAR10.TRAIN_IMG
            self.targets = CIFAR10.TRAIN_TARGETS
        elif mode == 'query':
            self.img = CIFAR10.QUERY_IMG
            self.targets = CIFAR10.QUERY_TARGETS
        elif mode == 'database':
            self.img = CIFAR10.DATABASE_IMG
            self.targets = CIFAR10.DATABASE_TARGETS
        else:
            self.img = CIFAR10.ALL_IMG
            self.targets = CIFAR10.ALL_TARGETS

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.img[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img)
