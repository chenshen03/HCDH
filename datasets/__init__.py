#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datasets.cifar10 as cifar10
import datasets.nus_wide as nus_wide


def create(opt):
    """加载数据

    Parameters
        opt: Parser
        参数

    Returns
        DataLoader
        数据加载器
    """
    if opt.dataset == 'cifar_s1':
        return cifar10.load_data(opt, data_setting='s1')
    elif opt.dataset == 'cifar_s2':
        return cifar10.load_data(opt, data_setting='s2')
    elif opt.dataset == 'nuswide_21':
        return nus_wide.load_data(opt)
