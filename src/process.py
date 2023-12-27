import argparse
import torch
import numpy as np
import pandas as pd
import os
import random


def normalize(array, axis=0):
    """
    归一化
    """
    _min = array.min(axis=axis, keepdims=True)
    _max = array.max(axis=axis, keepdims=True)
    factor = _max - _min
    return (array - _min) / np.where(factor != 0, factor, 1)


def parse_click_history(history_list):
    """
    解析用户点击历史，将其转换为适合模型处理的格式。
    in: [item id:时间戳, ...] ; len = 260087 ; 不同用户用0:0隔开
    out: [[item id, ...], ...] ; shape=260087x249 ; 整合成二维向量
    """
    clicks = list(map(lambda user_click: list(map(lambda item: item.split(':')[0],
                                                  user_click.split(','))),
                      history_list))
    _max_len = max(len(items) for items in clicks)
    clicks = [items + [0] * (_max_len - len(items)) for items in clicks]
    # TODO:1 np.long -> np.int32
    clicks = torch.tensor(np.array(clicks, dtype=np.int32)) - 1
    return clicks


def parse_user_protrait(protrait_list):
    """
    处理用户特征数据
    """
    return torch.tensor(normalize(np.array(list(map(lambda x: x.split(','),
                                                    protrait_list)),
                                           dtype=np.float32)))


def process_item(filename, outdir):
    """
    处理项目信息，创建项目特征、位置和价格的张量
    in: item_id item_vec price location
    out:
    """
    item_info = pd.read_csv(filename, ' ')
    item2id = np.array(item_info['item_id']) - 1  # 从0开始索引
    item2loc = torch.tensor(np.array(item_info['location'], dtype=np.float32)[item2id])
    item2price = torch.tensor(normalize(np.array(item_info['price'], dtype=np.float32)[item2id]) * 10, dtype=torch.float32)
    item2feature = torch.tensor(normalize(np.array(list(map(lambda x: x.split(','),
                                               item_info['item_vec'])),
                                      dtype=np.float32)[item2id]))
    item2info = torch.cat([item2feature, item2price[:, None], item2loc[:, None]], dim=-1)
    torch.save([item2info, item2price, item2loc], os.path.join(outdir, 'items_info.pt'))


def process_data(filename, outdir, savename):
    """
    读取数据文件，调用 parse_click_history 和 parse_user_protrait 函数，并保存处理后的数据。
    """
    dataset = pd.read_csv(filename, ' ')
    click_items = parse_click_history(dataset['user_click_history'])
    user_protrait = parse_user_protrait(dataset['user_protrait'])
    exposed_items = None
    if 'exposed_items' in dataset.columns:
        # TODO:1 np.long -> np.int32
        exposed_items = torch.tensor(np.array(list(map(lambda x: x.split(','),
                                                       dataset['exposed_items'])),
                                              dtype=np.int32) - 1)
    torch.save([user_protrait, click_items, exposed_items],
               os.path.join(outdir, savename))


def main(args):
    print('processing items info ...')
    process_item(args.itemset, args.outdir)
    print('processing trainset ...')
    process_data(args.trainset, args.outdir, 'train.pt')
    print('processing devset ...')
    process_data(args.devset, args.outdir, 'dev.pt')
    print('processing testset ...')
    process_data(args.testset, args.outdir, 'test.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO:1 去掉require 添加default
    parser.add_argument('--trainset', type=str, default='dataset/trainset.csv')
    parser.add_argument('--devset', type=str, default='dataset/track1_testset.csv')
    parser.add_argument('--testset', type=str, default='dataset/track2_testset.csv')
    parser.add_argument('--itemset', type=str, default='dataset/item_info.csv')
    parser.add_argument('--outdir', type=str, default='dataset')
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    main(args)

