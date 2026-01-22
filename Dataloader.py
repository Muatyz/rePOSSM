from itertools import chain, islice

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import json
meta_data = json.load(open("processed_data/meta_data.json", "r"))
# 读取 meta_data.json 中的数据

max_bin = meta_data["max_bin"]
max_token = meta_data["max_token"]
# max_bin: 所有 trial 中最大的 bin 数量
# max_token: 切分后一个 bin 中最大的 spikes 数量


class my_dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, path):
        '''
        通过 torch.load() 以 list 形式加载数据
        Args:
            self: 
            path: 数据路径, "processed_data/sliced_trials.pt"
        '''
        self.dataset = torch.load(path, weights_only=False)
        # 每个元素形式是 trial = {
        #     'trial_id': trial_id,    
        #     'spikes': active_spike, 
        #             # active_spike = List[bin],
        #             # bin = List[(channel_id, offset)] 或者 0
        #     'vel': vel
        # }
        # self.dataset = [trial_0, trial_1, ...]

    def __getitem__(self, idx):
        '''
        取数据集内 trial id 为 idx 的部分. 
        Args:
            self: 
            idx: trial 索引
        Returns:
            spike:  该 trial 中的 spikes 数据
            vel:    该 trial 中的 velocity 数据
        '''
        spike = self.dataset[idx]["spikes"]
        vel = self.dataset[idx]["vel"]
        # vel.shape = (time_length, 2), 2 表示 (vx, vy)
        return spike, vel

    def __len__(self):
        '''
        读取数据集长度
        Args:
            self:
        Returns:
            数据集长度
        '''
        return len(self.dataset)

def pad_collate_fn(batch):
    '''
    
    Args:
        batch: 通过 Dataset.__getitem__() 取得的 list, 形式为 [
            (spikes_trial_0, vel_trial_0), 
            (spikes_trial_1, vel_trial_1), 
            ...]
    '''
    trials, vel = zip(*batch)
    # 将 batch 列表拆分为 spikes 和 vel 两个部分
    # trials =  (spikes_trial_0, spikes_trial_1, ...)
    # spikes_trial_i = [bin_0, bin_1, ...]
    # bin_j = [(channel_id, offset), ...] or 0
    # vel =     (vel_trial_0, vel_trial_1, ...)
    # vel_trial_i = [[vx, vy], [vx, vy], ...]
    
    bins_per_trial = [len(trial) for trial in trials]
    # 记录每个 trial 内的 bin 数量, 以备后续再度以 trial 分割
    
    unfold_spikes = list(chain.from_iterable(trials))
    # trials = [
    #     trial_0_spikes, # List[bin_0, bin_1, ...], 
    #                     # bin_j = [(channel_id, offset), ...] or 0
    #     trial_1_spikes,
    #     ...
    # ]
    # -> unfold_spikes = [
    #     bin_0^0, bin_1^0, ..., # trial 0
    #     bin_0^1, bin_1^1, ..., # trial 1
    #     ...
    # ]
    # 将所有 trial 内的 bin 首尾相连 (chain) 成一个长 list, 即去掉 trial 边界
    
    new_unfold_spikes = [x if x != 0 else torch.empty(0, 2) for x in unfold_spikes]
    # 若有 bin 内无 spike(赋值为 0), 则替换为 shape(0, 2) 的空 tensor, 0 表示 spikes 数, 2 表示 (channel_id, offset)

    bin_seq_lens = torch.tensor([len(x) for x in new_unfold_spikes])
    # bin_seq_lens: 记录每个 bin 内的 spikes 数量
    # x.shape = (bin 内 spikes 数, 2)

    it_len = iter(bin_seq_lens)
    # 以 bin_seq_lens 生成迭代器 it_len
    
    bin_seq_lens_restored = [
        torch.tensor(
            list(islice(it_len, l)) 
            # 按照各 trial 的 bin 数量 l, 从迭代器中取出 l 个元素组成 list
            # 语法 islice(iterable, stop) 从当前位置开始连续取 stop 个元素
            ) 
        for l in bins_per_trial]
    # bins_per_trial = [trial 0 内 bin 数, trial 1 内 bin 数, ...]
    # 即通过 bins_per_trial 记录的各 trial 内 bin 数量恢复 trial 的信息
    # 若每个 bin 内的 spikes 数量列表(无 trial 分割) bin_seq_lens = [3,0,1,2,4], 
    # 且 每个 trial 内的 bin 数量为 bins_per_trial = [3,2] (表示第 1 个 trial 有 3 个 bin, 第 2 个 trial 有 2 个 bin)
    # 则 bin_seq_lens_restored = [[3,0,1], [2,4]]
    
    padded_seq_lens = pad_sequence(
        # 把 bin 数不同的一组 trial, 在尾部补 0 直至 max_bin 位
        bin_seq_lens_restored,  # sequences: List[Tensor]
        batch_first=True,       # batch_first: bool
        padding_value=0         # padding_value: float
        )
    # -> padded_seq_lens: tensor([
    #     [3, 0, 1],
    #     [2, 4, 0],
    #     ...
    # ])
    # 将 trial 并行处理, 即 trial 数作为 batch size(维度) 

    tensor_unfold_spikes = [torch.tensor(x) for x in new_unfold_spikes]
    # new_unfold_spikes 是去除 trial 边界的 List[bin], 且将 0 替换为 torch.empty(0, 2) 便于维数的统一
    # 通过 torch.tensor() 将所有 bin 转换为 tensor() 格式
    
    padded_unfold_spikes = pad_sequence(
        tensor_unfold_spikes,     # sequences: List[Tensor]
        batch_first = True,       # batch_first: bool
        padding_value = 0         # padding_value: float 填充值
        )
    # torch.tensor(x) 的长度不等, 因此对于长度较短的 tensor 会在尾部补 0 直至 max_token 位
    # -> padded_unfold_spikes: tensor([
    #     [[channel_id, offset], [0,0], [0,0]], # bin_0
    #     [[channel_id, offset], [channel_id, offset], [0,0]], # bin_1
    #     ...
    # ])
    # 对于 spikes 较少的 bin, 在后面补 [0,0] 直至 len(bin) == max_spikes
    # padded_unfold_spikes.shape = (bin 总数, max_spikes, 2)
    
    it = iter(padded_unfold_spikes)
    # 类似地, 以 padded_unfold_spikes 生成迭代器 it
    
    # 以各 trial 的 bin 数量重新分割 padded_unfold_spikes
    padded_spikes = [
        list(islice(it, trial_len))  
        # 按照各 trial 的 bin 数量 trial_len, 从迭代器中取出 trial_len 个元素组成 list
        for trial_len in bins_per_trial
        ]
    # bins_per_trial = [trial 0 内 bin 数, trial 1 内 bin 数, ...]
    # padded_unfold_spikes: tensor([
    #     [ [channel_id, offset], 
    #       [0,0], 
    #       [0,0]],                 ## bin_0^0
    #     [ [channel_id, offset], 
    #       [channel_id, offset], 
    #       [0,0]],                 ## bin_1^0
    #     ...
    #     [ [channel_id, offset], 
    #       [channel_id, offset], 
    #       [channel_id, offset]],  ## bin_0^1
    #     [ [channel_id, offset], 
    #       [0,0], 
    #       [0,0]],                 ## bin_1^1
    #     ...
    # ])
    # -> padded_spikes = [
    #     [                             # trial 0
    #         [ [channel_id, offset], 
    #           [0,0], 
    #           [0,0]],                     ## bin_0^0
    #         [ [channel_id, offset], 
    #           [channel_id, offset], 
    #           [0,0]],                     ## bin_1^0
    #         ...
    #     ],
    #     [                             # trial 1
    #         [ [channel_id, offset], 
    #           [channel_id, offset], 
    #           [0,0]],                     ## bin_0^1
    #         [ [channel_id, offset], 
    #           [channel_id, offset], 
    #           [channel_id, offset]],      ## bin_1^1
    #         ...
    #     ],
    #     ...
    # ]
    # bin.shape = (max_spikes_in_batch, 2)
    # 这里是按照 batch 内最大 spikes 数补齐而非全局参数 max_token, 否则会造成巨量的性能浪费

    torch_padded_spikes = [
        torch.stack(x)
        # torch.stack(): 
        for x in padded_spikes
        ]
    # padded_spikes = [
    #     [bin_0^0, bin_1^0, ...],  # trial 0
    #     [bin_0^1, bin_1^1, ...],  # trial 1
    #     ...
    # ]
    # -> torch_padded_spikes = [
    #     tensor([ bin_0^0, bin_1^0, ... ]),  # trial 0
    #     tensor([ bin_0^1, bin_1^1, ... ]),  # trial 1
    #     ...
    # ]
    
    padded_bin = pad_sequence(
        torch_padded_spikes, 
        batch_first = True, 
        padding_value = 0
        )
    # 各 trial 内的 bin 数不同, 因此对于 bin 数较少的 trial 会在尾部补 0 直至 max_bin 位
    
    trial_counts = torch.tensor(bins_per_trial)
    # 各 trial 内有效的 bin 数量列表
    
    max_bins = padded_bin.size(1)
    # batch 内最大的 bin 数量
    
    bin_mask = torch.arange(max_bins)[None, :] < trial_counts[:, None]
    # 构建 bin_mask, True 表示有效位置, False 表示 padding
    # (1, max_bins) 与 (batch_size, 1) 广播比较
    # bin_mask = [
    #     [1, 1, 1, 0, 0, ...], # trial 0: 前 L0 个 bin 有效
    #     [1, 1, 0, 0, 0, ...], # trial 1: 前 L1 个 bin 有效
    #     ...
    # ]

    max_seq_len = padded_bin.size(2)
    seq_range = torch.arange(max_seq_len).view(1, 1, -1)
    len_target = padded_seq_lens.unsqueeze(-1)
    spike_mask = seq_range < len_target

    vel = [torch.as_tensor(v) for v in vel]
    vel = pad_sequence(
        vel, 
        batch_first = True, 
        padding_value = 0.0
        ) # (batch_size, max_time_length, 2)
    vel_lens = torch.tensor([
        len(x) 
        for x in vel])
    # True: valid, False: padding value
    return padded_bin, bin_mask, spike_mask, vel, vel_lens


def get_dataloader(data_dir="processed_data/sliced_trials.pt", batch_size=16, n_workers=0):
    '''
    Generate dataloader
    
    Args:
        data_dir: 数据路径, "processed_data/sliced_trials.pt"
        batch_size: 批量大小
        n_workers: DataLoader 的子进程数量
    
    Returns:
        train_loader: 训练集 DataLoader
        valid_loader: 验证集 DataLoader
    '''
    
    dataset = my_dataset(data_dir)
    # 以 my_dataset 类加载数据集

    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)
    # 将数据集按 9:1 随机划分为训练集和验证集

    train_loader = DataLoader(
        trainset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = n_workers,
        pin_memory = True,
        collate_fn = pad_collate_fn,
    )
    valid_loader = DataLoader(
        validset,
        batch_size = batch_size,
        num_workers = n_workers,
        drop_last = True,
        pin_memory = True,
        collate_fn = pad_collate_fn,
    )

    return train_loader, valid_loader