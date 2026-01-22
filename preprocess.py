from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import torch

from Config import my_POSSMConfig
config = my_POSSMConfig()

import pandas as pd
lag_delta = pd.to_timedelta(config.time_lag, unit='ms')

from tqdm import tqdm
import json
import os
os.makedirs("processed_data", exist_ok=True)

# Load dataset
dataset = NWBDataset("./000128/sub-Jenkins/", "*train", split_heldout=False)

channel_ids = dataset.data['spikes'].keys()
# 读取数据集中原始电极/通道的 id 列表

# 统计神经元数量
num_channel = len(channel_ids) # 182 channels
channel_id_to_idx = {raw_id: i for i, raw_id in enumerate(channel_ids)}
# 原始的 channel_id 是电极编号(3, 7, 42...), 将其按顺序映射为 0~(num_channel-1) 的索引

# 1. Slice spikes into bins
def spike_slice_bins(input):
    time_length, num_channel = input.shape
    # time_length: 实验时间长度
    # num_channel: 信号通道数(神经元个数), 182
    
    active_spike = [0]*(time_length//config.bin_size + 1)
    # 初始化列表, 长度为所有 bin 的数量, 预先填充 0. 

    time_idxs, chan_idxs = np.nonzero(input)
    # time_idxs: spike 发生的时间点索引
    # chan_idxs: spike 发生的 channel 索引
    
    bin_indices = time_idxs // config.bin_size
    # bin_indices: spike 发生于第几个 bin

    offsets = time_idxs % config.bin_size
    # offsets: spike 在 bin 内的精确时间
    # spike 的 "精确时间" 只需要在 bin 内用 offset 表示

    # 构造每个 bin 内的 spike token 列表
    for spike in range(len(time_idxs)):
        bin_idx = bin_indices[spike]
        cid = chan_idxs[spike]
        offset = offsets[spike]
        # bin_idx: spike 发生所属的 bin
        # cid: 放电的神经元/channel
        # offset: spike 在 bin 内的相对时间
        
        if active_spike[bin_idx] == 0:
            # 目前该 bin 尚未写入 spike 的 (cid, offset) 信息
            active_spike[bin_idx] = [(cid, offset)]
        else:
            # 这个 bin 内有多个 spike, 追加新的 (cid, offset) 信息
            active_spike[bin_idx].append((cid, offset))

    return active_spike
    # [
    #    [(3,12), (87,31)], # 第 1 个 bin
    #    0,                 # 第 2 个 bin
    #    [(5,7)],           # 第 3 个 bin
    #    ...,
    # ]

# 2. 根据 trial_id 切分数据
# trial 是指一次完整的(明确起止时间)的实验过程
# 初始化存储列表. 
sliced_trials = []
max_bin = 0         # 一个 trial 最多有多少个 bin
max_token = 0       # 一个 bin 内最多有多少个 spike/token
max_time_length = 0 # 一个 trial 最多有多少个时间点(最长时间)

for index, row in tqdm(dataset.trial_info.iterrows()):
    # dataset.trial_info: 
    trial_id = row['trial_id']
    start_time = row['start_time']
    end_time = row['end_time']
    # trial_id: 每次 trial 的唯一编号
    # start_time: trial 的起始时间
    # end_time: trial 的结束时间

    trial_spike_df = dataset.data.loc[
        start_time : end_time
        ]["spikes"]
    trial_vel_df = dataset.data.loc[
        start_time + lag_delta : end_time + lag_delta
        ]["hand_vel"]
    # 以 .loc[a:b] 提取 trial 的 spikes 列和 hand_vel 列从时间 a 到 b 的数据. 
    # dataset.data["spikes"]: 每一行是一个长度为 182 的向量 [0, 0, 1, ...]
    # dataset.data["hand_vel"]: 每一行是一个长度为 2 的向量 [x, y]
    # 神经活动领先于行为输出, 因此在预处理时就将 vel 数据向后平移 lag_delta 时间
    
    s_vals = trial_spike_df.values
    v_vals = trial_vel_df.values
    # 通过 .values 提取 spike 和 vel 的数组
    
    min_len = min(len(s_vals), len(v_vals))
    # 注意: 虽然 s_vals 和 v_vals 对应的现实时间宽度是相同的
    # 由于稀疏/NaN 数据, 即使名义上每 ms 一行, 中间存在缺失
    # 因此, 取两者的最小长度
    s_vals = s_vals[:min_len]
    v_vals = v_vals[:min_len]

    spikes_has_nan = np.isnan(s_vals).any(axis=1)
    vel_has_nan = np.isnan(v_vals).any(axis=1)
    # 寻找 spikes 和 vel 中值为 NaN 的时间点
    
    valid_mask = ~spikes_has_nan & ~vel_has_nan
    # 取 "非 spike_nan" 与 "非 vel_nan" 的交集, 作为有效(valid) 时间索引. 其被称为 "掩码"(mask)

    spikes = s_vals[valid_mask] # shape: [time_length, channels]
    vel = v_vals[valid_mask] # shape: [time_length, 2] i.e. [[x, y], [x, y], ...]
    # 通过有效时间点筛选出 s_vals 和 v_vals 中的有效数据
    
    active_spike = spike_slice_bins(spikes) 
    # [
    #    [(3,12), (87,31)], # 第 1 个 bin
    #    0,                 # 第 2 个 bin
    #    [(5,7)],           # 第 3 个 bin
    #    ...,
    # ]
    # 0: bin 内无 spike
    # list[(channel_id, offset), ...]
    # 将清洗后的 spikes 数据按 bin_size 切分
    
    trial_max_token = max([len(bin) for bin in active_spike if bin != 0])
    # 对于该 trial, 去除空 bin 后, 计算各 bin 内 spike 数, 取最大值为 max_token

    sliced_trials.append({
        'trial_id': trial_id,   # 每次 trial 的唯一编号
        'spikes': active_spike, 
        # [
        #    [(3,12), (87,31)], # 第 1 个 bin, (channel_id, offset)
        #    0,                 # 第 2 个 bin
        #    [(5,7)],           # 第 3 个 bin
        #    ...,
        # ]
        'vel': vel              
    })

    max_bin = max(max_bin, len(active_spike))
    max_token = max(max_token, trial_max_token)
    max_time_length = max(max_time_length, len(vel))

torch.save(sliced_trials, "processed_data/sliced_trials.pt")

# =================【计算统计量】=================
# 1. 收集所有 trial 的 velocity 数据
all_vels = []
# 初始化速度存储列表

for trial in sliced_trials:
    all_vels.append(trial['vel']) # list of arrays

# 2. 拼接成一个巨大的数组 (Total_Time_Points, 2)
all_vels_concat = np.concatenate(all_vels, axis=0)

# 3. 计算均值和标准差 (axis=0 表示对每个特征维度 x, y 分别计算)
vel_mean = np.mean(all_vels_concat, axis=0) # shape (2,)
vel_std = np.std(all_vels_concat, axis=0)   # shape (2,)
# 用于后续计算归一化? 

print(f"Velocity Mean: {vel_mean}")
print(f"Velocity Std: {vel_std}")

meta_data = {
    "channel_ids": list(channel_ids),
    "num_channel": int(num_channel),
    "channel_id_to_idx": channel_id_to_idx,
    "num_bin_each_channel": {trial['trial_id']:len(trial["spikes"]) for trial in sliced_trials}, 
    "max_bin": max_bin, # 最大的 bin 数量
    "max_token": max_token, # 最大的 token 数量
    "max_time_length": max_time_length, # 最大的时间长度
    "vel_mean": vel_mean.tolist(), # [mean_x, mean_y]
    "vel_std": vel_std.tolist()    # [std_x, std_y]
}

with open("processed_data/meta_data.json", "w") as f:
    json.dump(meta_data, f, indent=4)