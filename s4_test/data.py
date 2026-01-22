import torch

def make_delay_data(batch_size, seq_len, delay, D=1):
    '''
    生成测试用的数据
    '''
    x = torch.randn(batch_size, seq_len, D)
    y = torch.zeros_like(x)
    y[:, delay:] = x[:, :-delay]
    return x, y
