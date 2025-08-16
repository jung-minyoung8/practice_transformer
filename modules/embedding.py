import torch
import math

def get_positional_encoding(seq_len,
                        input_size,
                        min_timescale=1.0,
                        max_timescale=1.0e4,
                        start_index=0,
                        device='cpu'):
    position = torch.arange(seq_len, dtype=torch.float, device=device)
    num_timescales = input_size // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(1, num_timescales - 1)
    inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float, device=device) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)  # 명시
    pos_en = torch.zeros(seq_len, input_size, device=device)
    pos_en[:, 0::2] = torch.sin(scaled_time)
    pos_en[:, 1::2] = torch.cos(scaled_time)
    return pos_en