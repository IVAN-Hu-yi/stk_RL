# this file is to make raw observations ready for obsEncoder
import torch

def pad_seq(data, Lmax):
    '''
     pad a sequence of tensors to length Lmax for each time step

     data: list of tensors shape (Seq_len, D), with size equals to the episode length
     returns:
        padded: tensor of shape (episode_length, Lmax, D)
        pad_seq_mask: tensor of shape (episode_length, Lmax) with 1 for valid items in Sequence-Type obs and 0 for padded items
    '''
    D = data[0].shape[1:]

    episode_length = len(data)
    device = data[0].device

    padded = torch.zeros((episode_length, Lmax, *D), device=device)
    pad_seq_mask = torch.zeros((episode_length, Lmax), dtype=torch.bool, device=device)

    for i, item in enumerate(data):
        L_use = min(item.shape[0], Lmax)
        if len(D) == 0:
            padded[i, :L_use] = item[:L_use]
        else:
            padded[i, :L_use, ...] = item[:L_use, ...]
        pad_seq_mask[i, :L_use] = True

    return padded, pad_seq_mask


def singleObsWrapper(raw_obs, Lmax=64):

    """
        wrap a single raw observation into box_obs, seq_obs, seq_obs_mask for obsEncoder
    """
    box_obs_keys = ['attachment_time_left', 'aux_ticks', 'center_path', 'center_path_distance', 'distance_down_track', 'energy', 'front', 'max_steer_angle', 'shield_time', 'skeed_factor', 'velocity']

    seq_obs_keys = ['items_position', 'items_type', 'karts_position', 'paths_distance', 'paths_end', 'paths_start', 'paths_width']

    box_obs = []
    seq_obs = {key: None for key in seq_obs_keys}
    seq_obs_mask = {key: None for key in seq_obs_keys}

    for key in raw_obs.keys():
        if key in box_obs_keys:
            box_obs.append(torch.tensor(raw_obs[key]))
        elif key in seq_obs_keys:
            temp = [torch.tensor(raw_obs[key])]
            padded, pad_seq_mask = pad_seq(temp, Lmax)
            seq_obs[key] = padded
            seq_obs_mask[key] = pad_seq_mask

    for key in seq_obs.keys():
        if len(seq_obs[key].shape) == 2:
            seq_obs[key] = seq_obs[key].unsqueeze(0).unsqueeze(-1)
        elif len(seq_obs[key].shape) == 3:
            seq_obs[key] = seq_obs[key].unsqueeze(0)

        if len(seq_obs_mask[key].shape) == 2:
            seq_obs_mask[key] = seq_obs_mask[key].unsqueeze(0)

    return box_obs, seq_obs, seq_obs_mask
        


    

