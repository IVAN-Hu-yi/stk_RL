# setup cuda, mps or cpu if available
import torch

class ppoconfig:
    def __init__(self, device, raw_dims):

        self.device = device
        Lmax = 32
        sequence_length = 32
        batch_size = 16
        update_interval = sequence_length * batch_size
        self.config = {
            "value_config": {
                "hidden_size": [64, 32],
                "hidden_activation": ["relu", "relu"],
                "output_activation": [""],
                "lr": 1e-4
            },
            "policy_config": {
                "disc_action": {
                    "hidden_size": [64, 32],
                    "hidden_activation": ["relu", "relu"],
                    "output_activation": [""]
                },
                "cont_action": {
                    "steer":{ 
                    "hidden_size": [64, 32],
                    "hidden_activation": ["relu", "relu"],
                    "output_activation": [""]
                    },
                    "acceleration":{ 
                    "hidden_size": [64, 32],
                    "hidden_activation": ["relu", "relu"],
                    "output_activation": [""]
                    },
                },
                "lr": 1e-4
            },
            'obsEncoder': {
                'boxEncoder': {
                    'output_dim': 32,
                    'hidden_size': [128, 64],
                    'hidden_activation': ['relu', 'relu'],
                    'output_activation': ['Linear']
                },
                'seqEncoder': {
                    'd_model': 64,
                    'n_heads': 2,
                    'n_layers': 2,
                    'dropout': 0.1,
                    'use_layer_norm': True
                }
            },
            'ReplayBuffer': {
                'capacity': 1e5,
                'sequence_length': sequence_length,
                'Lmax': Lmax,
                'device': device
            },
            'algo_config': {
                'ppo_epochs': 5,
                'gamma': 0.99,
                'lambda': 0.95,
                'clip_eps': 0.2,
                'Lmax': Lmax,
                "entropy_coef": 0.001,
            },
            "training_config": {
                "max_epochs": 10,
                "update_interval": update_interval,
                "batch_size": batch_size
            },
            "raw_dim": raw_dims

}
