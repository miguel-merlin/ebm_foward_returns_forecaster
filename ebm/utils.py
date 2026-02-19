import random
from typing import Dict, List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_data_by_time(
    data: Dict[str, torch.Tensor],
    n_assets: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    n_samples = data["past_prices"].shape[0]
    n_timesteps = n_samples // n_assets

    train_steps = int(n_timesteps * train_ratio)
    val_steps = int(n_timesteps * val_ratio)
    test_steps = n_timesteps - train_steps - val_steps
    if test_steps <= 0:
        raise ValueError(
            "Invalid split configuration. Ensure train_ratio + val_ratio < 1.0"
        )

    def _slice(start_step: int, end_step: int) -> Dict[str, torch.Tensor]:
        start = start_step * n_assets
        end = end_step * n_assets
        return {key: value[start:end] for key, value in data.items()}

    train_data = _slice(0, train_steps)
    val_data = _slice(train_steps, train_steps + val_steps)
    test_data = _slice(train_steps + val_steps, n_timesteps)
    return train_data, val_data, test_data


def build_hidden_dims(base_width: int, n_layers: int, decay: float) -> List[int]:
    dims = []
    for i in range(n_layers):
        dims.append(max(16, int(base_width * (decay**i))))
    return dims
