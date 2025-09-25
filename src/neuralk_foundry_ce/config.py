import torch


class _Config:
    def __init__(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            print('GPU detected, switching to GPU compute by default')
            self.device = "cuda"

    def set_device(self, device: str):
        self.device = device

    def __repr__(self):
        return f"<Config device={self.device}>"

global_config = _Config()
