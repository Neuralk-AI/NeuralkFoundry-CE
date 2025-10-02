import torch


class _Config:
    def __init__(self):
        self.device = "cpu"
        self.ensemble = True
        self.n_hyperopt_trials = 200
        if torch.cuda.is_available():
            print('GPU detected, switching to GPU compute by default')
            self.device = "cuda"

    def set_device(self, device):
        self.device = device

    def set(self, name, value):
        setattr(self, name, value)

    def get(self, name, default=None):
        return getattr(self, name, default)


global_config = _Config()
