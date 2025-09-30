import torch


class _Config:
    def __init__(self):
        self.config = {}

        self.config['device'] = "cpu"
        self.config['ensemble'] = False
        self.config['n_hp_search'] = 5
        if torch.cuda.is_available():
            print('GPU detected, switching to GPU compute by default')
            self.config['device'] = "cuda"

    def set(self, name, value):
        self.config[name] = value

    def get(self, name, default=None):
        return self.config.get(name, default=None)


global_config = _Config()
