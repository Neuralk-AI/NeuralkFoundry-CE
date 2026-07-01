from .base import ClassifierModel
from ...utils.splitting import with_masked_split
from ...config import global_config


# Foundation-model weights are loaded once per process and shared across all
# TabFMClassifier instances (folds, datasets). Keyed by backend name so users
# can switch between jax/pytorch without re-downloading.
_TABFM_BASE_MODEL = {}


def _load_base_model(backend: str):
    key = (backend, global_config.device)
    if key not in _TABFM_BASE_MODEL:
        if backend == 'pytorch':
            from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
            _TABFM_BASE_MODEL[key] = tabfm_v1_0_0.load(device=global_config.device)
        elif backend == 'jax':
            from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
            _TABFM_BASE_MODEL[key] = tabfm_v1_0_0.load()
        else:
            raise ValueError(f"Unknown TabFM backend: {backend!r}")
    return _TABFM_BASE_MODEL[key]


class TabFMClassifier(ClassifierModel):
    """Apply Google's TabFM foundation model to tabular data.

    The pretrained backbone is loaded once per Python process and shared
    across every TabFMClassifier instance (folds, datasets) to avoid the
    multi-hundred-MB reload each time.
    """
    name = 'tabfm-classifier'

    def __init__(self, backend: str = 'pytorch'):
        super().__init__()
        self.tunable = False
        self.backend = backend

    @with_masked_split
    def train(self, X, y):
        self.model.fit(X, y)

    @with_masked_split
    def forward(self, X):
        self.extras['y_score'] = self.model.predict_proba(X)
        return self.model.predict(X)

    def init_model(self, config):
        from tabfm import TabFMClassifier as _TabFMClassifier
        base = _load_base_model(self.backend)
        self.model = _TabFMClassifier(model=base, **config)

    def get_fixed_params(self, inputs):
        return {}

    def get_model_params(self, trial, inputs):
        return {}
