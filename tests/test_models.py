import pandas as pd
import numpy as np
from neuralk_foundry_ce.models.classifier import MLPClassifier  # adjust as needed
from neuralk_foundry_ce.utils.splitting import Split


def test_mlp_unknown_category_handling():
    # Create toy dataset with categorical features
    df_train = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'blue', 'red'],
        'shape': ['circle', 'square', 'triangle', 'square', 'circle'],
        'value': [1.2, 3.4, 2.1, 3.3, 1.1],
        'target': [0, 1, 0, 1, 0],
    })

    df_test = pd.DataFrame({
        'color': ['red', 'yellow'],  # yellow is unseen
        'shape': ['circle', 'triangle'],  # all seen
        'value': [1.0, 2.2],
    })

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    categorical_features = ['color', 'shape']
    for col in categorical_features:
        df_all[col] = df_all[col].astype('category')

    # Simulate split mask
    split_mask = np.array([Split.TRAIN] * len(df_train) + [Split.TEST] * len(df_test))

    model = MLPClassifier()
    model.extras = {}
    config = {
        'categorical_features': categorical_features,
        'activation': 'relu',
        'dropout': 0.0,
        'batchnorm': False,
        'n_hidden_layers': 1,
        'optimizer': 'adamw',
        'lr': 1e-3,
        'epochs': 5,
        'use_unknown_category': True,
        'simulate_unknowns': 0.5,  # high to test easily
    }

    inputs = {
        'X': df_all.drop(columns='target', errors='ignore'),
        'y': df_all['target'].fillna(0).astype(int),
    }

    config.update(model.get_fixed_params(inputs))
    model.init_model(config)

    model.train(inputs['X'], inputs['y'], split_mask, splits=[Split.TRAIN])

    # Check that only 'color' uses unknown category (has unseen value "yellow")
    assert model.include_unknown_index['color'] is True
    assert model.include_unknown_index['shape'] is False

    # Forward pass (on test set)
    y_pred = model.forward(inputs['X'], split_mask=split_mask, splits=[Split.TEST])
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(df_test)

    print("Test passed: unknown category logic behaves as expected.")

def test_hyperopt_refit_keeps_the_fixed_params():
    # After tuning, base.py refits the model from the best trial's sampled parameters.
    # get_fixed_params carries the objective, the metric and the verbosity, none of which
    # Optuna records, so a refit built from best_trial.params alone trains a differently
    # configured model than every trial that was scored.
    from neuralk_foundry_ce.config import global_config
    from neuralk_foundry_ce.models.classifier import LightGBMClassifier
    from neuralk_foundry_ce.utils.splitting import Split

    seen = []

    class Recorder(LightGBMClassifier):
        def init_model(self, config):
            seen.append(dict(config))
            super().init_model(config)

    rng = np.random.RandomState(0)
    n = 300
    X = rng.randn(n, 8)
    y = (X[:, 0] + X[:, 1] - X[:, 2] + rng.randn(n) * 0.3 > 0).astype(int)
    split = np.empty(n, dtype=object)
    split[:] = Split.TRAIN
    split[int(n * 0.6):int(n * 0.8)] = Split.VAL
    split[int(n * 0.8):] = Split.TEST

    global_config.set('n_hyperopt_trials', 2)
    global_config.set('ensemble', False)
    model = Recorder()
    model.n_ensemble = None
    model.logged_metrics = {}
    model._returned_outputs = {}
    model._execute({'X': X, 'y': y, 'splits': [split], 'metric_to_optimize': 'roc_auc'})

    assert len(seen) >= 2, 'expected at least one trial and one refit'
    refit = seen[-1]
    for key in ('objective', 'metric', 'verbose'):
        assert key in refit, (
            f'the post-tuning refit dropped {key!r} from get_fixed_params; it trains a '
            f'differently configured model than the trials that were scored'
        )


def test_every_model_exposes_get_fixed_params():
    # _execute and hyperopt both call get_fixed_params unconditionally, so a model without
    # one raises AttributeError before any model is built. Only the classifiers defined it.
    from neuralk_foundry_ce.models.base import BaseModel
    from neuralk_foundry_ce.models.regressor.lightgbm import LightGBMRegressor
    from neuralk_foundry_ce.models.regressor.xgboost import XGBoostRegressor
    from neuralk_foundry_ce.models.regressor.catboost import CatBoostRegressor

    assert hasattr(BaseModel, 'get_fixed_params'), (
        'the contract is called unconditionally, so it belongs on the base class'
    )
    for cls in (LightGBMRegressor, XGBoostRegressor, CatBoostRegressor):
        assert isinstance(cls().get_fixed_params({}), dict), (
            f'{cls.__name__} cannot supply fixed params, so hyperopt raises before fitting'
        )
