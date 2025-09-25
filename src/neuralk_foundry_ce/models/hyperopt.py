from typing import Callable, Dict, Any, List, Tuple, Optional
import threading
from copy import copy
import numpy as np
import heapq
import optuna
from ..utils.splitting import Split


class TopKOOFStore:
    """
    Thread-safe in-memory store that keeps only the top-K trials by score.
    Intended for single-objective studies.

    direction: 'maximize' or 'minimize'
    dtype:     convert stored OOF predictions to this dtype (e.g., np.float32) to save memory
    """
    def __init__(self, k, direction: str = "maximize", dtype=np.float32):
        assert direction in ("maximize", "minimize")
        self.k = int(k)
        self.direction = direction
        self.dtype = dtype

        # Min-heap on a comparable key, so the "worst" among kept items is at heap[0]
        # Heap items: (heap_key, score, trial_number, oof_array, params)
        self._heap: List[Tuple[float, float, int, np.ndarray, np.ndarray, Dict[str, Any]]] = []
        self._lock = threading.Lock()

    def _heap_key(self, score: float) -> float:
        # For maximize, smallest key = worst (lowest score)
        # For minimize, transform so that larger score => larger key; we want lowest key to be worst
        return score if self.direction == "maximize" else -score

    def add(self, *, score: float, trial_number: int, oof: np.ndarray, y: np.ndarray, params: Dict[str, Any] | None = None) -> bool:
        """
        Add a candidate (score + OOF). Returns True if it was kept (stored),
        False if it was discarded. If K is exceeded, the worst kept item is evicted.
        """
        if params is None:
            params = {}

        # Ensure contiguous & desired dtype to reduce memory
        oof = np.asarray(oof, dtype=self.dtype)

        item = (self._heap_key(score), float(score), int(trial_number), oof, y, dict(params))

        with self._lock:
            if len(self._heap) < self.k:
                heapq.heappush(self._heap, item)
                return True

            # If this candidate is better than the current worst, replace it
            if item[0] > self._heap[0][0]:
                heapq.heapreplace(self._heap, item)
                return True

            # Otherwise discard
            return False

    def topk(self) -> List[Dict[str, Any]]:
        """Return the kept items sorted from best to worst."""
        with self._lock:
            items = list(self._heap)

        # Sort by actual score depending on direction
        rev = (self.direction == "maximize")
        items.sort(key=lambda t: t[1], reverse=rev)

        # Materialize as dicts
        out = []
        for _, score, trial_number, oof, y, params in items:
            out.append({
                "score": score,
                "trial_number": trial_number,
                "oof": oof,              # np.ndarray
                "y": y,                  # np.ndarray
                "params": params,        # dict
            })
        return out

    def clear(self) -> None:
        with self._lock:
            self._heap.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)


def _as_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure predictions are 2D: (n_samples, n_targets)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _extract_truth(store_items: List[Dict[str, Any]]) -> np.ndarray:
    """
    Returns the ground truth used across models, check that it's consistent.
    """
    if not store_items:
        raise ValueError("Store is empty; nothing to ensemble.")
    y_trues = [_as_2d(item["y"]) for item in store_items]

    y_ref = None
    for i, y in enumerate(y_trues):
        if y_ref is None:
            y_ref = y
            continue
        if not np.allclose(y, y_ref):
            raise ValueError(f"Ground truth {i} is inconsistent with others")
    return y_ref

def _stack_store_items(store_items: List[Dict[str, Any]]) -> np.ndarray:
    """
    Build (n_models, n_samples, n_targets) stack from TopKOOFStore.topk() items.
    Raises if shapes are inconsistent.
    """
    if not store_items:
        raise ValueError("Store is empty; nothing to ensemble.")
    preds = [_as_2d(item["oof"]) for item in store_items]
    n_samples, n_targets = preds[0].shape
    for i, p in enumerate(preds):
        if p.shape != (n_samples, n_targets):
            raise ValueError(f"Model {i} OOF shape {p.shape} != ({n_samples},{n_targets})")
    return np.stack(preds, axis=0)  # (M, N, C)


def _weighted_mean(preds_stack: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Weighted mean across models with counts as non-negative weights.
    preds_stack: (M, N, C), counts: (M,)
    Returns: (N, C)
    """
    w = counts.astype(np.float64)
    s = w.sum()
    if s == 0:
        return np.zeros(preds_stack.shape[1:], dtype=np.float64)
    return np.tensordot(w / s, preds_stack, axes=(0, 0))


def _best_single_model(
    y: np.ndarray,
    preds_stack: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    maximize: bool,
) -> Tuple[int, float]:
    """Return (best_model_index, best_score) for single models."""
    scores = [metric(y_true=y, y_score=preds_stack[m]) for m in range(preds_stack.shape[0])]
    idx = int(np.argmax(scores) if maximize else np.argmin(scores))
    return idx, float(scores[idx])


def _greedy_iteration(
    y: np.ndarray,
    preds_stack: np.ndarray,
    counts: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    current_best: float,
    maximize: bool,
) -> Tuple[Optional[int], float]:
    """
    Try adding one count to each model; return (best_model_to_add or None, new_best_score).
    If no improvement, returns (None, current_best).
    """
    best_m = None
    best_score = current_best
    for m in range(preds_stack.shape[0]):
        counts[m] += 1
        y_pred = _weighted_mean(preds_stack, counts)
        score = metric(y_true=y, y_score=y_pred)
        counts[m] -= 1
        better = (score > best_score) if maximize else (score < best_score)
        if better:
            best_score = score
            best_m = m
    return best_m, float(best_score)


def caruana_ensemble_selection(
    store_items: List[Dict[str, Any]],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_iters: int = 50,
    early_stop: Optional[int] = 10,
    maximize: bool = True,
    warm_start: bool = True,
) -> Dict[str, Any]:
    """
    Greedy ensemble selection with replacement (Caruana et al., 2004).

    Parameters
    ----------
    store_items : list of dicts
        Output of TopKOOFStore.topk(); must include 'oof' arrays (N x C).
    metric : callable
        metric(y_true, y_pred) -> float, larger is better if maximize=True.
    n_iters : int
        Number of greedy iterations (ensemble size with replacement).
    early_stop : int or None
        Stop if no improvement for this many iterations. None disables.
    maximize : bool
        If False, the metric is minimized.
    warm_start : bool
        If True, start from the best single model. If False, start from empty.

    Returns
    -------
    dict with keys:
        - 'weights'      : np.ndarray (M,) normalized to sum to 1
        - 'counts'       : np.ndarray (M,) integer selection counts
        - 'order'        : list[int] sequence of selected model indices (length <= n_iters)
        - 'best_score'   : float metric value achieved by the ensemble on OOF
        - 'oof_pred'     : np.ndarray (N, C) ensembled OOF predictions
    """
    y = _extract_truth(store_items)
    preds_stack = _stack_store_items(store_items)  # (M, N, C)
    M = preds_stack.shape[0]

    counts = np.zeros(M, dtype=np.int64)
    order: List[int] = []

    if warm_start:
        m0, best_score = _best_single_model(y, preds_stack, metric, maximize)
        counts[m0] += 1
        order.append(m0)
    else:
        best_score = -np.inf if maximize else np.inf

    no_improve = 0
    for _ in range(1 if warm_start else 0, n_iters):
        best_m, new_best = _greedy_iteration(
            y=y,
            preds_stack=preds_stack,
            counts=counts,
            metric=metric,
            current_best=best_score,
            maximize=maximize,
        )

        if best_m is None:
            no_improve += 1
            if early_stop is not None and no_improve >= early_stop:
                break
            # keep ensemble size growing (plateau): repeat last choice if warm-started
            if order:
                counts[order[-1]] += 1
                order.append(order[-1])
        else:
            counts[best_m] += 1
            order.append(best_m)
            best_score = new_best
            no_improve = 0

    # Final outputs
    weights = counts.astype(np.float64)
    wsum = weights.sum()
    weights = weights / wsum if wsum > 0 else weights

    return {
        "weights": weights,
        "counts": counts,
        "order": order,
        "params": [item['params'] for item in store_items],
    }


def objective(
    trial,
    model,
    X: np.array,
    y: np.array,
    splits: List,
    metrics: List,
    inputs,
    store: TopKOOFStore=None,
):
    """
    Objective function for hyperparameter optimization using Optuna.

    This function applies preprocessing (if provided), trains the model on
    multiple train/validation splits, and evaluates it using the provided metrics.
    It returns the average validation score used to guide the optimization.

    Parameters
    ----------
    trial : int
        The current trial object provided by Optuna, used to sample hyperparameters.
    model : BasePredictModel
        The predictive model to be optimized.
    X : np.array
        Input feature matrix.
    y : np.array
        Target labels.
    splits : list
        List of dictionaries containing boolean masks for train and validation splits.
    metrics : list
        List of scoring functions to evaluate model performance.
    inputs : dict
        Additional inputs passed to the model (e.g., metadata, configuration).

    Returns
    -------
    float
        The validation metric to be optimized (e.g., R-squared for regression or ROC-AUC
        for classification). Returns negative infinity for failed trials in classification
        tasks and positive infinity for failed trials in regression tasks.
    Notes
    -----
    - Metrics are computed for each fold and averaged to obtain the final validation metric.
    - If an exception occurs during training or evaluation, the trial is marked as failed.
    """    """Objective function for Optuna hyperparameter optimization, including preprocessing."""
    
    model_params = model.get_model_params(trial, inputs) | model.get_fixed_params(inputs)

    # Copy the model for parallelization
    model.model = None
    model = copy(model)

    model.init_model(model_params)
    metric_to_optimize = inputs['metric_to_optimize']

    metric_results_val_list = {metric_name: [] for metric_name in metrics.keys()}
    y_scores = []
    y_trues = []
    for split_mask in splits:
        model.train(X, y=y, split_mask=split_mask, splits=[Split.TRAIN])

        y_val = y[np.isin(np.array(split_mask), [Split.VAL])]
        y_trues.append(y_val)
        y_pred = model.forward(X, split_mask=split_mask, splits=[Split.VAL])
        outputs = {}
        outputs['y_pred'] = y_pred
        outputs['y_true'] = y_val
        if 'y_score' in model.extras:
            outputs['y_score'] = model.extras['y_score']
            y_scores.append(model.extras['y_score'])
        if 'y_classes' in inputs:
            outputs['labels'] = np.arange(len(inputs['y_classes']))

        for metric_name, metric in metrics.items():
            metric_results_val_list[metric_name].append(metric(**outputs))
        
        if hasattr(model, 'clean_model'):
            model.clean_model()
    
    metric_results_val = {}
    for metric_name, values in metric_results_val_list.items():
        metric_results_val[metric_name] = np.mean(values)

    validation_metric = metric_results_val[metric_to_optimize]
    trial.set_user_attr("metric_results", metric_results_val)

    if store is not None:
        y_scores, y_trues = np.concatenate(y_scores, axis=0), np.concatenate(y_trues, axis=0)
        assert(y_scores.shape[0] == y_trues.shape[0])
        store.add(
            score=validation_metric, trial_number=trial.number, oof=y_scores,
            y=y_trues, params=model_params
        )
    
    return validation_metric


def hyperoptimize(optuna_kwargs, model, X, y, splits, metrics, inputs, n_ensemble=None, verbose=0):

    metric_to_optimize = inputs['metric_to_optimize']
    direction = "maximize" if metrics[metric_to_optimize].maximize else "minimize"
    store = None
    if n_ensemble:
        store = TopKOOFStore(k=n_ensemble, direction=direction)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)

    study.optimize(
        lambda trial: objective(
            trial,
            model,
            X, y,
            splits,
            metrics,
            inputs,
            store=store
        ),
        **optuna_kwargs
    )

    ensemble = None
    if n_ensemble:
        ensemble = caruana_ensemble_selection(store.topk(), metrics[metric_to_optimize])

    return study.best_trial, ensemble
