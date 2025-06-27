from enum import IntEnum

import numpy as np


class Split(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    NONE = 3


def with_masked_split(fn):
    """
    Decorator for applying masking to inputs regarding the split to use.
            
    Examples
    --------
    >>> @with_masked_split
    ... def my_function(X, y=None):
    ...     return len(X)
    ... 
    >>> # Use with split masking
    >>> result = my_function(X, y, split_mask=split_mask, splits=[Split.TRAIN])
    """
    def wrapper(self, X, y=None, split_mask=None, splits=[Split.TRAIN]):
        mask = np.isin(np.array(split_mask), splits)
        if y is not None:
            return fn(self, X[mask], y[mask])
        else:
            return fn(self, X[mask])
        
    return wrapper


class RareClassSafeSplitter:
    """
    A wrapper around scikit-learn splitters that handles classes with <2 samples
    by duplicating them temporarily to satisfy stratification constraints.
    Ensures that rare class samples are included in the training set.

    Parameters
    ----------
    base_splitter : object
        A scikit-learn splitter instance (e.g., StratifiedShuffleSplit, StratifiedKFold).
    """

    def __init__(self, base_splitter):
        self.base_splitter = base_splitter

    def split(self, X, y, *args, **kwargs):
        X = np.array(X)
        y = np.array(y)

        # Identify rare classes
        unique, counts = np.unique(y, return_counts=True)
        rare_classes = unique[counts < 2]

        if len(rare_classes) == 0:
            yield from self.base_splitter.split(X, y, *args, **kwargs)
            return

        # Duplicate rare class samples
        X_aug = X.copy()
        y_aug = y.copy()
        duplicated_indices = []

        for rare_class in rare_classes:
            idx = np.where(y == rare_class)[0][0]
            X_aug = np.vstack([X_aug, X[idx:idx+1]])
            y_aug = np.concatenate([y_aug, [rare_class]])
            duplicated_indices.append(len(y_aug) - 1)

        # Perform split on augmented data
        for train_idx_aug, test_idx_aug in self.base_splitter.split(X_aug, y_aug, *args, **kwargs):
            train_idx = np.setdiff1d(train_idx_aug, duplicated_indices)
            test_idx = np.setdiff1d(test_idx_aug, duplicated_indices)

            # Ensure rare class samples are in train
            for rare_class in rare_classes:
                orig_idx = np.where(y == rare_class)[0][0]
                if orig_idx not in train_idx:
                    test_idx = test_idx[test_idx != orig_idx]
                    train_idx = np.append(train_idx, orig_idx)

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.base_splitter.get_n_splits(X, y, groups)