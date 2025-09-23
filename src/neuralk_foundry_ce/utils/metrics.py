from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy, rankdata
from scipy.special import huber
from sklearn.metrics import (
    log_loss,
    recall_score,
    precision_score,
    f1_score,
    hinge_loss,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    ndcg_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    jaccard_score,
    accuracy_score
)
from scipy.spatial.distance import cdist, pdist, euclidean


class Metric(ABC):
    desc = ""
    maximize = True

    @abstractmethod
    def __call__(self, y_true, y_pred, **kwargs):
        """Computes metric value."""
        pass

    def _requires(self, kwargs, *args):
        """Takes a list of inputs and ensures that all inputs are present in kwargs."""
        ret = []
        for arg in args:
            if arg not in kwargs:
                raise ValueError(f"Missing required argument for metric {self.desc}: {arg}")
            ret.append(kwargs[arg])
        return ret


# Classification metrics



class CrossEntropy(Metric):
    desc = "Cross-entropy loss between true and predicted distributions."
    maximize = False

    def __call__(self, y_true, y_pred, **kwargs):
        """
        Computes the cross-entropy loss between the true and predicted distributions.
        Parameters
        ----------
        y_true : array-like
            Ground truth (one-hot or labels).

        y_pred : array-like
            Predicted probabilities.

        Returns
        -------
        results : float
            Cross-entropy loss.
        """
        if not 'y_score' in kwargs:
            raise ValueError("y_score must be provided for cross entropy calculation.")
        y_score = kwargs['y_score']
        if len(y_score.shape) == 2 and y_score.shape[1] == 2 and y_score.shape[1] > 1:
            y_score = y_score[:, 1]
        return log_loss(y_true, y_score, labels=kwargs.get('labels', None))


class AccuracyScore(Metric):
    desc = "Classification accuracy."
    maximize = True

    def __call__(self, **kwargs):
        """
        Computes the cross-entropy loss between the true and predicted distributions.
        Parameters
        ----------
        y_true : array-like
            Ground truth (one-hot or labels).

        y_pred : array-like
            Predicted probabilities.

        Returns
        -------
        results : float
            Cross-entropy loss.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        return accuracy_score(y_true, y_pred)


class KLDivergence(Metric):
    desc = "KL Divergence between true and predicted distributions."
    maximize = False

    def __call__(self, **kwargs):
        """
        Computes the KL Divergence between the true and predicted distributions.

        Parameters
        ----------
        y_true : array-like
            Ground truth distribution.

        y_pred : array-like
            Predicted distribution.

        Returns
        -------
        results : float
            KL Divergence.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        return np.sum(entropy(y_true, y_pred, axis=-1))


class Recall(Metric):
    desc = "Recall score."
    maximize = True

    def __call__(self, **kwargs):
        """
        Computes recall score.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.

        y_pred : array-like
            Predicted labels.

        Returns
        -------
        results : float
            Recall score.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        return recall_score(
            y_true, y_pred,
            average='binary' if len(np.unique(y_true)) == 2 else 'micro',
            labels=kwargs.get('labels', None))


class Precision(Metric):
    desc = "Precision score."
    maximize = True

    def __call__(self, **kwargs):
        """
        Computes precision score.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.

        y_pred : array-like
            Predicted labels.

        Returns
        -------
        results : float
            Precision score.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        return precision_score(
            y_true, y_pred,
            average='binary' if len(np.unique(y_true)) == 2 else 'micro',
            labels=kwargs.get('labels', None),
            zero_division=0)


class F1ScoreMetric(Metric):
    desc = "F1 score."
    maximize = True

    def __call__(self, **kwargs):
        """
        Computes the F1 score.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.

        y_pred : array-like
            Predicted labels.

        Returns
        -------
        results : float
            F1 score.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        return f1_score(
            y_true, y_pred, 
            average='binary' if len(np.unique(y_true)) == 2 else 'macro',
            labels=kwargs.get('labels', None),
            zero_division=0)


class HingeLossMetric(Metric):
    desc = "Hinge loss for binary classification."
    maximize = False

    def __call__(self, **kwargs):
        """
        Computes hinge loss for binary classification.

        Parameters
        ----------
        y_true : array-like
            Ground truth binary labels.

        y_pred : array-like
            Predicted scores.

        kwargs : dict
            Must include 'y_score' for hinge loss calculation.            

        Returns
        -------
        results : float
            Hinge loss.
        """
        y_true, y_score = self._requires(kwargs, "y_true", "y_score")
        if len(y_score.shape) == 2 and y_score.shape[1] == 2 and y_score.shape[1] > 1:
            y_score = y_score[:, 1]
        return hinge_loss(y_true, y_score, labels=kwargs.get('labels', None))


class ROCAUC(Metric):
    desc = "ROC-AUC score"
    maximize = True

    def __call__(self, **kwargs):
        """
        Computes the ROC-AUC score.

        Parameters
        ----------
        y_true : array-like
            Ground truth binary labels.

        y_pred : array-like
            Predicted probabilities.

        Returns
        -------
        results : float
            ROC-AUC score.
        """
        y_true, y_score = self._requires(kwargs, "y_true", "y_score")
        if len(y_score.shape) == 2 and y_score.shape[1] == 2 and y_score.shape[1] > 1:
            y_score = y_score[:, 1]
        return roc_auc_score(
            y_true, y_score, multi_class='ovo', 
            labels=kwargs.get('labels', None))


# Regression metrics


class L1Loss(Metric):
    desc = "L1 loss (mean absolute error) for regression tasks."
    maximize = False

    def __call__(self, **kwargs):
        """
        Computes the L1 loss (mean absolute error) for regression tasks.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.

        y_pred : array-like
            Predicted values.

        Returns
        -------
        results : float
            L1 loss.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        return mean_absolute_error(y_true, y_pred)


class L2Loss(Metric):
    desc = "L2 loss (mean squared error) for regression tasks."
    maximize = False

    def __call__(self, **kwargs):
        """
        Computes the L2 loss (mean squared error) for regression tasks.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.

        y_pred : array-like
            Predicted values.

        Returns
        -------
        results : float
            L2 loss.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        return mean_squared_error(y_true, y_pred)


class HuberLoss(Metric):
    desc = "Huber loss for regression tasks."
    maximize = False

    def __call__(self, **kwargs):
        """
        Computes the Huber loss for regression tasks.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.

        y_pred : array-like
            Predicted values.

        delta : float
            Threshold where linear behavior transitions to quadratic.

        Returns
        -------
        results : float
            Huber loss.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        huber_delta = kwargs.get("huber_delta", 1.0)

        losses = huber(huber_delta, y_true - y_pred)
        return np.mean(losses)


class RSquared(Metric):
    desc = "R-squared (coefficient of determination) score."
    maximize = True

    def __call__(self, **kwargs):
        """
        Computes the R-squared (coefficient of determination) score.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.

        y_pred : array-like
            Predicted values.

        Returns
        -------
        results : float
            R-squared score.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        return r2_score(y_true, y_pred)


# Clsutering metrics


class Silhouette(Metric):
    desc = "Silhouette Coefficient for clustering tasks."
    maximize = True

    def __call__(self, **kwargs):
        """
        Computes the Silhouette Coefficient for clustering tasks.

        Parameters
        ----------
        y_true : array-like
            Feature data.

        y_pred : array-like
            Cluster labels.

        Returns
        -------
        results : float
            Silhouette Coefficient.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        return silhouette_score(y_true, y_pred, **kwargs)


class AdjustedRandIndex(Metric):
    desc = "Adjusted Rand Index (ARI) for clustering tasks."
    maximize = True

    def __call__(self, **kwargs):
        """
        Computes the Adjusted Rand Index (ARI) for clustering tasks.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.

        y_pred : array-like
            Cluster labels.

        Returns
        -------
        results : float
            Adjusted Rand Index.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        return adjusted_rand_score(y_true, y_pred)


class NormalizedMutualInfo(Metric):
    desc = "Normalized Mutual Information (NMI) for clustering tasks."
    maximize = True

    def __call__(self, **kwargs):
        """
        Computes the Normalized Mutual Information (NMI) for clustering tasks.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.

        y_pred : array-like
            Cluster labels.

        Returns
        -------
        results : float
            Normalized Mutual Information.
        """
        y_true, y_pred = self._requires(kwargs, "y_true", "y_pred")
        return normalized_mutual_info_score(y_true, y_pred)




def _preprocess_linkage_task(y_true, y_pred):
    _, all_pairs = y_pred
    y_true_ = [y_true[i] == y_true[j] for i, j in all_pairs]
    return y_true_


class LinkageAveragePrecision(Metric):
    desc = "Linkage Average Precision."
    maximize = True

    def __call__(self, y_true, y_pred, **kwargs):
        if not 'y_pairs' in kwargs:
            raise ValueError("y_pairs must be provided for Linkage metrics.")
        if not 'all_pairs' in kwargs:
            raise ValueError("all_pairs must be provided for Linkage metrics.")
        y_pairs, all_pairs = kwargs['y_pairs'], kwargs['all_pairs']
        all_pairs = [p for y, p in zip(y_pairs, all_pairs) if y == 1]
        if len(all_pairs) == 0:
            return 0.
        y_score = np.mean([y_true[i] == y_true[j] for i, j in all_pairs])
        return y_score

class LinkageAverageRecall(Metric):
    desc = "Linkage Average Recall."
    maximize = True

    def __call__(self, y_true, y_pred, **kwargs):
        if not 'y_pairs' in kwargs:
            raise ValueError("y_pairs must be provided for Linkage metrics.")
        if not 'all_pairs' in kwargs:
            raise ValueError("all_pairs must be provided for Linkage metrics.")
        y_pairs, all_pairs = kwargs['y_pairs'], kwargs['all_pairs']
        y_pos = [y for y, (i, j) in zip(y_pairs, all_pairs) if y_true[i] == y_true[j]]
        if len(y_pos) == 0:
            return 0.
        y_score = np.mean(y_pos)
        return y_score
    

class JaccardSimilarity(Metric):
    desc = "Jaccard Similarity."
    maximize = True

    def __call__(self, y_true, y_pred, **kwargs):
        """
        Compute the Jaccard similarity coefficient score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth (correct) target values.

        y_pred : array-like of shape (n_samples,)
            Estimated target values.

        Returns
        -------
        score : float
            Jaccard similarity coefficient score, which is a value between 0 and 1.
            A score of 1 indicates perfect similarity, while 0 indicates no similarity.

        Notes
        -----
        This method uses the `jaccard_score` function from `sklearn.metrics` to compute
        the similarity score.
        """
        return jaccard_score(y_true, y_pred, average='micro')


def is_duplicated(arr):
    unique, counts = np.unique(arr, return_counts=True)
    duplicates = unique[counts > 1]
    return np.isin(arr, duplicates)


class GroupDetectionPrecision(Metric):
    desc = "Group Detection Precision."
    maximize = True

    def __call__(self, y_true, y_pred, **kwargs):
        d_true = is_duplicated(y_true)
        d_pred = is_duplicated(y_pred)
        return precision_score(d_true, d_pred)


class GroupDetectionRecall(Metric):
    desc = "Group Detection Recall."
    maximize = True

    def __call__(self, y_true, y_pred, **kwargs):
        d_true = is_duplicated(y_true)
        d_pred = is_duplicated(y_pred)
        return recall_score(d_true, d_pred)
