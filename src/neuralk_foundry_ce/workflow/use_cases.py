from collections import OrderedDict
from typing import List

from .workflow import WorkFlow
from .utils import notebook_display
from ..sample_selection.splitter.stratified_shuffle import StratifiedShuffleSplitter
from ..feature_engineering.vectorizer import TfidfVectorizer
from ..feature_engineering.preprocessing import ColumnTypeDetection, CategoricalPreprocessor, NumericalPreprocessor, LabelEncoder
from ..models.classifier import XGBoostClassifier


class Classification():

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.steps = OrderedDict()
        self.steps['Splitting'] = StratifiedShuffleSplitter()
        self.steps['Feature engineering'] = [
            ColumnTypeDetection(),
            CategoricalPreprocessor(),
            NumericalPreprocessor(),
            TfidfVectorizer(),
            LabelEncoder(),
        ]
        self.steps['Classifier'] = XGBoostClassifier()
        self.parameters = {}

    def notebook_display(self, level=None):
        return notebook_display(self.steps, level)

    def set_classifier(self, classifier):
        self.steps['Classifier'] = classifier

    def set_parameter(self, parameter_name, value):
        self.parameters[parameter_name] = value

    def _run(self, init_data):
        steps = [step for v in self.steps.values() for step in (v if isinstance(v, list) else [v])]
        workflow = WorkFlow(steps, cache_dir=self.cache_dir)
        for parameter_name, value in self.parameters.items():
            workflow.set_parameter(parameter_name, value)
        metric = {'metric_to_optimize': 'roc_auc', 'task': 'classification'}
        return workflow.run(init_data=init_data | metric)


    def run(self, X, y, fold_index=0, seed=0) -> tuple[dict, dict]:
        return self._run({'X': X, 'y': y, 'fold_index': fold_index, 'seed': seed})


class Categorisation(Classification):

    def run(self, X, y, fold_index=0, seed=0) -> tuple[dict, dict]:
        steps = [step for v in self.steps.values() for step in (v if isinstance(v, list) else [v])]
        workflow = WorkFlow(steps, cache_dir=self.cache_dir)
        metric = {'metric_to_optimize': 'f1_score', 'task': 'classification'}
        return workflow.run(init_data={'X': X, 'y': y, 'fold_index': fold_index, 'seed': seed} | metric)
