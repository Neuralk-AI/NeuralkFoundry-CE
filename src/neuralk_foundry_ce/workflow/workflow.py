from pathlib import Path
import copy
import json

import joblib
import pandas as pd
import numpy as np
from typing import List
import warnings

from .step import Step
from .utils import notebook_display
from ..utils.logging import log


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across numpy, random, and common ML libraries.
    """
    import random
    import os

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


class WorkFlow:
    """
    Sequence of processing steps.

    This class manages an ordered list of `Step` instances, executing them sequentially.
    It optionally supports caching intermediate results to a specified directory.

    Parameters
    ----------
    steps : list of Step
        An ordered list of `Step` instances that define the workflow.

    cache_dir : pathlib.Path, optional (default=Path('./cache'))
        Directory path where intermediate results or artifacts can be cached.

    Attributes
    ----------
    steps : list of Step
        The sequence of steps to be executed in the workflow.

    cache_dir : pathlib.Path
        The directory used for caching intermediate results.
    """

    def __init__(self, steps, cache_dir=Path('./cache')):
        self.steps = steps
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir

    def check_consistency(self, init_keys: dict = {}):
        """
        Check that a sequence of steps can be executed in order without missing inputs.

        This function verifies that each step in the list has its declared input fields 
        available based on the outputs of previous steps and the initial input keys.

        Parameters
        ----------
        steps : list of Step
            The ordered list of Step instances representing the workflow.

        init_keys : dict, optional (default={})
            A dictionary representing the initial set of available input fields 
            before any steps are run. Only the keys are used.

        Raises
        ------
        KeyError
            If any step requires an input field that is not available at that point in the sequence.

        Returns
        -------
        None
            The function performs checks in-place and raises if an inconsistency is found.
        """
        sane = True
        available = set(init_keys)

        for i, step in enumerate(self.steps):
            if not isinstance(step, Step):
                raise TypeError(f"Step {step.name} is not a PipelineStep.")

            required, outputs = step.get_inputs_outputs()

            unavailable = required - available
            if unavailable:
                warnings.warn(f"Step {i} {step.name} requires unavailable fields: {unavailable}")
                sane = False

            # Update available fields
            available.update(outputs)
        
        return sane





    def run(self, init_data: dict) -> tuple[dict, dict]:
        """
        Execute the workflow sequentially on the provided input data.

        This method runs each step in order, passing the accumulated data forward.
        If a cached result exists for a step, it is loaded instead of recomputing.
        Metrics logged by each step are collected and returned.

        Parameters
        ----------
        init_data : dict
            Initial input data provided to the workflow. Keys must match the required
            inputs for the first step, unless `check_first_step` is disabled.

        Returns
        -------
        data : dict
            Final merged dictionary of all outputs produced throughout the workflow.

        metrics : dict of str to dict
            A dictionary mapping each step name to its logged metrics.

        Raises
        ------
        KeyError
            If any step is missing required input fields at runtime.
        """

        self.check_consistency(init_data.keys())

        data = copy.copy(init_data)
        metrics = {}
        cache_dir = self.cache_dir

        for i_step, step in enumerate(self.steps):
            step_id = f'{i_step}_{step.name}'

            # Check if the output exists which indicates that the step ran successfuly
            if cache_dir:
                cache_dir = cache_dir / step_id
                step.set_cache_dir(cache_dir)

            # In case the seed is not set in the step, best effort to ensure reproducibility
            set_seed(i_step)
            new_data = step.run(data)

            data.update(new_data)
            metrics[step.name] = copy.copy(step.logged_metrics)

        return data, metrics
    
    def set_parameter(self, parameter_name, value, set_all=True, verbose=1):
        steps_with_parameter = []
        for step in self.steps:
            for field in step.params:
                if field.name == parameter_name:
                    steps_with_parameter.append(step)
        if len(steps_with_parameter) == 0:
            log(verbose, 1, f"No step in the workflow has parameter {parameter_name}")
            return
        elif len(steps_with_parameter) > 1 and not set_all:
            log(verbose, 1, f"Multiple steps in the workflow have parameter {parameter_name}, stopping.")
        
        log(verbose, 1, f"Setting parameter {parameter_name} for all steps in the workflow")
        for step in steps_with_parameter:
            log(verbose, 2, f"Setting parameter {parameter_name} for step {step.name}")
            step.set_parameter(parameter_name, value)

    def display(self):
        return notebook_display(self.steps)
