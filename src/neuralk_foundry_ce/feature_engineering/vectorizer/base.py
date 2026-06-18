from ...workflow import Step, Field


class BaseVectorizer(Step):
    """
    Base class for all vectorization steps in the pipeline.

    This class provides a template for transforming raw input features `X`
    (optionally using target `y`) into vectorized representations. Subclasses
    should implement the `forward` method to define the actual vectorization logic.

    Attributes
    ----------
    name : str
        Name of the step, set to 'vectorizer'.

    Inputs
    ------
    X : array-like or DataFrame
        The input feature matrix.
    y : array-like, optional
        The target labels (may be used in supervised vectorization).

    Outputs
    -------
    X : array-like or DataFrame
        The transformed feature matrix.
    num_features : int
        The number of output features (columns) in the transformed matrix.

    Methods
    -------
    forward(X, y=None):
        To be implemented by subclasses. Applies the vectorization logic.
    """
    name = 'vectorizer'
    # Vectorization outputs are reproducible and cheap to regenerate; skip
    # caching them to keep workflow outputs compact.
    cache_outputs = False
    inputs = [
        Field('X', 'Input features of the dataset'),
        Field('y', 'Target variable to predict'),
    ]
    outputs = [
        Field('X', 'Input features with vectorized features'),
    ]

    def _execute(self, inputs: dict):
        """
        Run the vectorization step.

        Parameters
        ----------
        inputs : dict
            Dictionary with keys "X" (input features) and optionally "y" (target labels).

        Returns
        -------
        dict
            Dictionary containing the transformed "X" and "num_features".
        """
        X = inputs['X']
        y = inputs.get('y', None)
        X = self.forward(X, y)
        self.output("X", X)
