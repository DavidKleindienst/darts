"""
NLL Gaussian Scorer
-------------------

Gaussian negative log-likelihood Scorer.

The anomaly score is the negative log likelihood of the actual series values
under a Gaussian distribution estimated from the stochastic predictions.
"""

import numpy as np
from scipy.stats import norm

from darts.ad.scorers.scorers import NLLScorer


class GaussianNLLScorer(NLLScorer):
    def __init__(self, window: int = 1) -> None:
        """NLL Gaussian Scorer

        Parameters
        ----------
        window
            Integer value indicating the size of the window W used by the scorer to transform the series into an
            anomaly score. A scorer will slice the given series into subsequences of size W and returns a value
            indicating how anomalous these subset of W values are. A post-processing step will convert this anomaly
            score into a point-wise anomaly score (see definition of `window_transform`). The window size should be
            commensurate to the expected durations of the anomalies one is looking for.
        """
        super().__init__(window=window)

    def __str__(self):
        return "GaussianNLLScorer"

    def _score_core_nllikelihood(
        self, vals: np.ndarray, pred_vals: np.ndarray
    ) -> np.ndarray:
        mu = np.mean(pred_vals, axis=1)
        std = np.std(pred_vals, axis=1)
        return -norm.logpdf(vals, loc=mu, scale=std)
