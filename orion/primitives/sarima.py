import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMA(object):
    """A Wrapper for the statsmodels.tsa.statespace.sarimax.SARIMAX class."""

    def __init__(self, p, d, q, s, steps):
        """Initialize the SARIMA object.
        Args:
            p (int):
                Integer denoting the order of the autoregressive model.
            d (int):
                Integer denoting the degree of differencing.
            q (int):
                Integer denoting the order of the moving-average model.
            s (int):
                Integer denoting the periodicity of data for the moving-average model.
            steps (int):
                Integer denoting the number of time steps to predict ahead.
        """
        self.p = p
        self.d = d
        self.q = q
        self.s = s
        self.steps = steps

    def predict(self, X):
        """Predict values using the initialized object.
        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.
        Returns:
            ndarray:
                N-dimensional array containing the predictions for each input sequence.
        """
        sarima_results = list()
        dimensions = len(X.shape)

        if dimensions > 2:
            raise ValueError("Only 1D or 2D arrays are supported")

        if dimensions == 1 or X.shape[1] == 1:
            X = np.expand_dims(X, axis=0)

        num_sequences = len(X)
        for sequence in range(num_sequences):
            sarima = SARIMAX(X[sequence],
                             order=(self.p, self.d, self.q),
                             seasonal_order=(self.p, self.d, self.q, self.s))
            sarima_fit = sarima.fit(disp=0)
            sarima_results.append(sarima_fit.forecast(self.steps)[0])

        sarima_results = np.asarray(sarima_results)

        if dimensions == 1:
            sarima_results = sarima_results[0]

        return sarima_results
