import numpy as np
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA


class STL_ARIMA(object):
    """A Wrapper for the statsmodels.tsa.statespace.sarimax.SARIMAX class."""

    def __init__(self, p, d, q, steps):
        """Initialize the SARIMA object.
        Args:
            p (int):
                Integer denoting the order of the autoregressive model.
            d (int):
                Integer denoting the degree of differencing.
            q (int):
                Integer denoting the order of the moving-average model.
            steps (int):
                Integer denoting the number of time steps to predict ahead.
        """
        self.p = p
        self.d = d
        self.q = q
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
        stl_arima_results = list()
        dimensions = len(X.shape)

        if dimensions > 2:
            raise ValueError("Only 1D or 2D arrays are supported")

        if dimensions == 1 or X.shape[1] == 1:
            X = np.expand_dims(X, axis=0)

        num_sequences = len(X)
        for sequence in range(num_sequences):
            stlf = STLForecast(X[sequence], ARIMA, model_kwargs=dict(order=(self.p, self.d, self.q)), period=3600)
            stlf_res = stlf.fit()
            stl_arima_results.append(stlf_res.forecast(self.steps)[0])

        stl_arima_results = np.asarray(stl_arima_results)

        if dimensions == 1:
            stl_arima_results = stl_arima_results[0]

        return stl_arima_results
