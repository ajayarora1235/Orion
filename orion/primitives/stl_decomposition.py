import numpy as np
from statsmodels.tsa.seasonal import STL, DecomposeResult

class STL_DECOMPOSITION(object):
    """A Wrapper for STL decomposition and forecasting
    within the statsmodels.tsa.forecasting.stl class"""

    def __init__(self):
        """Initialize the STL decomposition object.
        """
        continue

    def fit(self, X, period):
        """Predict values using the initialized object.
        Args:
            X (ndarray):
                1-dimensional array containing the signal for the model.
            period (int):
                Integer that is the estimated period of the input sequence.
        Returns:
            model_endog (ndarray):
                1-dimensional array containing the signal for the model with
                the seasonal component removed.
        """

        stl = STL(X, period=period)
        stl_fit: DecomposeResult = stl.fit(inner_iter=None, outer_iter=None)
        model_endog = stl_fit.trend + stl_fit.resid
        seasonal_endog = stl_fit.seasonal
        return model_endog, seasonal_endog

    def predict(self, y_hat, seasonal_endog, index, window_size=250, period, steps=10):
        """Predict values using the initialized object.
        Args:
            y_hat (ndarray):
                N-dimensional array containing the forecast sequences from the model.
        Returns:
            ndarray:
                N-dimensional array containing the predictions for each input sequence.
        """

        seasonal_all = list()
        num_sequences = len(y_hat)
        for sequence in range(num_sequences):
            offset = index[sequence] + window_size
            if (offset < period):
                seasonal = seasonal_endog[offset: offset + period]
            else:
                seasonal = seasonal_endog[offset - period : offset]
            seasonal = np.tile(seasonal, steps // period + ((steps % period) != 0))
            seasonal = seasonal[:steps]
            seasonal_all.append(seasonal[0])

        seasonal_all = np.asarray(seasonal_all)
        if dimensions == 1:
            seasonal_all = seasonal_all[0]
        return np.add(y_hat, seasonal_all)
