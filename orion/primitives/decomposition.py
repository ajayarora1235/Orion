import numpy as np
from statsmodels.tsa.seasonal import STL, DecomposeResult

class Decomposition:
    def decompose(self, X, period):
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

        stl_fit = STL(X, period=period).fit(inner_iter=None, outer_iter=None)
        return np.reshape(stl_fit.trend, (-1, 1)), np.reshape(stl_fit.resid, (-1, 1)), np.reshape(stl_fit.seasonal, (-1, 1))

    def compose(self, y_res, y_trend, y_seasonal):
        """Predict values using the initialized object.
        Args:
            y_hat (ndarray):
                N-dimensional array containing the forecast sequences from the model.
        Returns:
            ndarray:
                N-dimensional array containing the predictions for each input sequence.
        """

        return np.add(np.add(y_res, y_trend), y_seasonal)
