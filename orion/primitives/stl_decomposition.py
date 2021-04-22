import numpy as np
from statsmodels.tsa.seasonal import STL, DecomposeResult


class STL_DECOMPOSITION(object):
    """A Wrapper for STL decomposition and forecasting
    within the statsmodels.tsa.forecasting.stl class"""

    def __init__(self):
        """Initialize the STL decomposition object.
        """
        continue

    def get_period(self, X):
        """Predict values using the initialized object.
        Args:
            X (ndarray):
                1-dimensional array containing the input signal for the model.
        Returns:
            int:
                Estimated period of signal, based on fourier transform of signal to find peaks
        """

        # taken from https://stackoverflow.com/questions/49531952/find-period-of-a-signal-out-of-the-fft
        X = L
        L -= np.mean(L)
        fft = np.fft.rfft(L, norm="ortho")

        def abs2(x):
            return x.real**2 + x.imag**2

        selfconvol=np.fft.irfft(abs2(fft), norm="ortho")
        selfconvol=selfconvol/selfconvol[0]

        # let's get a max, assuming a least 4 periods...
        multipleofperiod=np.argmax(selfconvol[1:len(L)/4])
        Ltrunk=L[0:(len(L)//multipleofperiod)*multipleofperiod]

        fft = np.fft.rfft(Ltrunk, norm="ortho")
        selfconvol=np.fft.irfft(abs2(fft), norm="ortho")
        selfconvol=selfconvol/selfconvol[0]

        #get ranges for first min, second max
        fmax=np.max(selfconvol[1:len(Ltrunk)/4])
        fmin=np.min(selfconvol[1:len(Ltrunk)/4])
        xstartmin=1
        while selfconvol[xstartmin]>fmin+0.2*(fmax-fmin) and xstartmin< len(Ltrunk)//4:
            xstartmin=xstartmin+1

        xstartmax=xstartmin
        while selfconvol[xstartmax]<fmin+0.7*(fmax-fmin) and xstartmax< len(Ltrunk)//4:
            xstartmax=xstartmax+1

        xstartmin=xstartmax
        while selfconvol[xstartmin]>fmin+0.2*(fmax-fmin) and xstartmin< len(Ltrunk)//4:
            xstartmin=xstartmin+1

        period=np.argmax(selfconvol[xstartmax:xstartmin])+xstartmax

        return period

    def decomposition(self, X):
        """Predict values using the initialized object.
        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.
        Returns:
            ndarray:
                N-dimensional array containing the input sequences for the model with
                the seasonal component removed.
        """
        stl_decomp_results = list()
        seasonal_results = list()
        periods = list()

        num_sequences = len(X)
        for sequence in range(num_sequences):
            period = get_period(X[sequence])
            stl = STL(X[sequence], period=period)
            stl_fit: DecomposeResult = stl.fit(inner_iter=None, outer_iter=None)
            model_endog = stl_fit.trend + stl_fit.resid
            seasonal_endog = stl_fit.seasonal
            stl_decomp_results.append(model_endog)
            seasonal_results.append(seasonal_endog)
            periods.append(period)

        stl_decomp_results = np.asarray(stl_decomp_results)
        seasonal_results = np.asarray(seasonal_results)

        if len(stl_decomp_results) == 1:
            stl_decomp_results = stl_decomp_results[0]

        if len(seasonal_results) == 1:
            seasonal_results = seasonal_results[0]

        return stl_decomp_results, seasonal_results, periods

    def seasonal_forecast(self, X, current_forecast, seasonal_component, periods, steps):
        """Predict values using the initialized object.
        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.
        Returns:
            ndarray:
                N-dimensional array containing the predictions for each input sequence.
        """

        seasonal_all = list()
        num_sequences = len(X)
        seasonal_shaped = np.reshape(seasonal_component, (-1, 250))
        for sequence in range(num_sequences):
            period = periods[num_sequences]
            seasonal = np.asarray(seasonal_shaped[num_sequences])
            offset = X[sequence].shape[0]
            seasonal = seasonal[offset - period : offset]
            seasonal = np.tile(seasonal, steps // period + ((steps % period) != 0))
            seasonal = seasonal[:steps]
            seasonal_all.append(seasonal[0])
        return current_forecast + seasonal_all
