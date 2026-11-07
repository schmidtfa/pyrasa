"""Convert a one-sided PSD to its band-limited, normalized autocorrelation."""

import numpy as np
from attrs import define
from scipy.fft import ifft, next_fast_len


def psd_to_acf(
    freqs: np.ndarray,
    powers: np.ndarray,
    max_lag: float = 0.2,
    lag_step: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a normalized, band-limited autocorrelation from a one-sided PSD.

    The PSD must contain finite, nonnegative power values at uniformly
    spaced frequencies in Hz. The last dimension represents frequency,
    while all leading dimensions are preserved.

    The autocorrelation is computed as a discrete cosine sum using a
    zero-padded inverse FFT and normalized to unity at zero lag.

    Parameters
    ----------
    freqs : np.ndarray, shape (n_freqs,)
        Strictly increasing, uniformly spaced, nonnegative frequencies
        in Hz. At least two frequency values are required. The frequency
        range may represent a cropped band.
    powers : np.ndarray, shape (..., n_freqs)
        Finite, nonnegative, one-sided PSD values. The last dimension
        must correspond to `freqs`. Each spectrum must contain at least
        one positive power value.
    max_lag : float, default 0.2
        Maximum requested autocorrelation lag, in seconds.
        Must be finite and nonnegative. For the intended lag range,
        this should not exceed 1 / (2 * df), where df is the
        frequency-bin spacing.
    lag_step : float, default 0.001
        Maximum desired spacing between consecutive lags, in seconds.
        The actual spacing is 1 / (n_fft * df) and may be smaller
        because the FFT length is adjusted.

    Returns
    -------
    lags : np.ndarray, shape (n_lags,)
        Autocorrelation lags in seconds, starting at zero and extending
        to the largest grid point not exceeding `max_lag`.
    corrs : np.ndarray, shape (..., n_lags)
        Normalized, band-limited autocorrelation values, with
        `corrs[..., 0] == 1` for valid input spectra.

    Notes
    -----
    The inverse FFT evaluates a discrete cosine sum over the supplied
    frequency bins. All bins are weighted by their corresponding PSD
    values.

    Spectral amplitudes are scaled internally for numerical stability
    before the autocorrelation is normalized at zero lag.

    Frequencies outside the provided band are not extrapolated.
    Consequently, the result represents the autocorrelation associated
    with the supplied spectral band, not necessarily the autocorrelation
    of the full signal.
    """
    freqs = np.asarray(freqs, dtype=float)
    powers = np.asarray(powers, dtype=float)

    steps = np.diff(freqs)
    df = float(steps[0])

    if not (np.isfinite(max_lag) and max_lag >= 0):
        raise ValueError('max_lag must be finite and positive.')
    if not (np.isfinite(lag_step) and lag_step > 0):
        raise ValueError('lag_step must be finite and positive.')

    if max_lag > 1 / (2 * df):
        raise ValueError('max_lag must not exceed 1 / (2 * df), where df is the frequency-bin spacing.')

    # FFT needs a time grid fine enough to represent the highest PSD frequency.
    n_fft = next_fast_len(
        max(
            freqs.size,
            int(np.ceil(1.0 / (lag_step * df))),
            int(np.ceil(2.0 * freqs[-1] / df)),
        )
    )

    # Amplitude normalization avoids overflow / underflow for tiny EEG/MEG PSDs.
    amplitude = powers.max(axis=-1, keepdims=True)
    if np.any(amplitude <= 0):
        raise ValueError('PSD must have finite, positive power.')

    # # Trapezoidal cosine quadrature, including the actual frequency origin.
    weighted = powers / amplitude

    count = int(np.floor(max_lag * n_fft * df + 1e-10)) + 1
    lags = np.arange(count) / (n_fft * df)
    transformed = ifft(weighted, n=n_fft, axis=-1)[..., :count]
    corrs = (transformed * np.exp(2j * np.pi * freqs[0] * lags)).real
    corrs /= corrs[..., :1]

    return lags, corrs


@define
class IrasaACF:
    """
    Container for normalized, band-limited autocorrelation estimates.

    Stores autocorrelation values derived from a power spectrum,
    together with their corresponding lags and optional channel
    and time information.

    Autocorrelation curves are stored along the last dimension of
    `corrs`. Leading dimensions may represent channels, time points,
    or other independent observations.

    Parameters
    ----------
    lags : np.ndarray, shape (n_lags,)
        Autocorrelation lags in seconds, expected to be monotonically
        increasing.
    corrs : np.ndarray, shape (..., n_lags)
        Autocorrelation values corresponding to `lags`, expected to
        be normalized to unity at zero lag.
    ch_names : np.ndarray | None, optional
        Channel names associated with the autocorrelation estimates.
    source : str, default 'aperiodic'
        Label identifying the spectral component from which the
        autocorrelation was derived.
    time : np.ndarray | None, optional
        Time points associated with time-resolved autocorrelation
        estimates, in seconds.

    """

    lags: np.ndarray
    corrs: np.ndarray
    ch_names: np.ndarray | None = None
    source: str = 'aperiodic'
    time: np.ndarray | None = None

    def half_life(self) -> np.ndarray:
        """
        Estimate the half-life of each autocorrelation curve.

        Determine the first lag at which the autocorrelation crosses
        from above 0.5 to 0.5 or below. The crossing time is estimated
        by linear interpolation between the two surrounding lag values.

        Returns
        -------
        np.ndarray, shape corrs.shape[:-1]
            First half-life of each autocorrelation curve, in seconds.
            Returns NaN for curves that do not cross the threshold
            within the available lag range.

        Notes
        -----
        The half-life is defined as the first downward crossing of
        the normalized autocorrelation at 0.5. No exponential decay
        model is fitted.
        """
        half = 0.5
        half_lives = np.full(self.corrs.shape[:-1], np.nan)

        for i, acf in enumerate(self.corrs.reshape(-1, self.corrs.shape[-1])):
            crossings = np.flatnonzero((acf[:-1] > half) & (acf[1:] <= half))

            if crossings.size:
                j = crossings[0]
                half_lives.flat[i] = np.interp(
                    half,
                    acf[j : j + 2][::-1],
                    self.lags[j : j + 2][::-1],
                )

        return half_lives

    def at_lag(self, delay: float) -> np.ndarray:
        """
        Evaluate the autocorrelation at a specified lag.

        Calculate the autocorrelation at the requested delay using
        linear interpolation between the available lag values.

        Parameters
        ----------
        delay : float
            Lag at which to evaluate the autocorrelation, in seconds.
            Must lie within the range of `self.lags`.

        Returns
        -------
        np.ndarray, shape corrs.shape[:-1]
            Interpolated autocorrelation value for each curve.

        Raises
        ------
        ValueError
            If the requested delay lies outside the available lag range.
        """
        if not self.lags[0] <= delay <= self.lags[-1]:
            raise ValueError('delay outside the available lag range.')

        return np.array(
            [np.interp(delay, self.lags, acf) for acf in self.corrs.reshape(-1, self.corrs.shape[-1])]
        ).reshape(self.corrs.shape[:-1])
