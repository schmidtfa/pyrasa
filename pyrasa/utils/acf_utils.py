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
    """Return the normalized ACF implied by a band-limited, one-sided PSD.

    The PSD must contain nonnegative power at uniformly spaced frequencies
    in Hz. Its last dimension is frequency; leading dimensions are preserved.
    The transformation uses a trapezoidal cosine sum evaluated via an IFFT.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Uniformly spaced, nonnegative frequencies in Hz. Cropped bands are OK.
    powers : ndarray, shape (..., n_freqs)
        Nonnegative, one-sided PSD. The last axis must match ``freqs``.
    max_lag : float, default 0.2
        Largest requested lag, in seconds. Must be <= 1 / (2 * df).
    lag_step : float, default 0.001
        Desired *maximum* lag spacing in seconds; the actual spacing can be
        slightly smaller because an efficient integer FFT length is used.

    Returns
    -------
    lags : ndarray, shape (n_lags,)
        Delays in seconds, starting at zero.
    corrs : ndarray, shape (..., n_lags)
        Normalized, band-limited autocorrelation with ``corrs[..., 0] == 1``.

    Notes
    -----
    Missing frequencies outside the provided band are *not* extrapolated.
    The result is not necessarily the full-signal autocorrelation.
    """
    freqs = np.asarray(freqs, dtype=float)
    powers = np.asarray(powers, dtype=float)

    steps = np.diff(freqs)
    df = float(steps[0])

    if not (np.isfinite(max_lag) and max_lag >= 0):
        raise ValueError('max_lag must be finite and positive.')
    if not (np.isfinite(lag_step) and lag_step > 0):
        raise ValueError('lag_step must be finite and positive.')

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
    """Channel-wise, band-limited autocorrelation derived from a PSD."""

    lags: np.ndarray
    corrs: np.ndarray
    ch_names: np.ndarray | None = None
    source: str = 'aperiodic'

    def half_life(self) -> np.ndarray:
        """First crossing of 0.5, in seconds; NaN if no crossing exists."""
        half_life = 0.5
        result = np.full(self.corrs.shape[0], np.nan, dtype=float)
        for ch, corr in enumerate(self.corrs):
            crossings = np.flatnonzero((corr[:-1] >= half_life) & (corr[1:] < half_life))
            if crossings.size:
                i = int(crossings[0])
                result[ch] = self.lags[i] + (
                    (half_life - corr[i]) * (self.lags[i + 1] - self.lags[i]) / (corr[i + 1] - corr[i])
                )
        return result

    def at_lag(self, delay: float) -> np.ndarray:
        """Interpolated ACF at a delay in seconds, one value per channel."""
        if not np.isfinite(delay) or not self.lags[0] <= delay <= self.lags[-1]:
            raise ValueError('delay must be within the computed lag interval.')
        return np.array([np.interp(delay, self.lags, corr) for corr in self.corrs])
