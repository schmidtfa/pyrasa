"""Interface to use the IRASA algorithm with MNE objects."""

from .irasa_mne import irasa_epochs, irasa_raw

__all__ = ['irasa_epochs', 'irasa_raw']
