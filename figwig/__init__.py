# __init__.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

__version__ = '0.1.0'

from .bigwig import BigWig
from .bigwig import read_windows

__all__ = ['BigWig', 'read_windows']
