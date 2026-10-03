# __init__.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

__version__ = '0.1.0'

from .bigwig import BigWig
from .bigwig import read_bigwig
from .writer import BigWigWriter
from .writer import write_bigwig

__all__ = ['BigWig', 'read_bigwig', 'BigWigWriter', 'write_bigwig']
