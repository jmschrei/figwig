# __init__.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

__version__ = '0.2.0'

from .bigwig import BigWigReader
from .bigwig import read_bigwig
from .writer import BigWigWriter
from .writer import write_bigwig
from .figwig import FigWigReader
from .figwig import is_figwig
from .figwig import read_figwig
from .figwig import FigWigWriter
from .figwig import write_figwig
from .figwig import SparseWindows
from .figwig import SparseEntries

__all__ = ['BigWigReader', 'read_bigwig', 'BigWigWriter', 'write_bigwig',
	'FigWigReader', 'is_figwig', 'read_figwig', 'FigWigWriter', 'write_figwig',
	'SparseWindows', 'SparseEntries']
