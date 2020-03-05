import pstats
from pstats import SortKey
import sys
import clifford
import numpy as np
import numba


print('Python version: ' , sys.version)
print('Clifford version: ', clifford.__version__)
print('Numpy version: ', np.__version__)
print('Numba version: ', numba.__version__)


p = pstats.Stats('benchmark.prof')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(400)
