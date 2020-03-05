import pstats
from pstats import SortKey

p = pstats.Stats('benchmark.prof')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(400)
