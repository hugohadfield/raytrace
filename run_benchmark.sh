python3 -m cProfile -o benchmark.prof -s cumtime clifftrace.py
python3 readable_benchmark.py > benchmark.txt