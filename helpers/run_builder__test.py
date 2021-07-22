from run_builder import RunBuilder
from params import params

run_builder = RunBuilder()
runs = run_builder.get_runs(params)
print(runs)
