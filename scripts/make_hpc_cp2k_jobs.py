from _cp2k_cli import load, mode, parser
from pepp_initial_builder.cp2k_workflow import make_hpc_cp2k_jobs

args = parser().parse_args()
print(make_hpc_cp2k_jobs(load(args), mode(args)))
