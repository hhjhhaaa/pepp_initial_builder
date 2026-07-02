from _cp2k_cli import load, parser
from pepp_initial_builder.lmp_proj_adapter import discover_lmp_proj_modules

args = parser().parse_args()
result = discover_lmp_proj_modules(load(args))
print(result["txt"])
