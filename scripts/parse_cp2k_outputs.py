from _cp2k_cli import load, parser
from pepp_initial_builder.cp2k_workflow import parse_cp2k_outputs

args = parser().parse_args()
print(parse_cp2k_outputs(load(args)))
