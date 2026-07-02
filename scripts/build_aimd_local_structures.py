from _pore_cli import parser, load, mode
from pepp_initial_builder.pore_workflow import build_aimd_local_structures

args = parser().parse_args()
print(build_aimd_local_structures(load(args), mode(args)))
