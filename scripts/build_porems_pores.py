from _pore_cli import parser, load, mode
from pepp_initial_builder.pore_workflow import build_porems_pores

args = parser().parse_args()
print(build_porems_pores(load(args), mode(args)))
