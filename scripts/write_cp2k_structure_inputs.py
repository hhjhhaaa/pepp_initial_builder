from _pore_cli import parser, load
from pepp_initial_builder.pore_workflow import write_cp2k_structure_inputs

args = parser().parse_args()
print(write_cp2k_structure_inputs(load(args)))
