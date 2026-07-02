from _pore_cli import parser, load
from pepp_initial_builder.pore_workflow import validate_aimd_local_structures

args = parser().parse_args()
print(validate_aimd_local_structures(load(args)))
