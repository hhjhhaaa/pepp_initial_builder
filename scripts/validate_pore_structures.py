from _pore_cli import parser, load
from pepp_initial_builder.pore_workflow import validate_pore_structures

args = parser().parse_args()
print(validate_pore_structures(load(args)))
