from _pore_cli import parser, load
from pepp_initial_builder.pore_workflow import validate_full_pore_seed_structures

args = parser().parse_args()
print(validate_full_pore_seed_structures(load(args)))
