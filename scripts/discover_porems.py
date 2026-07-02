from _pore_cli import parser, load
from pepp_initial_builder.pore_workflow import write_porems_discovery

args = parser().parse_args()
print(write_porems_discovery(load(args)))
