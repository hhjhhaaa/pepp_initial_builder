from _pore_cli import parser, load
from pepp_initial_builder.pore_workflow import export_pore_aimd_manifests

args = parser().parse_args()
print(export_pore_aimd_manifests(load(args)))
