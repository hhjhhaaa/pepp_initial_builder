from _cp2k_cli import load, parser
from pepp_initial_builder.cp2k_workflow import export_aimd_dataset_manifest

args = parser().parse_args()
print(export_aimd_dataset_manifest(load(args)))
