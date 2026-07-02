from _cp2k_cli import load, parser
from pepp_initial_builder.cp2k_workflow import validate_aimd_dataset

args = parser().parse_args()
print(validate_aimd_dataset(load(args)))
