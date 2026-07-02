from _cp2k_cli import load, mode, parser
from pepp_initial_builder.cp2k_workflow import write_cp2k_label_inputs

args = parser().parse_args()
print(write_cp2k_label_inputs(load(args), mode(args)))
