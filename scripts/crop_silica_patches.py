from _pore_cli import parser, load, mode
from pepp_initial_builder.pore_workflow import crop_silica_patches

args = parser().parse_args()
print(crop_silica_patches(load(args), mode(args)))
