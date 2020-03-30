import sys
sys.path.insert(0, '.')

from argparse import ArgumentParser
import glob
import imageio
import platform

from MLExp.io.input_reader import VTKReader

def separator(system_str):

    if system_str == "Windows":
        return "\\"
    if system_str == "Linux":
        return "/"

parser = ArgumentParser(description='Argument parsers')
parser.add_argument('--path', type=str)
parser.add_argument('--case', type=str)

args = parser.parse_args()

path = args.path
case = args.case

sep = separator(platform.system())

data_directories = glob.glob(path + sep + case + "*.vti")

vtk_reader = VTKReader(data_directories=data_directories)
vtk_reader.exec()

vtk_reader.save(save_path=path, file_name=case)

solution = vtk_reader.global_solution_array

for vv in range(vtk_reader.number_of_variables):

    imageio.mimwrite(path + sep + case + "_{}_".format(vv) + ".mp4", solution[:, :, :, vv], fps=int(solution.shape[0]/5))

print("Conversion completed.")


