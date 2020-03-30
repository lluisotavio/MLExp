import numpy as np 
import os
from argparse import ArgumentParser
import tables 
import glob
import pyvista

# Script for reading multiple VTK files and write a HDF5 file. 
def name_index(input_string):
    
    strings = input_string.split("_")
    index = strings[-1]
    index = int(index) 
   
    return index

def extract_data_from_name(vti_file):

    sub_names = vti_file.split("\\")
    file_name = sub_names[-1]
    file_indices = file_name.split('_')[-1]
    file_indices = file_indices[:-4]
    coordinates = file_indices.split('i')[1:]
    iteration = int(coordinates[0][1:])
    partition = int(coordinates[1][1:])

    return iteration, partition


parser = ArgumentParser(description='Argument parsers')
parser.add_argument('--path', type=str)
parser.add_argument('--case', type=str)

args = parser.parse_args()

path = args.path
case = args.case

data_directories = glob.glob(path+"\\"+case+"*.vti")

variables = ['physPressure', 'physVelocity', 'physTemperature']
n_variables = len(variables) + 1

number_of_snapshots = len(data_directories)
number_of_partitions = 7
default_value = 99999

# Reading the information about the number of points
hdf5_path = path + 'rayleighBernard.hdf5'

# Creating the HDF5 file
h5f = tables.open_file(hdf5_path, mode='w')
#ds = h5f.create_array(h5f.root, 'AllData', np.empty((number_of_snapshots, n_points, n_variables)))

solution_dict = {partition: list() for partition in range(number_of_partitions)}
points_dict = dict()

for ss, vti_file in enumerate(data_directories):

    data = pyvista.read(vti_file)
    points = data.points
    n_points = points.shape[0]
    data_dimensions = data.GetDimensions()[:-1]

    # Recovering important information from the file name
    iteration, partition = extract_data_from_name(vti_file)

    points_dict[partition] = points.reshape(data_dimensions[::-1]+(points.shape[-1],))

    print("File {} read".format(vti_file))

    data = pyvista.read(vti_file)

    print("File {} read".format(vti_file))

    variables_list = list() 

    variables_dict = data.point_arrays

    spacing = data.GetSpacing()

    for name in variables:
        
        array = variables_dict[name]

        if len(array.shape) == 1:
            array = array[:, None]

        array = array.reshape(data_dimensions[::-1] + (array.shape[-1],))

        variables_list.append(array)
        print("Variable {} read".format(name))

    variables_array = np.dstack(variables_list)
    solution_dict[partition].append(variables_array)

    #ds[ss, :, :] = variables_array

x_list = list()
y_list = list()

for partition in range(number_of_partitions):

    list_of_arrays = solution_dict[partition]
    solution_dict[partition] = np.stack(list_of_arrays, 0)

    points_array = points_dict[partition]

    number_of_snapshots = solution_dict[partition].shape[0]

    x_max = points_array[:, :, 0].max()
    x_min = points_array[:, :, 0].min()

    y_max = points_array[:, :, 1].max()
    y_min = points_array[:, :, 1].min()

    x_list.append(x_max)
    x_list.append(x_min)

    y_list.append(y_max)
    y_list.append(y_min)

    y_list.append(y_max)

x_max = np.array(x_list).max()
x_min = np.array(x_list).min()

y_max = np.array(y_list).max()
y_min = np.array(y_list).min()

x_dim = int((x_max - x_min)/spacing[0]) + 1
y_dim = int((y_max - y_min)/spacing[1]) + 1

# TODO Now we need to reconstruct the original domain
# This allocation method should be replaced by a more convenient one
# such as HDF5 allocation in disk
global_x_array = np.linspace(x_min, x_max, x_dim)
global_y_array = np.linspace(y_min, y_max, y_dim)
global_z_array = np.zeros(1)

global_points_array = np.meshgrid(global_x_array, global_y_array, global_z_array)
global_points_array = np.dstack(global_points_array)

global_indices_hor = ((global_x_array - x_min)/spacing[0]).astype(int)
global_indices_vert = ((global_y_array - y_min)/spacing[1]).astype(int)
global_indices_deep = np.zeros(1).astype(int)

global_indices_array = np.meshgrid(global_indices_hor, global_indices_vert, global_indices_deep)
global_indices_array = np.dstack(global_indices_array)

global_solution_array = default_value*np.ones((number_of_snapshots, y_dim, x_dim, n_variables))

for partition in range(number_of_partitions):

    points_array = points_dict[partition]
    solution_array = solution_dict[partition]

    x_local_dim, y_local_dim = points_array.shape[:-1]

    x_local_max = points_array[:, :, 0].max()
    x_local_min = points_array[:, :, 0].min()
    y_local_max = points_array[:, :, 1].max()
    y_local_min = points_array[:, :, 1].min()

    x_array = np.linspace(x_local_min, x_local_max, x_local_dim)
    y_array = np.linspace(y_local_min, y_local_max, y_local_dim)
    z_array = np.zeros(1)

    indices_hor = ((x_array - x_min) / spacing[0]).astype(int)
    indices_vert = ((y_array - y_min) / spacing[1]).astype(int)
    indices_deep = np.zeros(1).astype(int)

    indices_array = np.meshgrid(indices_hor, indices_vert, indices_deep)
    indices_array = np.dstack(indices_array)

    j_max = indices_array[:, :, 0].max()
    j_min = indices_array[:, :, 0].min()

    i_max = indices_array[:, :, 1].max()
    i_min = indices_array[:, :, 1].min()

    global_solution_array[:, i_min:i_max+1, j_min:j_max+1, :] = solution_array

    print("Connecting the multiple partitions.")


# The Numpy format is used for small data with testing purposes
# For big data we consider to dump the data directly to HDF5 format
# still during the conversion
#global_solution_array = global_solution_array.transpose(0, 3, 1, 2)
np.save(path + "rayleighBernard.npy", global_solution_array)

h5f.close()

print("Input arguments read")
