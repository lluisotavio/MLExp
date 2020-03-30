import numpy as np 
import os
from argparse import ArgumentParser 
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import tables 

# Script for reading multiple VTK files and write a HDF5 file. 
def name_index(input_string):
    
    strings = input_string.split("_")
    index = strings[-1]
    index = int(index) 
   
    return index

parser = ArgumentParser(description='Argument parsers')
parser.add_argument('--path', type=str)

args = parser.parse_args()

path = args.path

items = [path + ii for ii in os.listdir(path)]
data_directories = list(filter(os.path.isdir, items))

data_directories = sorted(data_directories, key=name_index)

variables = {3: 'k', 4: 'nut', 5: 'p', 8: 'rho', 1: 'T', 9: 'U'}
n_variables = len(variables.keys()) + 2

variables_dict = {var:list()  for var in variables.values()}

number_of_snapshots = len(data_directories)

# Reading the information about the number of cells 
directory = data_directories[1]
vtu_file = directory + "\\"+"internal.vtu"
print("File {} read".format(vtu_file))

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(vtu_file)
reader.Update()
data = reader.GetOutput()
n_cells = data.GetNumberOfCells()

n_cells = data.GetNumberOfCells()
cells_dict = dict() 

for cell_id in range(n_cells):

    cell = data.GetCell(cell_id)
    points = vtk_to_numpy(cell.GetPoints().GetData())
    cells_dict[cell_id] = points.mean(0)
    print("Cell {} read".format(cell_id))

hdf5_path = path + 'PitzDaily.hdf5'

# Creating the HDF5 file
h5f = tables.open_file(hdf5_path, mode='w')
#atom = tables.Atom.from_dtype(np.float64)
ds = h5f.create_array(h5f.root, 'AllData', np.empty((number_of_snapshots, n_cells, n_variables)))

for ss, directory in enumerate(data_directories[1:]): 

    vtu_file = directory + "\\"+"internal.vtu"
    print("File {} read".format(vtu_file))

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file)
    reader.Update()
    data = reader.GetOutput()

    print("File {} readed".format(vtu_file))

    variables_list = list() 

    for ii, name in variables.items():
        
        array = vtk_to_numpy(data.GetCellData().GetArray(ii))
        if len(array.shape) ==1: array = array[:,None]
        variables_list.append(array) 
        print("Variable {} readed".format(name))

    variables_array = np.hstack(variables_list)
    ds[ss, :, :] = variables_array 

h5f.close()  

print("Input arguments read")
