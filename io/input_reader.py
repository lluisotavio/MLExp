import numpy as np
import pyvista
import itertools
import platform

class VTKReader:

    def __init__(self, data_directories=None):

        self.sep = self._separator(platform.system())

        self.data_directories = data_directories
        self.points_dict = dict()
        self.variables = self._get_variables(vti_file=self.data_directories[0])

        self.number_of_variables = len(self.variables) + 1

        self.number_of_partitions, self.number_of_snapshots, self.iterations_dict =\
            self._discover_number_of_partitions_and_iterations(data_directories=self.data_directories)

        self.solution_dict = {iteration: {partition: None for partition in range(self.number_of_partitions)}
                                            for iteration in self.iterations_dict.keys() }

        self.default_value = 99999


        self.global_solution_array = None

    def _separator(self, system_str):

        if system_str == "Windows":
            return "\\"
        if system_str == "Linux":
            return "/"

    # Script for reading multiple VTK files and write a HDF5 file.
    def _name_index(self, input_string):

        strings = input_string.split("_")
        index = strings[-1]
        index = int(index)

        return index

    def _get_variables(self, vti_file=None):

        data = pyvista.read(vti_file)
        return data.point_arrays.keys()

    def _discover_number_of_partitions_and_iterations(self, data_directories=None):

        partitions_list = list()
        iterations_list = list()

        for ss, vti_file in enumerate(data_directories):

            iteration, partition = self._extract_data_from_name(vti_file)

            partitions_list.append(partition)
            iterations_list.append(iteration)

        number_of_iterations = len(set(iterations_list))
        number_of_partitions = max(partitions_list) + 1

        iterations_list = sorted(set(iterations_list))

        iterations_dict = {value: key for key, value in enumerate(iterations_list)}

        return number_of_partitions, number_of_iterations, iterations_dict

    def _extract_data_from_name(self, vti_file):

        sub_names = vti_file.split(self.sep)
        file_name = sub_names[-1]
        file_indices = file_name.split('_')[-1]
        file_indices = file_indices[:-4]
        coordinates = file_indices.split('i')[1:]
        iteration = int(coordinates[0][1:])
        partition = int(coordinates[1][1:])

        return iteration, partition

    def exec(self):

        for ss, vti_file in enumerate(self.data_directories):

            data = pyvista.read(vti_file)
            points = data.points
            data_dimensions = data.GetDimensions()[:-1]

            # Recovering important information from the file name
            iteration, partition = self._extract_data_from_name(vti_file)

            self.points_dict[partition] = points.reshape(data_dimensions[::-1] + (points.shape[-1],))

            print("File {} read".format(vti_file))

            data = pyvista.read(vti_file)

            print("File {} read".format(vti_file))

            variables_list = list()

            variables_dict = data.point_arrays

            spacing = data.GetSpacing()

            for name in self.variables:

                array = variables_dict[name]

                if len(array.shape) == 1:
                    array = array[:, None]

                array = array.reshape(data_dimensions[::-1] + (array.shape[-1],))

                variables_list.append(array)
                print("Variable {} read".format(name))

            variables_array = np.dstack(variables_list)
            self.solution_dict[iteration][partition] = variables_array

        x_list = list()
        y_list = list()

        for iteration, partition in itertools.product(self.iterations_dict.keys(), range(self.number_of_partitions)):

            points_array = self.points_dict[partition]

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

        x_dim = int((x_max - x_min) / spacing[0]) + 1
        y_dim = int((y_max - y_min) / spacing[1]) + 1

        global_solution_array = self.default_value * np.ones((self.number_of_snapshots, y_dim, x_dim, self.number_of_variables))

        print("Connecting the multiple partitions.")

        for iteration, partition in itertools.product(self.iterations_dict.keys(), range(self.number_of_partitions)):

            points_array = self.points_dict[partition]
            solution_array = self.solution_dict[iteration][partition]

            x_local_dim, y_local_dim = points_array.shape[:-1]

            x_local_max = points_array[:, :, 0].max()
            x_local_min = points_array[:, :, 0].min()
            y_local_max = points_array[:, :, 1].max()
            y_local_min = points_array[:, :, 1].min()

            x_array = np.linspace(x_local_min, x_local_max, x_local_dim)
            y_array = np.linspace(y_local_min, y_local_max, y_local_dim)

            indices_hor = ((x_array - x_min) / spacing[0]).astype(int)
            indices_vert = ((y_array - y_min) / spacing[1]).astype(int)
            indices_deep = np.zeros(1).astype(int)

            indices_array = np.meshgrid(indices_hor, indices_vert, indices_deep)
            indices_array = np.dstack(indices_array)

            j_max = indices_array[:, :, 0].max()
            j_min = indices_array[:, :, 0].min()

            i_max = indices_array[:, :, 1].max()
            i_min = indices_array[:, :, 1].min()

            it_index = self.iterations_dict[iteration]

            global_solution_array[it_index, i_min:i_max + 1, j_min:j_max + 1, :] = solution_array


        self.global_solution_array = global_solution_array[:,::-1, :, :]


    def save(self, save_path=None, file_name=None):

        np.save(save_path + file_name + ".npy", self.global_solution_array)


