import numpy as np

def one_hot_array(arrays_list):

    one_hot_arrays_list = list()
    for array in arrays_list:

        one_hot = array.flatten()
        one_hot_arrays_list.append(one_hot)

    one_hot_array = np.hstack(one_hot_arrays_list)

    return one_hot_array


