from os import listdir
from os.path import isfile, join
import numpy as np

import util.fileutil as fu
import util.datautil as du

import pdb

class driverdata(object):
    """Driver data
    read and manage driver data
    """

    def __init__(self, datapath, suffix=".dat"):
        """constructor
        create expert and inexpert data
        Args:
            datapath (str): data path.
                'expert' and 'inexpert' directories should be under the path.
                'expert' directory contains expert's driving data,
                'inexpert' directory contains inexpert driving data
            suffix (str): data file extension. default is ".dat"
        """
        self.input_data = {
            "expert": self._create_data(datapath + "/expert", suffix, 0),
            "inexpert": self._create_data(datapath + "/inexpert", suffix, 1)
        }

    def get_cross_validation_input(self, num_groups, group):
        """get_cross_validation_input
        Args:
            num_groups (int): seperate entire data to num_groups
            group (int): group number. Start from 0 to num)groups-1
        Return:
            dictionary with train and test data
        """
        train_data = []
        test_data = []
        for i in range(num_groups):
            for key in self.input_data:
                if(i != group):
                    train_data = np.append(train_data, self.input_data[key][i::num_groups])
                else:
                    test_data = np.append(test_data, self.input_data[key][i::num_groups])
        train_data = np.array(np.random.permutation(train_data))
        test_data = np.array(np.random.permutation(test_data))
        # return {"train":train_data, "test":test_data}
        return self.standardize({"train":train_data, "test":test_data})

    def standardize(self, data):
        train_data = data["train"]
        test_data = data["test"]

        merged_table = list()
        for d in train_data:
            merged_table += d["features"].tolist()
        mean, std = du.get_mean_std(np.array(merged_table))

        for d in train_data:
            d["features"] = du.standardize_data_by(d["features"], mean, std)
        for d in test_data:
            d["features"] = du.standardize_data_by(d["features"], mean, std)

        return {"train":train_data, "test":test_data}

    def _create_data(self, path, suffix, label):
        """_create_data
        Args:
            path: directory path
            suffix: data file suffix
            label: 0 => [0, 1], 1 => [1, 0]
        Return:
            list suffled data with label
            example_data_structure)
            [dictionary{features:data, label:label}, dictionary{features:data, label:label}, ...]
        """
        data_set = self._read_data(path, suffix)
        data_set = self._parse_data(data_set)
        data_set = self._add_label(data_set, label)
        data_set = np.random.permutation(data_set)
        return data_set

    def _read_data(self, path, suffix):
        """ _read_data
        Args:
            path: directory path
            suffix: data file suffix
        Return:
            data set
        """
        files = fu.getAllFileNames(path, suffix)
        return np.array(fu.readDataFiles(files, " "))

    def _parse_data(self, data_set):
        """_parse_data
        skip header and convert string value to float
        Args:
            data_set: three dimensional numpy array
        Return:
            modified data_set without header
        """
        result = []
        for data in data_set:
            # convert data to numpy array
            data = np.array(data)

            # replace empty string to 0
            data[data == ""] = 0

            # convert string to float
            result.append(data[1:,:].astype(np.float))
        return np.array(result)

    def _add_label(self, data_set, label):
        """ _add_label
        add label on data
        Args:
            data_set: three dimensional numpy array
            label: 0 => [0, 1], 1 => [1, 0]
        Return:
            data_set with label
        """
        result = []
        for data in data_set:
            data_dictionary = {
                "features": data,
                "label": [0,1] if label == 0 else [1,0]
            }
            result.append(data_dictionary)
        return np.array(result)

if __name__ == "__main__":
    data_path = "../data"
    data = driverdata(data_path) # data.input_data, data.get_cross_validation_input(num_groups, group)
    data = data.get_cross_validation_input(4, 1)
    
