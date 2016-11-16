from os import listdir
from os.path import isfile, join
import numpy as np

import util.fileutil as fu

class driverdata(object):
    """Driver data
    read and manage driver data
    """

    def __init__(self, datapath, suffix=".dat"):
        """constructor
        create expert and inexpert data
        Args:
            datapath: data path.
                'expert' and 'inexpert' directories should be under the path.
                'expert' directory contains expert's driving data,
                'inexpert' directory contains inexpert driving data
            suffix: data file extension. default is ".dat"
        """
        self.expert_data = self.createData(datapath + "/expert", suffix)
        self.inexpert_data = self.createData(datapath + "/inexpert", suffix)

    def createData(self, path, suffix):
        """ createData
        Args:
            path: directory path
            suffix: data file suffix
        Returns:
            void
        """
        files = fu.getAllFileNames(path, suffix)
        return self.modifyData(np.array(fu.readDataFiles(files, " ")))

    def modifyData(self, data_set):
        """modifyData
        skip header and convert string value to float
        Args:
            data_set: three dimensional numpy array
        Returns:
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
        return result

