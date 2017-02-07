import numpy as np

def get_mean_std(datatable):
    """compute mean and std
    Args:
        datatable (2D array)
    Returns:
        mean, std, num_data
    """
    mean = np.mean(datatable, axis=0)
    std = np.std(datatable, axis=0)
    num_data = datatable.shape[0]
    return mean, std

def standardize_data_by(datatable, mean, std):
    return np.nan_to_num((datatable - mean) / std)

def standardize_data(datatable):
    mean, std = get_mean_std(datatable)
    return standardize_data_by(datatable, mean, std)

if __name__ == "__main__":
    test_data = [[1, 2, 3], [9, 6, 3]]
    test_table = np.array(test_data)
    mean, std = get_mean_std(test_table)
    print(mean, std)
    print(standardize_data_by(test_table, mean, std))
    print(standardize_data(test_table))

