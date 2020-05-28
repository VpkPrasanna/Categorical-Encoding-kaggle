import pandas as pd
import config

# path = config.dataset_path


def read_data(path):
    dataframe = pd.read_csv(path)
    return dataframe
