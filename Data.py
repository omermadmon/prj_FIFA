import pandas as pd
import sklearn.model_selection
import sklearn
import logging
import os

logger = logging.getLogger(__name__)


def set_logger():
    if logger.hasHandlers():
        return
    project_path = os.getcwd()
    log_file_path = project_path + '/Logs' + '/logg.txt'

    f_handler = logging.FileHandler(log_file_path, mode='a')
    f_format = logging.Formatter("%(asctime)s :: %(name)s\t :: %(levelname)s :: %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)


class Data:

    @staticmethod
    def read_csv(path):
        """
        creates a dataframe from a csv, and than instances a Data object and returns it
        might raise FIleNotFoundError
        :param path: a path to csv file
        :return: a Data object
        """
        try:
            set_logger()
            return Data(pd.read_csv(path))
        except FileNotFoundError as e:
            logger.error(e)
            raise e

    def __init__(self, data):
        """
        creates a Data object from a dataframe
        :param data: a dataframe contains all data from FIFA18 dataset
        """
        set_logger()
        self.data = data  # load full dataset to dataframe
        self.relevant_positions = {'gk', 'cb', 'rb', 'lb', 'rwb', 'lwb', 'cdm', 'cm', 'cam',
                                   'rm', 'lm', 'rw', 'lw', 'st', 'cf'}  # all relevant positions
        self.features = self.data.iloc[:, 38:72]  # player's features on field
        self.preferences = self.data.iloc[:, 163:190]  # player positions preferences
        self.foot = self.data.iloc[:, 190:192]  # is right or left footed
        self.weak_foot = self.data.iloc[:, 37:38]  # weak foot rate
        self.positions = self.data.iloc[:, 72:99]  # ground truth for setting score per position
        self.preprocess()  # delete irrelevant positions
        logger.info("Data object has created successfully")

    def preprocess(self):
        """
        deletes irrelevant positions from preferences and positions dataframes
        :return: no return value
        """
        col_to_drop = []
        for col in self.positions:
            if col not in self.relevant_positions:
                col_to_drop.append(col)
        self.positions = self.positions.drop(col_to_drop, axis=1)

        col_to_drop.clear()
        for col in self.preferences:
            tmp = col.split("_")[1]
            if tmp not in self.relevant_positions:
                col_to_drop.append(col)
        self.preferences = self.preferences.drop(col_to_drop, axis=1)

    def splitToFiveFolds(self):
        """
        split the data into five folds for cross validation
        :return: five ways to split the data to train and test (by indices)
        """
        kfolds = sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=1)
        return kfolds.split(self.data)
