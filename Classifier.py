from Data import Data
import pandas as pd
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import logging
import os
import random

logger = logging.getLogger(__name__)
project_path = os.getcwd()


def set_logger():
    if logger.hasHandlers():
        return

    log_file_path = project_path + '/Logs' + '/logg.txt'

    f_handler = logging.FileHandler(log_file_path, mode='a')
    f_format = logging.Formatter("%(asctime)s :: %(name)s\t :: %(levelname)s :: %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)


def logging_decor(func):
    """
    for every decorated function make a logging.info() of started,ended
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        self = args[0]
        logger.info(func.__name__ + " function has started")
        ret_val = func(*args, **kwargs)
        logger.info(func.__name__ + " function has ended")
        return ret_val

    return wrapper


class Classifier:

    def __init__(self, dataset, mode='oriented'):
        """
        creates a classifier for positions scores determination
        :param dataset: a Data object
        """
        set_logger()
        assert mode in {'FIFA', 'oriented'}
        self.mode = mode  # mode can be either 'FIFA' or 'oriented'
        self.dataset = dataset
        self.evaluations = []  # saves all evaluations calculated during cross validation
        self.all_pos = ['rw', 'rm', 'rb', 'rwb', 'st', 'lw', 'cf', 'cam', 'cm', 'lm',
                        'cdm', 'cb', 'lb', 'lwb', 'gk']  # all relevant positions
        self.all_pos_weights = {}  # k: position name, v: list of weights
        logger.info("new Classifier object has been created successfully")

    def build_list_from_indexes(self, train_index, test_index):
        """
        gets the return values from Data.Data.splitToFiveFolds method and converts them into lists
        :param train_index: 1st return values from Data.Data.splitToFiveFolds
        :param test_index: 2nd return values from Data.Data.splitToFiveFolds
        :return: two lists of indices, for train and test sets
        """
        train_list = []
        test_list = []
        for elem in train_index:
            train_list.append(elem)
        for elem in test_index:
            test_list.append(elem)
        return train_list, test_list

    def crossvalidation(self):
        """
        uses Data.Data.splitToFiveFolds to create five fold of the data
        each time uses one fold as test_set and all four others as train_set
        for every choose of test_set, set scores and evaluates the results
        at the end reports the mean of avg. deviation per position
        :return: no return value, only prints the classifier's results for five iteration
        """

        print("Cross validation (5 folds): ")
        iter_count = 1

        # for each fold out of the 5 folds as the test dataset:
        for train_index, test_index in self.dataset.splitToFiveFolds():

            print("Iteration number ", end='')
            print(iter_count)

            # create lists of indices for train & test datasets:
            train_list, test_list = self.build_list_from_indexes(train_index, test_index)

            # create dataframes of players for train & test datasets (by using the indices lists):
            train_set = Data(self.dataset.data.iloc[train_list])
            test_set = Data(self.dataset.data.iloc[test_list])
            ground_truth = self.dataset.positions.iloc[test_list]

            self.compute_all_weights(train_set)  # fit
            predicted_scores = self.set_scores(test_set)  # predict
            self.evaluations.append(Classifier.evaluate(predicted_scores, ground_truth))  # evaluate

            # for changing indices of players names:
            idx_dict = {}
            for old_index, new_index in zip(test_index, predicted_scores.index):
                idx_dict[old_index] = new_index

            print("Predictions:")
            print(pd.concat([test_set.data['Name'].rename(idx_dict), predicted_scores], axis=1))
            print("Ground Truth:")
            print(pd.concat([test_set.data['Name'], ground_truth], axis=1))
            print("Average deviation per position: ", end='')
            print(self.evaluations[-1])
            print("")
            print("**********************************************************")
            print("")

            iter_count += 1

        print("Mean of average deviation per position: ", end='')
        print(round(sum(self.evaluations) / len(self.evaluations), 2))

    @staticmethod
    def evaluate(predicted, ground_truth):
        """
        designed for mode 'FIFA'
        calculate the average deviation per position
        :param predicted: a dataframe contains predicted scores per position for all players on test set
        :param ground_truth: a dataframe contains FIFA2018 given scores per position for all players on test set
        :return: average deviation per position
        """
        avg_deviation_per_pos = 0
        iter_count = 0
        assert predicted.shape == ground_truth.shape
        for (index1, r1), (index2, r2) in zip(predicted.iterrows(), ground_truth.iterrows()):
            for col in predicted.columns:
                avg_deviation_per_pos += abs(predicted.at[index1, col] - ground_truth.at[index2, col])
                iter_count += 1
        return avg_deviation_per_pos / iter_count

    @staticmethod
    def all_in_bounds(solution):
        """
        :param solution: np.array with solution of a matrix, representing one option of weights vec per position
        :return: True if all components in vec are in bounds, False otherwise
        """
        for item in solution:
            if item < -0.1 or item > 1:
                return False
        return True

    @staticmethod
    def compute_weights_per_position(train_set, position, all_pos):
        """
        create a list of weights for each feature, per position, using the train set
        might raise an exception if the no solutions has been found during 10000 iterations
        :param train_set: a Data object representing the train set
        :param position: a string representing position to create features weights list
        :param all_pos: a list contains all relevant positions
        :return: features weights list for the position
        """

        assert (position in all_pos)

        # finding col index of irrelevant features:
        penalties_i = [i for i, x in enumerate(list(train_set.features.columns)) if x == 'penalties'][0]
        free_kick_accuracy_i = [i for i, x in enumerate(list(train_set.features.columns)) if x == 'free_kick_accuracy'][
            0]

        # if pos is gk, compute basic gk weights vec:
        if position == 'gk':
            train_set_features_gk = train_set.features.loc[train_set.preferences['prefers_gk'] == 1]
            cols = list(train_set_features_gk.columns)
            cols = [x for x in cols if (x[:3] == 'gk_')]
            train_set_features_gk = train_set_features_gk.filter(items=cols, axis=1)
            gk_means = list(train_set_features_gk.mean())

            norm = 0
            for item in gk_means:
                norm += item

            for i in range(len(gk_means)):
                gk_means[i] /= norm

            # cosmetic changes, to make sure all weights vectors are the same
            weights = [0] * 27
            weights += gk_means
            weights.insert(penalties_i, 0)
            weights.insert(free_kick_accuracy_i, 0)
            return weights

        # pos is not gk:
        train_set_features_no_gk = train_set.features.loc[train_set.preferences['prefers_gk'] != 1]
        cols = list(train_set_features_no_gk.columns)
        cols = [x for x in cols if (x[:3] != 'gk_' and x not in {'penalties', 'free_kick_accuracy'})]

        train_set_features_no_gk = train_set_features_no_gk.filter(items=cols, axis=1)
        train_set_position_no_gk = train_set.positions.loc[train_set.preferences['prefers_gk'] != 1, position]

        # create an eqv Ax = y
        # A - all non-gk player's features
        # x - weights vec per position (solution)
        # y - all non-gk player's scores in position
        results = pd.DataFrame()  # each row will be a solution
        num_sol = 0  # number of solutions found

        assert len(list(train_set_features_no_gk.columns)) == 27
        counter = 0  # num of tries to find a solution
        while num_sol < 15:
            if counter > 10000:  # max tries the program can tolerate
                if num_sol != 0:
                    logger.warning("weights vec for position: " + str(position) + \
                                   " has been created from only " + str(num_sol) + "solutions")
                    break
                else:
                    logger.error("weights vec for position: " + str(position) + " can not be created")
                    raise Exception("weights vec for position: " + str(position) + " can not be created")
            counter += 1

            rand_indxs = random.sample(list(train_set_features_no_gk.index), 27)
            matrix = train_set_features_no_gk.loc[rand_indxs].values  # A
            values = train_set_position_no_gk.loc[rand_indxs].values  # y

            if np.linalg.matrix_rank(matrix) != 27:
                continue

            try:
                solution = np.linalg.solve(matrix, values)
            except:
                logger.warning("solve func has crashed")
                continue

            if Classifier.all_in_bounds(solution):
                results = results.append(pd.Series(solution), ignore_index=True)
                num_sol += 1

        # compute one weights vec from mean of all solutions found
        weights_vec = results.mean(axis=0)
        weights = list(weights_vec)
        # cosmetic changes, to make sure all weights vectors are the same
        weights_gk = [0] * 5
        weights += weights_gk
        weights.insert(penalties_i, 0)
        weights.insert(free_kick_accuracy_i, 0)
        return weights

    @logging_decor
    def compute_all_weights(self, train_set):
        """
        create all weights lists and save them in a dictionary all_pos_weights (fit)
            *might raise an exception if there is one position that cannot be weighted
        compute correlation matrix
        calculate how many features should be considered in weights vec, by using stdv scale and parameters per position
        creating and fitting a rocchio classifier using train_set
        :param train_set: a Data object representing the train set
        :return: no return value. creates weights lists and stores them in all_pos.weights dictionary
        """
        # create basic weights vec
        for pos in self.all_pos:
            try:
                self.all_pos_weights[pos] = Classifier.compute_weights_per_position(train_set, pos, self.all_pos)
            except Exception as e:
                raise e

        # fitting NearestCentroid for future determination of tested player's natural position:
        self.rocchio_train_set = train_set.features
        self.rocchio_train_labels = []
        for index, player in train_set.positions.iterrows():
            self.rocchio_train_labels.append(player.idxmax())
        self.clf = NearestCentroid()
        self.clf.fit(self.rocchio_train_set, self.rocchio_train_labels)
        logger.info("Rocchio classifier created and fit")

    @staticmethod
    def create_linear_func(x1, x2, y1, y2):
        """
        creates ax+b func
        :param x1:
        :param x2:
        :param y1:
        :param y2:
        :return: (a,b)
        """
        a = (y1 - y2) / (x1 - x2)
        b = (a * x2 * (-1)) + y2
        return (a, b)

    @staticmethod
    def compute_position_score(player, weights):
        """
        computes a position score for a player using weighted arithmetic mean
        :param player: a series contains player's features
        :param weights: a basic weights list for a specific position
        :return:
        """
        position_score = 0
        for i in range(len(player)):
            position_score += player[i] * weights[i]
        return position_score

    @staticmethod
    def foot_factor(player, data, index, pos):
        """
        creates a factor based on position and foot data
        :param player: pd.Series of player features
        :param data: Data object of test_set
        :param index: player's index for data
        :param pos: str of position
        :return: a value between 0.92 to 1.02 for scaling player's score in position
        """
        is_left = data.foot.at[index, 'foot_Left']
        is_right = data.foot.at[index, 'foot_Right']
        weak_foot_score = int(data.weak_foot.at[index, 'weak_foot'])
        max_ratio = 1.02
        min_ratio = 0.92
        a, b = Classifier.create_linear_func(1, 6, min_ratio, max_ratio)
        dominant_foot_positions = {'rb', 'rwb', 'rm', 'lb', 'lwb', 'lm'}
        att_positions = {'rw', 'lw'}

        if pos in dominant_foot_positions:
            # this is his natural foot position
            if is_left and pos[0] == 'l' or is_right and pos[0] == 'r':
                return a * 6 + b
            else:
                return a * weak_foot_score + b

        elif pos in att_positions:
            if player.loc['crossing'] > player.loc['long_shots']:
                if is_left and pos[0] == 'l' or is_right and pos[0] == 'r':
                    return a * 6 + b
                else:
                    return a * weak_foot_score + b
            else:
                if is_left and pos[0] == 'l' or is_right and pos[0] == 'r':
                    return a * weak_foot_score + b
                else:
                    return a * 6 + b
        else:
            return 1.0

    @logging_decor
    def set_scores(self, test_set):
        """
        set scores for all players and positions (predict)
        just like the scores per position given by FIFA, we want to zero field-position-score
        for goalkeepers, or goalkeeping score for field-players
        :param test_set: a Data object representing the test set
        :return: positions scores predictions for player in test set, as a dataframe
        """
        predictions = pd.DataFrame(columns=list(self.all_pos))
        for index, player in test_set.features.iterrows():
            # predicting player's natural position by rocchio classifier:
            natural_pos = self.clf.predict([player])[0]
            player_pred_scores = []
            for pos in self.all_pos:
                # if pos. is field and natural is gk, or the opposite
                if (natural_pos == 'gk' and pos != 'gk') or (natural_pos != 'gk' and pos == 'gk'):
                    player_pred_scores.append(0)
                else:
                    # predicting score for player at pos:
                    score = (Classifier.compute_position_score(player, self.all_pos_weights[pos]))

                    if self.mode == 'oriented':
                        score *= self.foot_factor(player, test_set, index, pos)

                    player_pred_scores.append(int(round(score)))
            predictions = predictions.append(pd.Series(player_pred_scores, index=predictions.columns),
                                             ignore_index=True)
        return predictions

    @logging_decor
    def set_squad_scores(self, team_name):
        """
        gets a team name and prints the predictions, the ground truth and the avg. dev. per pos.
        :param team_name: a string representing a club's name
        :return: no return value
        """
        assert team_name in set(self.dataset.data.loc[:, 'club'])

        squad = self.dataset.data.loc[self.dataset.data['club'] == team_name]
        others = self.dataset.data.loc[self.dataset.data['club'] != team_name]
        test_set = Data(squad)
        train_set = Data(others)
        ground_truth = test_set.positions

        self.compute_all_weights(train_set)  # fit
        predictions = self.set_scores(test_set)  # predict
        eval = Classifier.evaluate(predictions, ground_truth)  # evaluate

        # with numerical indexes instead of player names
        ret_val = pd.DataFrame(predictions)

        # for changing indexes of players names:
        idx_dict = {}
        for old_index, new_index in zip(squad.index, predictions.index):
            idx_dict[old_index] = new_index

        Classifier.output(squad, ground_truth, predictions, team_name, eval)
        return ret_val

    @staticmethod
    def output(squad, ground_truth, predictions, club_name, eval):
        # for changing indexes of players names:
        idx_dict = {}
        for old_index, new_index in zip(squad.index, predictions.index):
            idx_dict[old_index] = new_index

        # write results:
        res_path = project_path + '/Results/' + str(club_name) + '_predictions.txt'
        with open(res_path, 'w', encoding='utf8') as r:
            r.write("Predictions: \n\n")
            r.write(str(pd.concat([squad['Name'].rename(idx_dict), predictions], axis=1).to_string(index=False)))
            r.write('\n')
            r.write("Ground Truth: \n\n")
            r.write(str(pd.concat([squad['Name'], ground_truth], axis=1).to_string(index=False)))
            r.write('\n\nAverage deviation per position: ')
            r.write(str(eval))
