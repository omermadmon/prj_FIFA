from Data import Data
from Classifier import Classifier
import Club_Names
from Formation import Formation
from Tactic import Tactic
import pandas as pd
import logging
import os
import errno
import sys

logger = logging.getLogger(__name__)
project_path = os.getcwd()


def read_team(club_name, positions_scores):
    data_path = 'Full_DF.csv'
    data = pd.read_csv(data_path)
    squad = data.loc[data['club'] == club_name]
    data_adm = squad.loc[:, ['Name', 'club', 'Player_ID', 'club_id']]
    data_positions_scores = positions_scores
    data_adm = data_adm.reset_index(drop=True)
    squad = data_adm.join(data_positions_scores)
    return squad


def create_tactic(formation, club_squad, club_name, file_mode, k):
    tactic = Tactic(formation, club_squad, num_orders=k)
    tactic.set()
    res_path = project_path + '/Results'
    tactic.output(file_path=res_path + '/' + str(club_name) + "_tactics.txt", file_mode=file_mode)


def set_dir_and_logger():
    log_path = project_path + '/Logs'
    res_path = project_path + '/Results'
    try:
        os.mkdir(log_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    log_file_path = project_path + '/Logs' + '/logg.txt'

    try:
        os.mkdir(res_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # create new empty file named logg.txt
    with open(file=log_file_path, mode='w'):
        pass
    # create new empty file named logg_scores.txt
    with open(log_path + "/logg_scores.txt", 'w'):
        pass

    # create and set new logger for the project
    f_handler = logging.FileHandler(log_file_path, mode='a')
    f_format = logging.Formatter("%(asctime)s :: %(name)s\t :: %(levelname)s :: %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)


if __name__ == '__main__':
    
    # set logger and check if club name (or cv) has been passed:
    set_dir_and_logger()
    if len(sys.argv) == 1:
        logger.error("no arguments had been passed (club name or cv should be passes as an argument)")
        raise Exception("no arguments had been passed (club name or cv should be passes as an argument)")
    else:
        logger.info("GETTING STARTED!")

    # club name has been given as an argument:
    if sys.argv[1] != 'cv':

        print("loading dataset . . .")

        # create Data and Classifier objects:
        dataset = Data.read_csv("Full_DF.csv")
        classifier = Classifier(dataset, mode='oriented')

        print("dataset has been loaded successfully.")

        # predict score for a club:
        club_name = ''
        for i in range(1, len(sys.argv)):
            club_name = club_name + ' ' + str(sys.argv[i])
        club_name = club_name[1:]

        try:
            club_name = Club_Names.get_real(club_name)
        except KeyError as e:
            print(e)

        print("predicting FIFA scores for " + club_name + " players . . .")

        positions_scores = classifier.set_squad_scores(club_name)

        print("scores predictions for " + club_name + " players have been written to Results", end='')
        print("\\" + club_name + "_predictions.txt")

        # read data for tactics setting:
        club_squad = read_team(club_name, positions_scores)

        # create basic formations list:
        f_433 = Formation("4-3-3", ['gk', 'lb', 'cb', 'cb', 'rb', 'cdm', 'cm', 'cm', 'lw', 'rw', 'st'])
        f_352 = Formation("3-5-2", ['gk', 'cb', 'cb', 'cb', 'rwb', 'lwb', 'cm', 'cm', 'cam', 'st', 'st'])
        f_442_diamond = Formation("4-4-2-Diamond", ['gk', 'rb', 'cb', 'cb', 'lb', 'cdm', 'cm', 'cm', 'cam', 'st', 'st'])
        f_442 = Formation("4-4-2", ['gk', 'rb', 'cb', 'cb', 'lb', 'cdm', 'cdm', 'lm', 'rm', 'cf', 'st'])
        basic_formations = [f_442_diamond, f_442, f_433, f_352]

        print("creating tactics for " + club_name + " squad . . .")

        # set 3 best tactics for each formation:
        for i, formation in enumerate(basic_formations):
            mode = 'a'
            if i is 0:
                mode = 'w'  # create a new empty file
            create_tactic(formation, club_squad, club_name, mode, 3)

        print("tactics for " + club_name + " squad have been written to Results", end='')
        print("\\" + club_name + "_tactics.txt")

    # cv has been given as an argument:
    else:
        print("loading dataset . . .")

        # create Data and Classifier objects:
        dataset = Data.read_csv("Full_DF.csv")
        classifier = Classifier(dataset, mode='FIFA')

        print("dataset has been loaded successfully.")

        # cross validation:
        classifier.crossvalidation()
