import pandas as pd
import unicodedata
import re
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


def change_latin(club):
    """
    for example: changes 'ć' to 'c'
    :param club: string of club name
    :return:
    """
    return ''.join(char for char in
                   unicodedata.normalize('NFKD', club)
                   if unicodedata.category(char) != 'Mn')


def remove_numbers(club):
    return re.sub('\d+', '', club)


def replace_hyphen(club):
    return re.sub('-', ' ', club)


def remove_fin_ini(club):
    """
    for example: 'FC Barcelona' to 'Barcelona'
    :param club:
    :return:
    """
    result = re.sub('[A-Z][A-Z] ', '', club)
    result = re.sub('[A-Z][A-Z][A-Z] ', '', result)
    result = re.sub(' [A-Z][A-Z]', '', result)
    result = re.sub(' [A-Z][A-Z][A-Z]', '', result)
    return result


def make_dct():
    data = pd.read_csv("Full_DF.csv")
    data = data.loc[:, 'club']
    data = data.drop_duplicates()
    data = data.to_list()
    res = {k: set() for k in data}

    for item in data:
        res[item].add(item)
        club = str(item)
        club = remove_fin_ini(club)
        res[item].add(club)
        club = change_latin(club)
        res[item].add(club)
        club = club.lower()
        res[item].add(club)
        club = remove_numbers(club)
        res[item].add(club)
        club = replace_hyphen(club)
        res[item].add(club)

    res['Paris Saint-Germain'].add('PSG')
    res['Juventus'].add('juve')
    res['FC Barcelona'].add('barca')
    res['Manchester City'].add('man city')
    res['Manchester United'].add('man united')
    res['Borussia Dortmund'].add('dortmund')
    return res


def get_real(name):
    """
    :param name: name to check
    :return: if found a variation of name in clubs than return the real club name
             otherwise Exception is raised
    """
    set_logger()
    clubs = make_dct()
    name_tmp = name.lower()
    name_tmp = remove_numbers(name)
    for club, options in clubs.items():
        if name in options:
            logger.info('found that ' + str(name) + ' is actually ' + str(club))
            return club
    logger.error('there is no such team ' + str(name))
    raise KeyError('there is no such team')
