import Formation
import logging
import pandas as pd
import numpy as np
import os
import errno

logger = logging.getLogger(__name__)
project_path = os.getcwd()
log_path = project_path + '/Logs'


def set_logger():
    if logger.hasHandlers():
        return
    log_file_path = log_path + '/logg.txt'

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
        logger.info("in formation: " + self._formation._name + ": " + func.__name__ + " function has started")
        ret_val = func(*args, **kwargs)
        logger.info("in formation: " + self._formation._name + ": " + func.__name__ + " function has ended")
        return ret_val

    return wrapper


class Tactic:
    # static variable
    NUM_POSITIONS = 11

    def __init__(self, formation, data, mode='standard', num_orders=3):
        set_logger()
        # __init__ data members:
        self._formation = formation
        self._players = data
        self._mode = mode
        self._positions_quan = {k: self._formation._positions.count(k) for k in self._formation._positions}
        self._positions = {k: {} for k in self._formation._positions}
        self._orders = []
        if num_orders is 0:
            logger.warning("num_orders is 0")
            self._num_orders = 1
        else:
            self._num_orders = num_orders

        # inside logger - used for logging all new scores submitted
        self._scores_logger = logging.getLogger("Formation " + self._formation._name)
        if self._scores_logger.hasHandlers():
            return
        f_handler_scores = logging.FileHandler(log_path + "/logg_scores.txt", mode='a')
        f_format = logging.Formatter("%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s")
        f_handler_scores.setFormatter(f_format)
        self._scores_logger.addHandler(f_handler_scores)
        self._scores_logger.setLevel(logging.DEBUG)

        # other functions data members:
        #   (*) self._players_re        # only positions scores and name. useful
        #   (*) self._positions_order   # for rec
        #   (*) self._positions_re      # for rec

    def __str__(self):
        res = "Tactic " + self._formation._name + '\n'
        if len(self._orders) is 0:
            res += "No orders yet. You should run set() function"
        else:
            res += "Best Orders are: \n"
            for i, order_and_score in enumerate(self._orders):
                res += "Order #" + str(i + 1) + ":\n"
                # print order by format (example:):
                # " Order #1:
                #   GK: Ter Stegen 85
                #   Order score is: 78.6 "
                for order, score in order_and_score.items():
                    for position, player in order:
                        player_name = player[0]
                        player_score = player[1]
                        res += str(position[:-1].upper() + ": " + str(player_name) + " " + str(player_score))
                        res += '\n'
                res += ("\nOrder score is: " + str(score) + '\n\n')
        return res

    @logging_decor
    def set(self, p=3, t=3):
        """
        main function of Tactic.
        Runs the algorithm by calling other functions.
        :param p: stands for min number of players per position
        :param t: stands for number of positions per player
        :return: no return value. changes self._orders
        """

        # Preprocess
        self.filter_positions_by_formation()

        # First attitude: player based
        # self.set_players_to_positions(t)
        # self.fill_blanks(p)
        # Second attitude: position based (not in use now):
        self.set_players_to_positions_by_positions(5)

        # TODO: if p>5
        # building order with backtracking
        self.filter_positions_by_relevance(p)

        self.run_all_options()

    def output(self, file_path, file_mode):
        """
        Writes the output of best order into given file_path (param)
        :param file_path: path of wanted output file
        :param file_mode: mode to open file with
        :return: no return value
        """
        # 'utf8' in order to being able to write characters like 'é', 'ć'
        try:
            with open(file=file_path, mode=file_mode, encoding='utf8') as r:
                logger.info("file " + file_path + " has opened")
                res = str(self.__str__())
                # print (res)
                r.write(res)
                r.write("\n---------------------------------------------------\n\n")
                logger.info("result was written")
        except OSError as e:
            logger.error(str(__name__) + ": reading file failed")

    def filter_positions_by_formation(self):
        """
        filters the data about the players to the formation relevant positions (and Name) only
        :return: no return value. changes self._players
        """
        tmp = list(set((self._formation._positions)))
        tmp.append('Name')
        self._players_re = self._players.filter(items=tmp)

    @logging_decor
    def set_players_to_positions_by_positions(self, p):
        """
        Second Attitude - Position Based
        assigns p (param) best players per position to the self._positions dictionary
        :param p: number of players per unique position.
        :return: no return value. changes
        """
        for position in self._positions:
            players_dct = {}
            for i, player in self._players_re.iterrows():
                players_dct[player.loc['Name']] = player.loc[position]

            players_dct = sorted(players_dct.items(), key=lambda kv: kv[1], reverse=True)
            actual_p = p * self._positions_quan[position]
            players_dct = players_dct[:actual_p]
            self._positions[position] = dict(players_dct)

    def find_best_players_in_position(self, position, p=3):
        """
        used by self.filter_positions_by_relevance(p)
        finds the best p (param) players in the given position (param)
        :param position: <string>
        :param p: number of players who should be best in position
        :return: <dict> of p players represented by player_name:score_in_position
        """
        position_dct = self._positions[position]
        # print (position, position_dct)
        player_dct = sorted(position_dct.items(), key=lambda kv: kv[1], reverse=True)
        actual_p = p * self._positions_quan[position]

        return dict(player_dct[:actual_p])

    @logging_decor
    def filter_positions_by_relevance(self, p):
        for position in self._positions:
            self._positions[position] = self.find_best_players_in_position(position, p=p)

    @logging_decor
    def run_all_options(self):
        """
        wrapped function for the recursion one (self.rec_run())
        :return: no return value
        """
        # for position,player in self._positions.items():
        #    print (position, player)
        self.run_preprocess()
        self.rec_run({}, [], self._positions_order[0])

    @staticmethod
    def find_min(lst):
        """
        finds minimum from dicts values in a list
        :param lst: a list of dicts [{k:v}, {k:v},...]
        :return: minimum of values in tuple: (min_index, min_value)
        """
        assert len(lst) is not 0
        # default values
        min_val = 100 * Tactic.NUM_POSITIONS  # biggest value possible
        min_index = -1
        for i, dct in enumerate(lst):
            cur_val = list(dct.values())[0]
            if cur_val < min_val:
                min_val = cur_val
                min_index = i
        return min_index, min_val

    @staticmethod
    def dct_to_tuple(dct):
        res = ()
        for k, v in dct.items():
            for inner_k, inner_v in v.items():
                tmp = (k, (inner_k, inner_v))
                res = res + (tmp,)
        return res

    def append_new_order(self, order, order_score):
        """
        used by check_and_set_order_alt
        sorts the mult_pos, than checks if this order is already exists. if not than appends it
        :param order: order to check
        :param order_score: order's(param) score
        :return: no return value. may change self._orders
        """

        # positions with more than a single instance. for ex: 'cb'
        mult_pos = [(k, v) for k, v in self._positions_quan.items() if v > 1]

        for tup in mult_pos:
            position = tup[0]
            quantity = tup[1]
            # read all names that are in the same position in the given order
            players = {}
            for i in range(quantity):
                player_name = list(order[position + str(i)].keys())[0]
                player_score = list(order[position + str(i)].values())[0]
                players[player_name] = player_score
            # sort it
            players = sorted(players.items(), key=lambda kv: kv[0], )
            # put them again in order, now sorted
            for i, player in enumerate(players):
                player_name = player[0]
                player_score = player[1]
                order[position + str(i)] = {player_name: player_score}
        # check if same as one that had already been submitted
        # using tuple (immutable) conversion because cur_order is a dict(mutable), which can't be a key
        tmp = Tactic.dct_to_tuple(order)
        if self.exists_same_order(tmp):
            return

        # submit new order
        dct = {tmp: order_score}  # {order:score}
        self._orders.append(dct)
        self._scores_logger.info("New order has been set with score: {0:.3f}".format(order_score))

    def exists_same_order(self, other_order):
        """
        checks if an order like this has already been submitted
        :param other_order: order to check
        :return: True if exists one, False otherwise
        """
        for cur_order in self._orders:
            tmp = list(cur_order.keys())[0]
            # print (tmp)
            # print (other_order)

            if tmp == other_order:
                # print ("True\n")
                return True
        # print("False\n")
        return False

    def check_and_set_order(self, cur_order):
        """
        used by self.rec_run_new(order, name_exist, position)
        Checks if given order (param) is not exists and if it's within best self._num_orders
        :param cur_order: new order that has been found
        :return: no return value. may change self._orders
        """
        cur_score = Tactic.calc_score(cur_order)
        if len(self._orders) is not self._num_orders:
            self.append_new_order(cur_order, cur_score)
        else:
            min_index, min_score = Tactic.find_min(self._orders)
            if min_score < cur_score:
                del self._orders[min_index]
                self.append_new_order(cur_order, cur_score)

    def rec_run(self, order, name_exist, position):
        """
        Goes over every option for order, based on the filtering done before.
        Done by backtracking (recursion)
        :param order: current order filled
        :param name_exist: current names in order. player can't be in 2 different positions
        :param position: current position to fill
        :return: no return value. changes self._order
        """

        # stop condition
        if len(order) is Tactic.NUM_POSITIONS:
            # if it's within _self.num_orders than add it. otherwise don't add it.
            self.check_and_set_order(order)
            return

        # for every player in this position that is not in order yet, call the rec_run again, with him in order now.
        # create copies of order and name_exist in order
        tmp_order = dict(order)
        for name, score in self._positions_re[position].items():
            if name not in name_exist:
                tmp_order[position] = {name: score}
                tmp_name_exist = list(name_exist)
                tmp_name_exist.append(name)
                # if in last position, next position is None. but in order to get to stop-condition,
                # next_position got to have some value, which will be not used anyway
                try:
                    next_position = self.get_next_position(position)
                except IndexError as e:
                    next_position = 'not important'
                self.rec_run(order=tmp_order, name_exist=tmp_name_exist, position=next_position)

    def run_preprocess(self):
        """
        Creates self._positions_order and self._positions_re for easier use in recursion.
        :return:
        """
        # both below used only in run_all_option_alt.
        self._positions_order = []  # to be able to determine "next" position in recursion.
        self._positions_re = {}  # 11 positions. for example: 'cb0','cb1'.
        for position in self._positions:
            for count in range(self._positions_quan[position]):
                self._positions_order.append(position + str(count))
                self._positions_re[(position + str(count))] = self._positions[position]

    def get_next_position(self, position):
        """
        :param position: current position
        :return: next position in order (based on self._position_order)
        """
        index = [i for i, x in enumerate(self._positions_order) if x == position][0]
        if index is not 10:
            res = self._positions_order[index + 1]
        else:
            res = 'end'
            raise IndexError()
        return res

    @staticmethod
    def calc_score(order):
        """
        Formula to measure the goodness of a given order (param).
        Formula is average of all score_in_position values
        :param order: dict of 11 players by {position:{Name:score_in_position}}
        :return: score of the formula
        """
        res = 0
        # print (list(order.values()))
        for player in list(order.values()):
            player_score = (list(player.values())[0])
            res += player_score
        return res / Tactic.NUM_POSITIONS
