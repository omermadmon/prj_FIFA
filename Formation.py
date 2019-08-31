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


class Formation:
    # for example 4-3-3 or 4-4-2

    def __init__(self, name, positions):
        set_logger()
        # data members
        self._name = name
        self._positions = positions
        # check GK
        if not self.has_gk():
            logger.error("GK_ERROR has been raised")
            raise Formation.GK_ERROR("There's no GK in your formation")
        logger.info("formation " + self._name + " has been set")

    def has_gk(self):
        # a formation is legete only if there's a Goal-Keeper in it
        return "gk" in self._positions

    class GK_ERROR(Exception):
        # Exception class
        def __init__(self, message):
            self._message = message
            raise
