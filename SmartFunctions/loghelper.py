#siple logger helper that takes a initial fileName and then returns a logger object that can be used to log messages
# to the file. The logger object is returned by the getLogger function. The logger object can be used to log messages
# to the file using the info, debug, warning, and error methods.

import logging
import os
import sys
import time
import datetime

# create logger 
def getLogger(fileName):
    # get current date and time
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H-%M")
    # create log file name
    
    logFileName = "./Logs/"+fileName + "_" + date + "-" + time + ".log"
    log_directory = os.path.dirname(logFileName)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    # create logger
    logger = logging.getLogger(fileName)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logFileName)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
