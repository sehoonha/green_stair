#!/usr/bin/env python

import pydart
import logging
from simulation import Simulation
from window import Window
import utils
import sys

# from logging_tree import printout  # pip install logging_tree
# printout()
root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.handlers = []

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
logfmt = '[%(levelname)s][%(asctime)s][%(module)s:%(lineno)d] %(message)s'
formatter = logging.Formatter(logfmt)
ch.setFormatter(formatter)
root.addHandler(ch)


# # Configure a logger
# logfmt = '[%(levelname)s][%(asctime)s][%(module)s:%(lineno)d] %(message)s'
# logging.basicConfig(level=logging.DEBUG,
#                     format=logfmt,
#                     datefmt='%m/%d/%Y %I:%M:%S %p')
# printout()
# logging.error('test1-2-3')
# exit(0)

# Get a logger for this file
logger = logging.getLogger(__name__)
logger.info('Green stair project')

# Register jsonpickle numpy handler
utils.jsonpickle_numpy.register_handlers()

step_activation = None
prefix = ''
postfix = ''
if len(sys.argv) > 1:
    index = 1
    try:
        step_activation = float(sys.argv[index])
        index += 1
    except ValueError:
        logger.info('The first argument is not float: %s' % sys.argv[index])

    for i, arg_i in enumerate(sys.argv[index:]):
        if i == 0:
            prefix = arg_i
        elif i == 1:
            postfix = arg_i

sim = Simulation(step_activation)
sim.prefix = prefix
sim.postfix = postfix
logger.info('prefix/postfix = %s/%s' % (sim.prefix, sim.postfix))
# Run the application
pydart.qtgui.run(title=sim.title(), simulation=sim, cls=Window)
