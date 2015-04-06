#!/usr/bin/env python

import pydart
import logging
from simulation import Simulation
from window import Window
import utils
import sys

# Configure a logger
logfmt = '[%(levelname)s][%(asctime)s][%(module)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format=logfmt,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

# Get a logger for this file
logger = logging.getLogger(__name__)
logger.info('Green stair project')

# Register jsonpickle numpy handler
utils.jsonpickle_numpy.register_handlers()

step_activation = None
if len(sys.argv) > 1:
    step_activation = float(sys.argv[1])

sim = Simulation(step_activation)

# Run the application
pydart.qtgui.run(title='Green stair', simulation=sim, cls=Window)
