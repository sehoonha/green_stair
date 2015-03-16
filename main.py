#!/usr/bin/env python

import pydart
import logging
from simulation import Simulation
from window import Window
import utils

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

sim = Simulation()

# Run the application
pydart.qtgui.run(title='Green stair', simulation=sim, cls=Window)
