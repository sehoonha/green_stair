import pydart
import logging
from simulation import Simulation
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
tb = pydart.qtgui.Trackball(phi=-1.4, theta=-12.0, zoom=1.0,
                            rot=[-0.10, 0.09, -0.00, 0.99],
                            trans=[-0.06, 0.21, -3.01])
pydart.qtgui.run(title='Green stair', simulation=sim, trackball=tb)
