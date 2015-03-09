import pydart
import logging
from fileinfoworld import FileInfoWorld
import numpy as np


logfmt = '[%(levelname)s][%(asctime)s][%(module)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format=logfmt,
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.info('Green stair project')

pydart.init()
logger.info('pydart initialization OK')
world = pydart.create_world(1.0 / 2000.0,
                            'data/skel/fullbody_baselineStairs2.skel')
logger.info('pydart create_world OK: dt = %f' % world.dt)

refmotion = FileInfoWorld()
refmotion.load('data/other/halfCycle.txt')

human = world.skels[0]
human.q = refmotion.pose[0][0]


def step_callback(world):
    pass


def keyboard_callback(world, key):
    """ Programmable interactions """
    pass

# Run the application
pydart.glutgui.run(title='Green stair', simulation=world, trans=[0, 0, -3.0],
                   step_callback=step_callback,
                   keyboard_callback=keyboard_callback)
