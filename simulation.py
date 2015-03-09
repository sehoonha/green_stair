import logging
import numpy as np
import pydart
from fileinfoworld import FileInfoWorld
from controller import Controller


class Simulation(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logger = self.logger
        # Init pydart
        pydart.init()
        logger.info('pydart initialization OK')

        # Create world
        skel_filename = 'data/skel/fullbody_baselineStairs2.skel'
        self.world = pydart.create_world(1.0 / 2000.0, skel_filename)
        logger.info('pydart create_world OK: dt = %f' % self.world.dt)

        # Configure human
        self.skel = self.world.skels[0]

        # Configure stair: disable the movement of the first step
        self.world.skels[1].set_mobile(False)

        # Load the reference motion
        self.ref = FileInfoWorld()
        self.ref.load('data/other/halfCycle.txt')
        logger.info('load reference motions OK: # %d' % self.ref.num_frames)

        # Create the controller
        self.skel.controller = Controller(self.skel,
                                          self.world.dt,
                                          self.ref)

        # Reset the scene
        self.reset()
        logger.info('set the initial pose OK')

    def reset(self):
        self.skel.q = self.ref.pose_at(0, skel_id=0)
        self.skel.qdot = np.zeros(self.skel.ndofs)
        self.world.reset()
        self.logger.info('reset OK')

    def step(self):
        self.skel.controller.update_target_by_frame(self.world.frame)
        self.world.step()

    def num_frames(self):
        return self.world.num_frames()

    def set_frame(self, idx):
        self.world.set_frame(idx)

    def render(self):
        self.world.render()

    def key_pressed(self, key):
        self.logger.info('key pressed: [%s]' % key)
