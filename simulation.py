import logging
import numpy as np
import pydart
from fileinfoworld import FileInfoWorld
from mutable_motion import MutableMotion
from controller import Controller
import gltools
import posetools


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

        # Contruct the mutable motion
        self.motion = MutableMotion(self.skel, self.ref)
        x = self.motion.params()
        x += (np.random.rand() - 0.5) * 0.1
        self.motion.set_params(x)

        # Create the controller
        self.skel.controller = Controller(self.skel,
                                          self.world.dt,
                                          self.ref)

        # For check the target
        self.target_index = 0

        # Reset the scene
        self.reset()
        logger.info('set the initial pose OK')

    def reset(self):
        init_pose = self.motion.pose_at(0.0)
        # init_pose = self.ref.pose_at(0, skel_id=0)
        # init_pose[2] -= 0.25
        # init_pose[3] += 0.19
        # init_pose[4] += 0.07
        # I = self.skel.dof_indices(['j_heel_left_1', 'j_heel_right_1'])
        # init_pose[I] += 0.2

        self.skel.q = init_pose
        self.skel.qdot = np.zeros(self.skel.ndofs)
        self.world.reset()
        self.logger.info('reset OK')

    def step(self):
        # self.skel.controller.update_target_by_frame(self.world.frame)
        self.skel.controller.qhat = self.motion.pose_at(self.world.t)
        self.world.step()

    def num_frames(self):
        return self.world.num_frames()

    def set_frame(self, idx):
        self.world.set_frame(idx)

    def render(self):
        gltools.render_COM(self.skel)
        self.world.render()

    def contacts(self):
        return self.world.contacts()

    def key_pressed(self, key):
        # self.logger.info('key pressed: [%s]' % key)
        if key == ']':
            self.target_index = (self.target_index + 10) % self.ref.num_frames
            q = self.motion.pose_at(float(self.target_index) / 2000.0)
            # q = posetools.mirror_pose(self.skel, q)
            self.skel.q = q
        elif key == '[':
            self.target_index = (self.target_index - 10) % self.ref.num_frames
            self.skel.q = self.ref.pose_at(self.target_index, skel_id=0)
