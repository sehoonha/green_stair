import logging
import numpy as np
import pydart
from fileinfoworld import FileInfoWorld
from mutable_motion import MutableMotion
from motion_evaluator import MotionEvaluator
from controller import Controller
from motion_optimizer import MotionOptimizer
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
        for i, body in enumerate(self.skel.bodies):
            print i, body.name
        for i, dof in enumerate(self.skel.dofs):
            print i, dof.name

        # Configure stair: disable the movement of the first step
        self.world.skels[1].set_mobile(False)

        # Load the reference motion
        self.ref = FileInfoWorld()
        self.ref.load('data/other/halfCycle.txt')
        logger.info('load reference motions OK: # %d' % self.ref.num_frames)

        # Contruct the mutable motion
        self.motion = MutableMotion(self.skel, self.ref)
        # self.motion.save('test.json')
        # self.motion2 = MutableMotion(self.skel, self.ref)
        # self.motion2.load('test.json')
        # for t in np.arange(0.0, 0.1, 0.01):
        #     print t
        #     print self.motion.pose_at(t)
        #     print self.motion2.pose_at(t)

        # x = self.motion.params()
        # x += (np.random.rand() - 0.5) * 1.0
        # self.motion.set_params(x)

        # Construct the motion evaluator
        self.evaluator = MotionEvaluator(self.skel, self.motion)

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
        # self.skel.qdot = np.zeros(self.skel.ndofs)
        self.skel.qdot = self.motion.velocity_at_last()
        print 'Cdot:', self.skel.Cdot
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
        self.evaluator.render()

    def contacts(self):
        return self.world.contacts()

    def update_to_target(self):
        t = float(self.target_index) / 2000.0
        q = self.motion.pose_at(t)
        self.skel.q = q
        self.logger.info('time: %f\n%s' % (t, str(self.evaluator)))

    def key_pressed(self, key):
        # self.logger.info('key pressed: [%s]' % key)
        if key == ']':
            self.target_index = (self.target_index + 10) % self.ref.num_frames
            self.update_to_target()
        elif key == '[':
            self.target_index = (self.target_index - 10) % self.ref.num_frames
            self.update_to_target()
        elif key == 'O':
            solver = MotionOptimizer(self.motion, self.evaluator)
            solver.solve()
