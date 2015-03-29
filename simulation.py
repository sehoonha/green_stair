import logging
import numpy as np
import pydart
from fileinfoworld import FileInfoWorld
# from mutable_motion import MutableMotion
from controller import Controller
from motion_optimizer import MotionOptimizer
import gltools
from spring_stair import SpringStair
from motion import StepOffsetMotion


class Simulation(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logger = self.logger
        # Init pydart
        pydart.init()
        logger.info('pydart initialization OK')

        # Create world
        # skel_filename = 'data/skel/fullbody_baselineStairs2.skel'
        skel_filename = 'data/skel/fullbody_springStair.skel'
        self.world = pydart.create_world(1.0 / 1000.0, skel_filename)
        logger.info('pydart create_world OK: dt = %f' % self.world.dt)

        # Configure human
        self.skel = self.world.skels[0]
        for i, body in enumerate(self.skel.bodies):
            print i, body.name
        for i, dof in enumerate(self.skel.dofs):
            print i, dof.name

        # # Configure stair: disable the movement of the first step
        # self.stair = self.world.skels[1]
        # self.stair.set_mobile(False)
        self.stair = SpringStair(self.world)

        # Load the reference motion
        self.ref = FileInfoWorld()
        self.ref.load('data/other/halfCycle.txt')
        logger.info('load reference motions OK: # %d' % self.ref.num_frames)
        self.ref.append_mirrored_motion(self.skel)
        self.ref.append_shifted_motion(self.skel)
        logger.info('modify reference motions OK: # %d' % self.ref.num_frames)

        # Contruct the mutable motion
        self.motion = StepOffsetMotion(self.skel, self.ref)

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
        self.stair.reset()
        self.skel.q = self.motion.pose_at_frame(0, isRef=True)
        q = self.skel.q
        # q[3] -= 0.01
        q['j_thigh_left_z'] += 0.15
        q['j_shin_left'] -= 0.33
        q['j_heel_left_1'] += 0.15
        self.skel.q = q
        self.skel.qdot = 1.0 * self.motion.velocity_at_frame(0, isRef=True)

        self.world.reset()
        self.world.reset()
        self.logger.info('reset OK')

    def step(self):
        self.stair.apply_force()

        i = self.world.frame
        c = self.skel.controller
        c.qhat = self.motion.pose_at_frame(i, isRef=False)
        c.qdhat = self.motion.velocity_at_frame(i, isRef=True)

        self.world.step()

    def num_frames(self):
        return self.world.num_frames()

    def set_frame(self, idx):
        self.world.set_frame(idx)

    def render(self):
        gltools.render_COM(self.skel)
        self.world.render()
        self.render_target()
        # self.evaluator.render()
        # self.planner.render()

    def render_target(self):
        x = self.skel.x
        frame = min(self.world.frame, self.ref.num_frames - 1)
        qhat = self.ref.pose_at(frame, self.skel.id)
        self.skel.q = qhat
        self.skel.render_with_color(0.3, 0.3, 0.3, 0.5)
        self.skel.x = x

    def contacts(self):
        return self.world.contacts()

    def update_to_target(self):
        q = self.ref.pose_at(self.target_index, self.skel.id)
        self.skel.q = q

        # t = float(self.target_index) / 2000.0
        # # q = self.motion.pose_at(t)
        # q = self.planner.pose_at(self.target_index)
        # self.skel.q = q
        # self.logger.info('time: %f\n%s' % (t, str(self.evaluator)))

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
