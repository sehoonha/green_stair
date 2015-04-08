import logging
import numpy as np
import pydart
from fileinfoworld import FileInfoWorld
from controller import Controller
import gltools
from spring_stair import SpringStair
from motion import StepOffsetMotion, RadialBasisMotion, Optimizer
from plotter_torque import PlotterTorque
# from guppy import hpy


class Simulation(object):
    def __init__(self, step_activation=None):
        self.logger = logging.getLogger(__name__)
        logger = self.logger
        # Init pydart
        pydart.init()
        logger.info('pydart initialization OK')

        # Create world
        if step_activation is None:
            skel_filename = 'data/skel/fullbody_baselineStairs2.skel'
        else:
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
        logger.info('set step activation: %s' % step_activation)
        self.stair.set_activation(step_activation)

        # Load the reference motion
        self.ref = FileInfoWorld()
        self.ref.load('data/other/halfCycle.txt')
        logger.info('load reference motions OK: # %d' % self.ref.num_frames)
        self.ref.append_mirrored_motion(self.skel)
        self.ref.append_shifted_motion(self.skel)
        self.ref.add_offset()
        logger.info('modify reference motions OK: # %d' % self.ref.num_frames)

        # Contruct the mutable motion
        # self.motion = StepOffsetMotion(self.skel, self.ref, self.stair)
        self.motion = RadialBasisMotion(self.skel, self.ref, self.stair)

        # Create the controller
        self.skel.controller = Controller(self.skel,
                                          self.world.dt,
                                          self.ref)

        # For check the target
        self.target_index = 0

        # Reset the scene
        self.reset_counter = 0
        self.reset()
        logger.info('set the initial pose OK')

    def reset(self):
        self.skel.controller.reset()
        self.stair.reset()
        self.world.reset()
        q = self.motion.pose_at_frame(0, isRef=True)
        q = pydart.SkelVector(q, self.skel)
        q['j_heel_right_1'] += 0.05
        self.skel.q = q
        self.skel.qdot = self.motion.velocity_at_frame(0, isRef=True)
        # if self.reset_counter % 50 == 0:
        #     h = hpy()
        #     print h.heap()
        # self.reset_counter += 1

    def step(self):
        self.stair.apply_force()

        i = max(self.world.frame, -200)
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
        frame = min(self.world.frame, self.ref.num_frames - 1)
        self.render_target_at_frame(frame)

    def render_target_at_frame(self, frame):
        x = self.skel.x
        qhat = self.ref.pose_at(frame, self.skel.id)
        self.skel.q = qhat
        self.skel.render_with_color(0.3, 0.3, 0.3, 0.5)
        self.skel.x = x

    def contacts(self):
        return self.world.contacts()

    def update_to_target(self):
        q = self.ref.pose_at(self.target_index, self.skel.id)
        # q = self.motion.pose_at_frame(self.target_index, self.skel.id)
        self.skel.q = q

    def key_pressed(self, key):
        # self.logger.info('key pressed: [%s]' % key)
        if key == ']':
            print self.target_index
            self.target_index = (self.target_index + 10) % self.ref.num_frames
            self.update_to_target()
        elif key == '[':
            print self.target_index
            self.target_index = (self.target_index - 10) % self.ref.num_frames
            self.update_to_target()

    def optimize(self):
        self.solver = Optimizer(self, self.motion)
        self.solver.launch()

    def kill_optimizer(self):
        self.solver.to_be_killed = True

    def plot_torques(self):
        pl = PlotterTorque()
        if self.stair.num_steps() == 0:
            title = 'Normal Stair'
        else:
            title = 'Stair(%.3f)' % self.stair._activation
        pl.plot(self.skel.controller, title)
