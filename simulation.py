import logging
import numpy as np
import pydart
from fileinfoworld import FileInfoWorld
from controller import Controller
import gltools
from spring_stair import SpringStair
from motion import *
from hyper import *
from plotter_torque import PlotterTorque
# from guppy import hpy


class Simulation(object):
    def __init__(self, step_activation=None):
        self.prefix = ''
        self.postfix = ''
        self.logger = logging.getLogger(__name__)
        logger = self.logger
        # Init pydart
        pydart.init()
        logger.info('pydart initialization OK')

        # Create world
        # if step_activation is None:
        #     skel_filename = 'data/skel/fullbody_baselineStairs2.skel'
        # else:
        step_activation = 0.0
        skel_filename = 'data/skel/fullbody_springStair.skel'
        # skel_filename = 'data/skel/soft_springStair.skel'
        self.world = pydart.create_world(1.0 / 1000.0, skel_filename)
        logger.info('pydart create_world OK: dt = %f' % self.world.dt)

        # Configure human
        self.skel = self.world.skels[0]
        print 'Skeleton mass = ', self.skel.m
        for i, body in enumerate(self.skel.bodies):
            print i, body.name
        for i, dof in enumerate(self.skel.dofs):
            print i, dof.name

        # # Configure stair: disable the movement of the first step
        # self.stair = self.world.skels[1]
        # self.stair.set_mobile(False)
        self.stair = SpringStair(self.world, self)
        logger.info('set step activation: %s' % step_activation)
        self.stair.set_activation(step_activation)

        # Load the reference motion
        self.ref = FileInfoWorld()
        self.ref.load('data/other/halfCycle.txt')
        logger.info('load reference motions OK: # %d' % self.ref.num_frames)
        self.ref.modify_pose(self.skel)
        self.ref.append_mirrored_motion(self.skel)
        self.ref.append_shifted_motion(self.skel)
        self.ref.add_offset()
        logger.info('modify reference motions OK: # %d' % self.ref.num_frames)

        # Contruct the mutable motion
        # self.motion = StepOffsetMotion(self.skel, self.ref, self.stair)
        # self.motion = RadialBasisMotion(self.skel, self.ref, self.stair)
        # self.motion = NNFeedbackMotion(self.skel, self.ref, self.stair)
        # self.motion = FeedbackMotion(self.skel, self.ref, self.stair)
        # self.motion = SinglePoseMotion(self.skel, self.ref, self.stair)
        # self.motion = WindowedMotion(self.skel, self.ref, self.stair)
        # self.motion = GlobalWindowedMotion(self.skel, self.ref, self.stair)
        self.motion = AdaptiveWindowedMotion(self.skel, self.ref, self.stair)
        self.motion.sim = self

        # Create the controller
        self.skel.controller = Controller(self,
                                          self.skel,
                                          self.world.dt,
                                          self.motion)

        # For check the target
        self.target_index = 0

        # Reset the scene
        self.random_force = np.array([0.0, 0.0, 0.0])
        self.reset_counter = 0
        self.reset()
        self.begin_time = 0.0
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

        # self.random_force = 400.0 * (np.random.rand(3) - 0.5)
        # self.random_force[1] = 0.0
        # self.random_force[2] *= 0.1
        # self.random_force = np.array([200.0, 0.0, 0.0])
        # self.random_force = np.array([0.0, 0.0, 0.0])
        # self.logger.info('force: %s' % self.random_force)
        # if self.reset_counter % 50 == 0:
        #     h = hpy()
        #     print h.heap()
        # self.reset_counter += 1

    def get_time(self):
        return self.world.t + self.begin_time - 0.0

    def get_frame(self):
        begin_index = int(1000.0 * self.begin_time)
        return self.world.frame + begin_index - 0

    def step(self):
        self.stair.apply_force()

        # i = max(self.world.frame, -200)
        i = max(self.get_frame(), 0)
        # if 800 < self.get_frame() < 900:
        #     self.skel.body('h_head').add_ext_force(self.random_force)

        c = self.skel.controller
        c.qhat = self.motion.pose_at_frame(i, isRef=False)
        c.q_ref = self.motion.pose_at_frame(i, isRef=True)
        c.qdhat = self.motion.velocity_at_frame(i, isRef=True)
        # print i, self.skel.contacted_body_names()

        # print self.world.t
        # print self.skel.body('h_heel_left').C
        # print [self.stair.step_height(j) for j in range(3)]

        self.world.step()
        # # Debug purpose
        # self.skel.q = self.motion.pose_at_frame(i, isRef=True)

    def num_frames(self):
        return self.world.num_frames()

    def set_frame(self, idx):
        self.world.set_frame(idx)

    def render(self):
        gltools.render_COM(self.skel)
        self.world.render()
        # self.render_target()
        if 800 < self.get_frame() < 900:
            C = self.skel.body('h_head').C
            f = self.random_force
            gltools.render_line(C, C + 0.1 * f, (1, 0, 0))
        # self.evaluator.render()
        # self.planner.render()

    def render_target(self):
        # frame = min(self.world.frame, self.ref.num_frames - 1)
        frame = min(self.get_frame(), self.ref.num_frames - 1)
        frame = max(frame, 0)
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
        self.logger.info('key pressed: [%s]' % key)
        if key == ']':
            print self.target_index
            self.target_index = (self.target_index + 10) % self.ref.num_frames
            self.update_to_target()
        elif key == '[':
            print self.target_index
            self.target_index = (self.target_index - 10) % self.ref.num_frames
            self.update_to_target()
        else:
            if hasattr(self.motion, 'key_pressed'):
                self.motion.key_pressed(key)

    def optimize(self):
        # self.solver = Optimizer(self, self.motion)
        # self.solver.launch()
        self.motion.launch(self)

    def optimize_hyper(self):
        print('optimize hyper parameters')
        h = StairLatch(self)
        h.launch()

    def kill_optimizer(self):
        self.solver.to_be_killed = True

    def title(self, full=True):
        if self.stair.num_steps() == 0:
            title = 'Normal Stair'
        else:
            title = 'Stair_%.3f' % self.stair._activation
        if not full:
            return title
        if len(self.prefix) > 0:
            title = '%s_%s' % (self.prefix, title)
        if len(self.postfix) > 0:
            title = '%s_%s' % (title, self.postfix)
        return title

    def plot_torques(self):
        pl = PlotterTorque()
        pl.prefix = self.prefix
        pl.postfix = self.postfix
        pl.plot(self.skel.controller, self.title(False))

    def change_solution(self, x, y):
        self.logger.info('change solution x, y = %d, %d' % (x, y))
        self.motion.set_solution(x, y)

    def refresh_solutions(self):
        (nx, ny) = self.motion.num_solutions()
        self.logger.info('nx, ny = %d, %d' % (nx, ny))
        return (nx, ny)
