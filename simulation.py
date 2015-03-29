import logging
import numpy as np
import pydart
from fileinfoworld import FileInfoWorld
from mutable_motion import MutableMotion
from motion_evaluator import MotionEvaluator
from controller import Controller
from motion_optimizer import MotionOptimizer
import gltools
from spring_stair import SpringStair


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
        # self.world = pydart.create_world(1.0 / 2000.0, skel_filename)
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

        # Contruct the mutable motion
        self.motion = MutableMotion(self.skel, self.ref)

        # Construct the motion evaluator
        self.evaluator = MotionEvaluator(self.skel, self.motion)

        # A new planner
        # self.planner = planner.Planner(self.skel, self.ref)
        # self.planner.solve()
        self.ref.append_mirrored_motion(self.skel)
        self.ref.append_shifted_motion(self.skel)

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
        self.skel.q = self.ref.pose_at(0, self.skel.id)
        q = self.skel.q
        # q[3] -= 0.01
        q['j_thigh_left_z'] += 0.15
        q['j_shin_left'] -= 0.33
        q['j_heel_left_1'] += 0.15
        self.skel.q = q

        self.skel.qdot = 1.0 * self.motion.velocity_at_first()

        self.world.reset()
        self.logger.info('reset OK')

    # def step_stair(self):
    #     if not self.stair.is_mobile():
    #         return
    #     pound_per_inch = 21
    #     kg_per_meter = (pound_per_inch * 0.453) / 0.0254
    #     K = kg_per_meter * 9.8
    #     x = (self.stair.q[0] - 0.2)
    #     xdot = self.stair.qdot[0]
    #     if x < -0.2:
    #         K *= 10
    #     f = -K * x - 0.1 * (K ** 0.5) * xdot
    #     self.stair.tau = np.array([f])

    def step(self):
        self.stair.apply_force()

        # self.skel.controller.update_target_by_frame(self.world.frame)

        # index = int(self.world.t / 0.0005)
        # self.skel.controller.qhat = self.planner.pose_at(index)

        # self.skel.controller.qhat = self.motion.pose_at(self.world.t)
        i = self.world.frame
        self.skel.controller.qhat = self.motion.ref_pose_at_frame(i)
        self.skel.controller.qdhat = self.motion.ref_velocity_at_frame(i)

        t = self.world.t
        offset = []
        offset.append((0.0, 0.05, 0.10, -0.11))
        offset.append((0.1, 0.0, 0.10, -0.3))
        offset.append((1.0, -0.2, 0.20, -0.3))
        for i in range(3):
            offset.append(offset[0])

        for i, H in enumerate(np.arange(0.0, 5.0, 1.0)):
            swing = 'left' if i % 2 == 0 else 'right'
            stance = 'right' if i % 2 == 0 else 'left'
            o = offset[i]
            if H <= t and t <= H + 0.5:
                q = pydart.SkelVector(self.skel.controller.qhat, self.skel)

                q['j_shin_%s' % swing] -= 0.33
                q['j_heel_%s_1' % swing] -= 0.35
                self.skel.controller.qhat = q
            if H + 0.5 <= t and t <= H + 0.7:
                q = pydart.SkelVector(self.skel.controller.qhat, self.skel)
                q['j_shin_%s' % swing] += 0.1
                self.skel.controller.qhat = q

            if H + 0.6 <= t and t <= H + 1.0:
                q = pydart.SkelVector(self.skel.controller.qhat, self.skel)
                q['j_thigh_left_z'] += o[3]
                q['j_thigh_right_z'] += o[3]
                self.skel.controller.qhat = q

            if H + 0.4 <= t and t <= H + 1.0:
                q = pydart.SkelVector(self.skel.controller.qhat, self.skel)
                q['j_thigh_%s_z' % swing] += o[0]
                q['j_shin_%s' % swing] += o[1]
                q['j_heel_%s_1' % swing] += o[2]
                q['j_heel_%s_1' % stance] += o[2]
                self.skel.controller.qhat = q

            # Lateral balance
            if 0.8 <= t and t <= 1.0:
                q = pydart.SkelVector(self.skel.controller.qhat, self.skel)
                delta = -0.0
                q['j_heel_%s_2' % stance] += delta
                q['j_heel_%s_2' % swing] += delta
                self.skel.controller.qhat = q

        # if 0 <= t and t <= 0.5:
        #     q = pydart.SkelVector(self.skel.controller.qhat, self.skel)
        #     q['j_shin_left'] -= 0.33
        #     q['j_heel_left_1'] -= 0.35
        #     self.skel.controller.qhat = q
        # if 0.5 <= t and t <= 0.8:
        #     q = pydart.SkelVector(self.skel.controller.qhat, self.skel)
        #     q['j_shin_left'] += 0.1
        #     self.skel.controller.qhat = q
        # H = 1.0
        # if H <= t and t <= H + 0.5:
        #     q = pydart.SkelVector(self.skel.controller.qhat, self.skel)
        #     # q['j_thigh_left_z'] -= t
        #     q['j_shin_right'] -= (0.33 + 0.0)
        #     q['j_heel_right_1'] -= (0.35 + 0.0)
        #     self.skel.controller.qhat = q
        # if H + 0.5 <= t and t <= H + 0.7:
        #     q = pydart.SkelVector(self.skel.controller.qhat, self.skel)
        #     if t <= H + 0.6:
        #         q['j_thigh_right_z'] += 0.4
        #         q['j_heel_right_1'] += 0.3
        #     q['j_shin_right'] += 0.1
        #     self.skel.controller.qhat = q

        # # Lateral balance
        # if 0.8 <= t and t <= 1.0:
        #     q = pydart.SkelVector(self.skel.controller.qhat, self.skel)
        #     delta = -0.05
        #     q['j_heel_left_2'] += delta
        #     q['j_heel_right_2'] += delta
        #     self.skel.controller.qhat = q

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
