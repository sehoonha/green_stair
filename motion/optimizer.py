import logging
import threading
import cma
import numpy as np
from numpy.linalg import norm
import math
import os
import glob


def optimizer_worker(opt):
    opt.solve()


class Optimizer(object):
    def __init__(self, sim, motion):
        self.logger = logging.getLogger(__name__)
        self.sim = sim
        self.motion = motion
        self.to_be_killed = False

    def launch(self):
        self.logger.info('launch the solver as the thread')
        self.thread = threading.Thread(target=optimizer_worker, args=(self,))

        self.logger.info('thread initialized')
        self.thread.start()
        self.logger.info('thread started')

    def cost(self, _x):
        self.eval_counter += 1
        if self.to_be_killed:
            return 10e8
        self.logger.info('eval #%d' % self.eval_counter)

        skel = self.sim.skel
        world = self.sim.world
        stair = self.sim.stair

        # num_steps = self.step_index + 1 if self.step_index != -1 else 2
        # MAX_TIME = float(num_steps) * self.sim.stair.step_duration
        # MAX_TIME += 0.2
        T = stair.step_duration
        RT = -0.1 if stair._activation is None else stair._activation
        # num_steps = self.step_index + 1
        # MAX_TIME = T * 2
        num_steps = 2
        MAX_TIME = T * num_steps
        if self.step_index == 0:
            MAX_TIME += 0.0
        # self.motion.set_params(_x, self.step_index)
        self.motion.set_params(_x)
        self.sim.begin_time = float(self.step_index) * T
        self.logger.info('begin_time = %f' % self.sim.begin_time)
        self.sim.reset()
        if self.init_state is not None:
            world.x = self.init_state
        v = 0.0
        v_1 = 0.0
        v_2 = 0.0
        v_sk = 0.0
        balanced = True
        while world.t < MAX_TIME and balanced:
            self.sim.step()
            # COM deviation
            frame = max(self.sim.get_frame(), 0)
            Chat = self.motion.ref_com_at_frame(frame)
            dist = 0.5 * norm(skel.C - Chat) ** 2
            v_1 += dist

            # Head deviation
            H = skel.body('h_head').C
            Hhat = self.motion.ref_head_at_frame(frame)
            dist = 0.5 * norm(H - Hhat) ** 2
            v_2 += dist

            # Check falling
            MaxDeltaC = np.array([0.4, 0.4, 0.2])
            for i in range(3):
                if math.fabs(skel.C[i] - Chat[i]) > MaxDeltaC[i]:
                    print 'unbalanced', i, skel.C[i], Chat[i]
                    balanced = False

            t = self.sim.get_time()
            sim_step = int(t / self.sim.stair.step_duration) + 1
            # Check early foot take-off
            if world.t < RT + 0.01:
                swing = 'left' if sim_step % 2 == 1 else 'right'
                sw_foot = skel.body('h_heel_%s' % swing)
                sw_foot_y = sw_foot.C[1] - 0.04
                step_y = stair.step_height(self.step_index)
                if math.fabs(sw_foot_y - step_y) > 0.05:
                    print 'early take-off', swing, sw_foot_y, step_y
                    balanced = False

            if sim_step % 2 == 1:
                stance_foot = skel.body('h_toe_right').C
                stance_foot_hat = self.motion.ref_rfoot_at_frame(frame)
            else:
                stance_foot = skel.body('h_toe_left').C
                stance_foot_hat = self.motion.ref_lfoot_at_frame(frame)
            w_sk = np.array([5.0, 1.0, 1.0])
            v_sk += 0.5 * norm((stance_foot - stance_foot_hat) * w_sk) ** 2

        self.final_state = world.x

        # Give more penalty to the final frame
        final_frame_index = int((T * num_steps) / world.dt)
        Chat_T = self.motion.ref_com_at_frame(final_frame_index)
        v_c = 1000.0 * norm(skel.C - Chat_T) ** 2

        # Give more penalty to the head final frame
        H = skel.body('h_head').C
        Hhat_T = self.motion.ref_head_at_frame(final_frame_index)
        w_hh = np.array([1.0, 1.0, 1.0])
        v_hh = 1000.0 * norm((H - Hhat_T) * w_hh) ** 2

        # Final COMdot to the initial frame (continuous momentum)
        Cdothat_T = np.array(self.motion.ref_com_dot_at_frame(0))
        if num_steps == 1:
            Cdothat_T[0] *= 0.6
            Cdothat_T[2] *= -1
        elif num_steps == 2:
            Cdothat_T[0] *= 0.75
        elif num_steps == 3:
            Cdothat_T[0] *= 0.75
            Cdothat_T[2] *= -1
        w_cd = np.array([20.0, 0.5, 10.0])
        v_cd = 10.0 * norm((skel.Cdot - Cdothat_T) * w_cd) ** 2
        self.logger.info('%s, %s --> %f' % (Cdothat_T, skel.Cdot, v_cd))

        # Final foot location
        if num_steps % 2 == 1:
            swing_foot = skel.body('h_toe_left').C
            swing_foot_hat = self.motion.ref_lfoot_at_frame(final_frame_index)
        else:
            swing_foot = skel.body('h_toe_right').C
            swing_foot_hat = self.motion.ref_rfoot_at_frame(final_frame_index)
        w_f = np.array([10.0, 10.0, 1.0])
        v_f = 1000.0 * norm((swing_foot - swing_foot_hat) * w_f) ** 2

        v = v_1 + v_2 + v_sk + v_hh + v_c + v_cd + v_f
        v = 10.0 * v + 500.0 * (MAX_TIME - world.t)
        self.logger.info('%.6f (%.4f) <-- %.4f %.4f %.4f/%.4f %.4f %.4f/%.4f' %
                         (v, world.t, v_1, v_2, v_sk, v_c, v_hh, v_cd, v_f))

        # self.logger.info('%s' % repr(list(_x)))
        # self.logger.info('final %s' % repr(list(skel.x)))
        self.logger.info('')
        if self.eval_counter % 48 == 0:
            for i in range(5):
                self.logger.info('    >==============<')

        return v

    def solve_step(self, step_index):
        self.to_be_killed = False
        self.eval_counter = 0
        self.step_index = step_index
        opts = cma.CMAOptions()
        opts.set('verb_disp', 1)
        if step_index == 0:
            opts.set('ftarget', 150.0)
            opts.set('popsize', 64)
            opts.set('maxiter', 200)
        else:
            opts.set('ftarget', 300.0)
            opts.set('popsize', 64)
            opts.set('maxiter', 200)

        # dim = self.motion.num_params(self.step_index)
        # x0 = np.zeros(dim)
        # x0 = self.motion.params_at_step(self.step_index)
        x0 = self.motion.params
        self.logger.info('')
        self.logger.info('')
        self.logger.info('------------------ CMA-ES ------------------')
        self.logger.info('  step_index = %d' % self.step_index)
        res = cma.fmin(self.cost, x0, 0.2, opts)
        self.to_be_killed = False
        self.logger.info('--------------------------------------------')
        self.logger.info('solution: %s' % repr(res[0]))
        self.logger.info('value: %.6f' % res[1])
        self.logger.info('--------------------------------------------')
        self.logger.info('  set parameters for step = %d' % self.step_index)
        self.motion.set_params(res[0])
        # self.motion.set_params(res[0], self.step_index)
        self.logger.info('  final and set the new initial state')
        self.cost(res[0])
        self.init_state = self.final_state
        self.logger.info('  final and set the new initial state .. OK')
        self.logger.info('  copy result files step = %d' % self.step_index)
        for fin in glob.glob('outcmaes*.dat'):
            fout = fin.replace('outcmaes', 'step%d_outcmaes' % self.step_index)
            cmd = 'mv %s %s' % (fin, fout)
            self.logger.info('cmd = [%s]' % cmd)
            os.system(cmd)
        return res

    def solve(self):
        self.init_state = None
        max_step = 1
        answers = []
        for step in range(max_step):
            res = self.solve_step(step)
            answers.append(res)
        # res = self.solve_step(-1)
        # answers.append(res)

        self.logger.info('--------------- all results ----------------')
        for res in answers:
            self.logger.info('solution: %s' % repr(res[0]))
            self.logger.info('value: %.6f' % res[1])
        self.logger.info('--------------------------------------------')
        self.logger.info('---------------- parameters ----------------')
        self.logger.info('optimal parameters: %s' % repr(self.motion.params))
        self.logger.info('--------------------------------------------')
        self.logger.info('reset begin time')
        self.sim.begin_time = 0.0
        self.logger.info('--------------------------------------------')
