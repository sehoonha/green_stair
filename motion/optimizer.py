import logging
import threading
import cma
import numpy as np
from numpy.linalg import norm
import math


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
        if self.to_be_killed:
            return 10e8

        skel = self.sim.skel
        world = self.sim.world

        num_steps = 3
        MAX_TIME = float(num_steps) * self.sim.stair.step_duration
        self.motion.set_params(_x)
        self.sim.reset()
        v = 0.0
        v_1 = 0.0
        v_2 = 0.0
        balanced = True
        while world.t < MAX_TIME and balanced:
            self.sim.step()
            # COM deviation
            Chat = self.motion.ref_com_at_frame(world.frame)
            dist = 0.5 * norm(skel.C - Chat) ** 2
            v_1 += dist

            # Head deviation
            H = skel.body('h_head').C
            Hhat = self.motion.ref_head_at_frame(world.frame)
            dist = 0.5 * norm(H - Hhat) ** 2
            v_2 += dist

            # Check falling
            MaxDeltaC = np.array([0.4, 0.4, 0.2])
            for i in range(3):
                if math.fabs(skel.C[i] - Chat[i]) > MaxDeltaC[i]:
                    balanced = False

        # Give more penalty to the final frame
        final_frame_index = int(MAX_TIME / world.dt)
        Chat_T = self.motion.ref_com_at_frame(final_frame_index)
        v_c = 1000.0 * norm(skel.C - Chat_T) ** 2

        # Final COMdot to the initial frame (continuous momentum)
        Cdothat_T = np.array(self.motion.ref_com_dot_at_frame(0))
        Cdothat_T[0] *= 0.8
        if num_steps % 2 == 1:
            Cdothat_T[2] *= -1
        w_cd = np.array([10.0, 0.5, 10.0])
        v_cd = 10.0 * norm((skel.Cdot - Cdothat_T) * w_cd) ** 2
        self.logger.info('%s, %s --> %f' % (Cdothat_T, skel.Cdot, v_cd))

        # Final foot location
        if num_steps % 2 == 1:
            swing_foot = skel.body('h_toe_left').C
            swing_foot_hat = self.motion.ref_lfoot_at_frame(final_frame_index)
        else:
            swing_foot = skel.body('h_toe_right').C
            swing_foot_hat = self.motion.ref_rfoot_at_frame(final_frame_index)
        v_f = 1000.0 * norm(swing_foot - swing_foot_hat) ** 2

        # Range check (0.05 --> 5000.0?)
        bound_penalty = 0.0
        # if _x[-1] < 0.0:
        #     bound_penalty += (0.0 - _x[-1]) ** 2
        # if _x[-1] > 0.2:
        #     bound_penalty += (_x[-1] - 0.2) ** 2

        v = v_1 + v_2 + v_c + v_cd + v_f
        v = 10.0 * v + 500.0 * (MAX_TIME - world.t) + 10e5 * bound_penalty
        self.logger.info('%.6f (%.4f) <-- %.4f %.4f %.4f %.4f %.4f' %
                         (v, world.t, v_1, v_2, v_c, v_cd, v_f))

        self.logger.info('%s' % repr(list(_x)))
        self.logger.info('')
        return v

    def solve(self):
        opts = cma.CMAOptions()
        opts.set('verb_disp', 1)
        opts.set('ftarget', 5.0)
        opts.set('popsize', 32)
        opts.set('maxiter', 1000)

        # dim = self.motion.num_params()
        # x0 = np.zeros(dim)
        x0 = self.motion.params
        self.logger.info('------------------ CMA-ES ------------------')
        res = cma.fmin(self.cost, x0, 0.2, opts)
        self.logger.info('--------------------------------------------')
        self.logger.info('solution: %s' % repr(res[0]))
        self.logger.info('value: %.6f' % res[1])
