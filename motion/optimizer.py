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

        MAX_TIME = 2.8
        self.motion.set_params(_x)
        self.sim.reset()
        v = 0.0
        balanced = True
        while world.t < MAX_TIME and balanced:
            self.sim.step()
            # Fast check of balance
            Chat = self.motion.ref_com_at_frame(world.frame)
            dist = 0.5 * norm(skel.C - Chat) ** 2
            v += dist

            MaxDeltaC = np.array([0.4, 0.4, 0.2])
            for i in range(3):
                if math.fabs(skel.C[i] - Chat[i]) > MaxDeltaC[i]:
                    balanced = False

        # Give more penalty to the final frame
        final_frame_index = int(MAX_TIME / world.dt)
        Chat_T = self.motion.ref_com_at_frame(final_frame_index)
        v += 500.0 * norm(skel.C - Chat_T) ** 2

        # Final COMdot
        v += 500.0 * norm(skel.Cdot) ** 2

        # Range check (0.05 --> 5000.0?)
        bound_penalty = 0.0
        if _x[-1] < 0.0:
            bound_penalty += (0.0 - _x[-1]) ** 2
        if _x[-1] > 0.2:
            bound_penalty += (_x[-1] - 0.2) ** 2

        v = v + 5000.0 * (MAX_TIME - world.t) + 10e6 * bound_penalty
        self.logger.info('%.6f (%.4f) <-- %s' % (v, world.t, repr(list(_x))))
        return v

    def solve(self):
        opts = cma.CMAOptions()
        opts.set('verb_disp', 1)
        opts.set('ftarget', 5.0)
        opts.set('popsize', 32)
        opts.set('maxiter', 1000)

        dim = self.motion.num_params()
        x0 = np.zeros(dim)
        self.logger.info('------------------ CMA-ES ------------------')
        res = cma.fmin(self.cost, x0, 0.1, opts)
        self.logger.info('--------------------------------------------')
        self.logger.info('solution: %s' % res[0])
