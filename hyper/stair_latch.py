import numpy as np
import logging
import threading
import scipy
import math


def optimizer_worker(obj):
    obj.solve()


class StairLatch(object):
    def __init__(self, sim):
        self.sim = sim
        self.skel = self.sim.skel
        self.logger = logging.getLogger(__name__)

    def cost(self, x):
        sim = self.sim
        world = sim.world
        stair = sim.stair

        # Reset the simulation
        sim.begin_time = 0.0
        stair.set_activation(x)
        sim.reset()

        # Initiliaze the time variables
        t = 0.0
        frame = 0
        h = world.time_step()
        # During the time_window
        while t < 0.8:
            sim.step()
            t += h
            frame += 1

        # Evaluate the history
        v_array = []
        con = self.skel.controller
        j_index = con.skel.dof_index('j_shin_right')
        for frame, (torque, q) in enumerate(zip(con.history['tau'],
                                                con.history['q'])):
            # if frame % 10 == 0:
            #     print frame, torque[j_index]
            v_array.append(torque[j_index] * q[j_index])
        v_final = max([math.fabs(v_i) for v_i in v_array])
        # v_final = max([-v_i for v_i in v_array])

        self.logger.info('release at %.4f --> %.4f' % (x, v_final))
        return v_final

    def launch(self):
        self.logger.info('launch solver thread')
        self.thread = threading.Thread(target=optimizer_worker, args=(self,))

        self.logger.info('thread initialized')
        self.thread.start()
        self.logger.info('thread started')

    def solve(self):
        self.logger.info('solving.....')
        self.logger.info('initial cost = %f' % self.cost(0.3))
        res = scipy.optimize.minimize_scalar(self.cost,
                                             bounds=(0.0, 0.3),
                                             # method='brent')
                                             method='bounded')
        self.logger.info('res = %s' % res)
        self.logger.info('solution = %f' % res.x)
        self.logger.info('solving..... OK')
