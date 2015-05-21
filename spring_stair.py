import logging
import numpy as np


class SpringStair(object):
    def __init__(self, world, sim):
        self.logger = logging.getLogger(__name__)
        self.world = world
        self.sim = sim
        self.step_duration = 0.8

        # Default stiffness
        pound_per_inch = 21
        kg_per_meter = (pound_per_inch * 0.453) / 0.0254
        K = kg_per_meter * 9.8

        # Collect the stairs
        self.stairs = []
        for skel in self.world.skels:
            if 'step' not in skel.name:
                continue
            params = dict()
            params['name'] = skel.name
            params['skel'] = skel
            params['init_height'] = skel.q[0]
            params['rest_height'] = skel.q[0] + 0.2
            params['activation'] = float(len(self.stairs)) * self.step_duration
            params['K'] = K
            self.stairs.append(params)
            # Debug output
            self.logger.info('new spring: %s' % params['name'])
            for k, v in params.iteritems():
                if k == 'skel':
                    continue
                self.logger.info('  %s: %s' % (k, v))

    def num_steps(self):
        return len(self.stairs)

    def reset(self):
        for params in self.stairs:
            skel = params['skel']
            init_height = params['init_height']
            skel.q = np.array([init_height])
            skel.qdot = np.array([0.0])
            skel.set_mobile(False)

    def activate(self):
        # t = self.world.t
        t = self.sim.get_time()
        for i, params in enumerate(self.stairs):
            skel = params['skel']
            activation = params['activation']
            # if not skel.is_mobile() and activation < t and i != 0:
            if not skel.is_mobile() and activation < t:
                skel.set_mobile(True)
                # self.logger.info('Activate %s at %f' % (params['name'], t))

    def set_activation(self, dt):
        self._activation = dt
        for i, params in enumerate(self.stairs):
            T = self.step_duration
            params['activation'] = float(i) * T + dt

    def apply_force(self):
        self.activate()

        for params in self.stairs:
            skel = params['skel']
            if not skel.is_mobile():
                continue
            rest_height = params['rest_height']
            K = params['K']

            x = (skel.q[0] - rest_height)
            xdot = skel.qdot[0]
            if x < -0.2:
                K *= 10
            f = -K * x - 0.5 * (K ** 0.5) * xdot
            skel.tau = np.array([f])

    def step_height(self, step_index):
        skel = self.stairs[step_index]['skel']
        return skel.C[1] + 0.1
