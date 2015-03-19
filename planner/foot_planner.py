import logging
import numpy as np
import gltools


class FootPlanner(object):
    def __init__(self, skel, ref):
        self.logger = logging.getLogger(__name__)
        self.skel = skel
        self.ref = ref
        self.offset = np.zeros(3)

        self.toes = []
        for q in self.ref.poses(self.skel.id):
            self.skel.q = q
            toe = self.skel.body('h_toe_left').to_world([0.13, -0.05, 0.0])
            self.toes.append(toe)

        self.set_params([0.0, 0.1, 0.0, 0.1])

    def num_params(self):
        return 4

    def set_params(self, params):
        a, b, c, d = params
        self.pts = np.array([[0, a, c, 0],
                             [0, b, d, 0],
                             [0, 0, 0, 0]])

    def params(self):
        ab = self.pts[1][:2]
        cd = self.pts[2][:2]
        return np.concatenate((ab, cd))

    def toe_at_frame(self, index):
        w = float(index) / float(len(self.toes) - 1.0)
        w0 = (1 - w) * (1 - w) * (1 - w)
        w1 = 3.0 * (1 - w) * (1 - w) * w
        w2 = 3.0 * (1 - w) * w * w
        w3 = w * w * w
        offset = self.pts.dot(np.array([w0, w1, w2, w3]))
        return self.toes[index] + offset + self.offset

    def bake(self):
        nframes = len(self.toes)
        self.traj = [self.toe_at_frame(i) for i in range(nframes)]

    def cost(self, params, verbose=False):
        pass

    def solve(self):
        self.bake()

    def shift(self, x=0, y=0, z=0):
        self.offset = np.array([x, y, z])
        self.bake()

    def render(self):
        gltools.render_trajectory(self.traj, [1.0, 0.0, 1.0])
