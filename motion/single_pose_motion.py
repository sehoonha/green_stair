import numpy as np
import logging
from parameterized_motion import ParameterizedMotion
from pydart import SkelVector


class SinglePoseMotion(ParameterizedMotion):
    def __init__(self, skel, ref, stair):
        super(SinglePoseMotion, self).__init__(skel, ref)
        self.set_stair_info(stair)
        self.logger = logging.getLogger(__name__)

        # Set weight
        ndofs = self.skel.ndofs
        w = SkelVector(0.3 * np.ones(ndofs), self.skel)
        w[:6] = 0
        self.w = w
        for i in range(ndofs):
            name = self.skel.dofs[i].name
            if ('bicep' in name) or ('forearm' in name) or ('hand' in name):
                w[i] = 0

        # Set parameter
        p0 = np.random.rand(self.num_params()) - 0.5
        # p0 = np.zeros(self.num_params())
        p0 = np.array([-0.14766899,  0.07656574,  0.05981923, -0.28327248,  0.41531486,
                       -1.22343379,  0.24066063,  0.20566484, -0.13205165,  0.26162915,
                       0.13429722,  0.77767611,  0.49972143, -0.54483672,  0.56015346,
                       -0.01900684, -0.0020604 ,  0.17788071,  0.17654917,  0.18143294,
                       -0.33411403])
        self.set_params(p0)

        self.logger.info('dim = %s' % self.num_params())
        self.logger.info('weight = %s' % self.w)
        self.logger.info('params = %s' % self.params)
        self.logger.info('delta = %s' % self.delta())

    def set_stair_info(self, stair):
        self.step_duration = stair.step_duration

    def num_params(self):
        return sum([1 if w_i > 0.0001 else 0 for w_i in self.w])

    def set_params(self, params):
        self.params = params

    def delta(self):
        ndofs = self.skel.ndofs
        ret = np.zeros(ndofs)
        j = 0
        for i in range(ndofs):
            if self.w[i] < 0.0001:
                continue
            else:
                ret[i] += self.w[i] * self.params[j]
                j += 1
        return ret

    def parameterized_pose_at_frame(self, frame_index):
        q0 = self.pose_at_frame(frame_index, isRef=True)
        q = q0 + self.delta()

        return q
