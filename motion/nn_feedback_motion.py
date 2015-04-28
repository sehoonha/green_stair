import logging
import numpy as np
from parameterized_motion import ParameterizedMotion
from pybrain.tools.shortcuts import buildNetwork
from pydart import SkelVector


class NNFeedbackMotion(ParameterizedMotion):
    def __init__(self, skel, ref, stair):
        super(NNFeedbackMotion, self).__init__(skel, ref)
        self.set_stair_info(stair)
        self.logger = logging.getLogger(__name__)

        self.net = buildNetwork(7, 7, bias=True)

        self.logger.info('dim = %d' % self.num_params())
        # p0 = np.random.rand(self.num_params()) - 0.5
        p0 = np.zeros(self.num_params())
        self.set_params(p0)

    def set_stair_info(self, stair):
        self.step_duration = stair.step_duration

    def num_params(self):
        return len(self.net.params)

    def set_params(self, params):
        self.params = params
        self.net._setParameters(params)

    def parameterized_pose_at_frame(self, frame_index):
        # Fetch time variables
        t = float(frame_index) * self.h
        phase_t = t % self.step_duration
        step_counter = int(t / self.step_duration)

        # Fetch skel info
        skel = self.skel
        C = skel.C
        Cd = skel.Cdot

        # Activate neural network
        state = np.concatenate([[phase_t], C, Cd])
        feedback = self.net.activate(state)
        (dSwHip, dSwKnee, dSwHeel,
         dStHip, dStKnee, dStHeel, dStHeel2) = feedback

        # Modify the target pose
        q = self.pose_at_frame(frame_index, isRef=True)
        q = SkelVector(q, skel=self.skel)
        swing = 'left' if step_counter % 2 == 0 else 'right'
        stance = 'right' if step_counter % 2 == 0 else 'left'
        q['j_thigh_%s_z' % swing] += dSwHip
        q['j_shin_%s' % swing] += dSwKnee
        q['j_heel_%s_1' % swing] += dSwHeel
        q['j_thigh_%s_z' % stance] += dStHip
        q['j_shin_%s' % stance] += dStKnee
        q['j_heel_%s_1' % stance] += dStHeel
        q['j_heel_%s_2' % stance] += dStHeel2
        return q
