import logging
import numpy as np
from math import fabs
from parameterized_motion import ParameterizedMotion
from pybrain.tools.shortcuts import buildNetwork
from pydart import SkelVector


class NNFeedbackMotion(ParameterizedMotion):
    def __init__(self, skel, ref, stair):
        super(NNFeedbackMotion, self).__init__(skel, ref)
        self.set_stair_info(stair)
        self.logger = logging.getLogger(__name__)

        self.net_st = buildNetwork(1, 4, bias=True)
        self.net_sw = buildNetwork(1, 3, bias=True)
        # self.param_bal = np.array([1.0, 1.0, 1.0, 1.0])
        self.param_bal = np.array([0.3, 0.03])

        self.logger.info('dim = %d' % self.num_params())
        # p0 = np.random.rand(self.num_params()) - 0.5
        p0 = np.zeros(self.num_params())
        self.set_params(p0)

    def set_stair_info(self, stair):
        self.step_duration = stair.step_duration

    def num_params(self):
        n0 = len(self.net_st.params)
        n1 = len(self.net_sw.params)
        n2 = len(self.param_bal)
        return n0 + n1 + n2

    def set_params(self, params):
        self.params = params
        n0 = len(self.net_st.params)
        n1 = len(self.net_sw.params)
        n2 = len(self.param_bal)
        assert(len(params) == n0 + n1 + n2)
        self.net_st._setParameters(params[:n0])
        self.net_sw._setParameters(params[n0:n0 + n1])
        self.param_bal = params[n0 + n1:]

    def parameterized_pose_at_frame(self, frame_index):
        # Fetch time variables
        t = float(frame_index) * self.h
        phase_t = t % self.step_duration
        step_counter = int(t / self.step_duration)
        # Determine swing and stance foot
        swing = 'left' if step_counter % 2 == 0 else 'right'
        stance = 'right' if step_counter % 2 == 0 else 'left'
        flip = 1.0 if step_counter % 2 == 0 else -1.0

        # Fetch skel info
        skel = self.skel
        C = skel.C
        Cd = skel.Cdot
        # F = skel.body('h_toe_%s' % swing).C
        # Fd = skel.body('h_toe_%s' % swing).Cdot

        # Fetch target trajectories
        Chat = self.ref_com_at_frame(frame_index)
        # if swing == 'left':
        #     Fhat = self.ref_lfoot_at_frame(frame_index)
        # else:
        #     Fhat = self.ref_rfoot_at_frame(frame_index)

        SCALE = 0.3
        # Activate stance foot neural network
        state_st = np.array([phase_t]),
        # state_st[3] *= flip
        # state_st[6] *= flip
        state_st_lo = np.array([0.0])
        state_st_hi = np.array([0.8])
        state_st_norm = (state_st - state_st_lo) / (state_st_hi - state_st_lo)
        # print state_st_norm
        feedback_st = self.net_st.activate(state_st_norm)
        (dStHip, dStKnee, dStHeel, dCz) = 0.0 * feedback_st
        # (dStHip, dStKnee, dStHeel, dStHeel2) = SCALE * feedback_st
        dCz *= flip
        Chat = np.array(Chat + np.array([0.0, 0.0, dCz]))

        # Activate swing foot neural network
        state_sw = np.array([phase_t])

        state_sw_lo = np.array([0.0])
        state_sw_hi = np.array([0.8])
        state_sw_norm = (state_sw - state_sw_lo) / (state_sw_hi - state_sw_lo)
        feedback_sw = self.net_sw.activate(state_sw_norm)
        (dSwHip, dSwKnee, dSwHeel) = 0.0 * feedback_sw

        # Modify the target pose
        q = self.pose_at_frame(frame_index, isRef=True)
        q = SkelVector(q, skel=self.skel)

        heel_offset = -0.15 - 1.0 * (Cd[0] - 0.38)
        # print phase_t, Cd, heel_offset

        q['j_thigh_%s_z' % stance] += dStHip
        q['j_shin_%s' % stance] += dStKnee
        q['j_heel_%s_1' % stance] += (dStHeel + heel_offset)
        # q['j_heel_%s_1' % stance] += (-0.2)
        # q['j_heel_%s_2' % stance] += dStHeel2

        q['j_thigh_%s_z' % swing] += (dSwHip + 0.2)
        q['j_shin_%s' % swing] += dSwKnee
        q['j_heel_%s_1' % swing] += dSwHeel

        # # Modify for the balance
        # kSbal = fabs(self.param_bal[0])
        # kDbal = 0.05 * fabs(self.param_bal[1])
        # lateral = kSbal * (C[2] - Chat[2]) - kDbal * Cd[2]
        # q['j_heel_%s_2' % stance] += lateral

        # kSbal = fabs(self.param_bal[2])
        # kDbal = 0.05 * fabs(self.param_bal[3])
        # lateral = kSbal * (C[0] - Chat[0]) - kDbal * Cd[0]
        # q['j_heel_%s_1' % stance] += lateral

        return q
