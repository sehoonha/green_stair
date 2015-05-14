import logging
import numpy as np
import math
from parameterized_motion import ParameterizedMotion
from pydart import SkelVector


class FeedbackMotion(ParameterizedMotion):
    def __init__(self, skel, ref, stair):
        super(FeedbackMotion, self).__init__(skel, ref)
        self.set_stair_info(stair)
        self.logger = logging.getLogger(__name__)

        self.logger.info('dim = %d' % self.num_params())
        p0 = np.zeros(self.num_params())
        self.set_params(p0)

    def set_stair_info(self, stair):
        self.step_duration = stair.step_duration

    def num_params(self):
        return 8

    def set_params(self, params):
        self.params = params

    def parameterized_pose_at_frame(self, frame_index):
        # Fetch time variables
        t = float(frame_index) * self.h
        T = self.step_duration
        phase_t = t % T
        step_counter = int(t / self.step_duration)
        # Determine swing and stance foot
        swing = 'left' if step_counter % 2 == 0 else 'right'
        stance = 'right' if step_counter % 2 == 0 else 'left'
        flip = 1.0 if step_counter % 2 == 0 else -1.0

        # Fetch skel info
        skel = self.skel
        # C = skel.C
        Cd = skel.Cdot
        # Chat = self.ref_com_at_frame(frame_index)

        # Modify the target pose
        q = self.pose_at_frame(frame_index, isRef=True)
        q = SkelVector(q, skel=self.skel)
        i = np.nditer(self.params)

        # Lift the swing thigh
        q['j_thigh_%s_z' % swing] += 0.2
        (a, b) = (i.next(), i.next())
        # sw_heel_offset = 0.0 - 1.0 * (Cd[0] - 0.38)
        sw_heel_offset = a - b * (Cd[0] - 0.38)
        q['j_heel_%s_1' % swing] += sw_heel_offset
        # print phase_t, Cd, sw_heel_offset

        # Adjust the stance heel w.r.t. forward momentum
        (a, b) = (i.next(), i.next())
        # st_heel_offset = -0.20 - 0.2 * (Cd[0] - 0.38)
        st_heel_offset = a - b * (Cd[0] - 0.38)
        # print phase_t, Cd, st_heel_offset
        q['j_heel_%s_1' % stance] += st_heel_offset - 0.15

        # Adjust the stance thigh w.r.t. forward momentum
        (a, b) = (i.next(), i.next())
        # st_heel_offset = -0.20 - 0.2 * (Cd[0] - 0.38)
        st_thigh_offset = 0.1 * (a - b * (Cd[0] - 0.38))
        # print phase_t, Cd, st_heel_offset
        q['j_thigh_%s_z' % stance] += st_thigh_offset

        # Adjust the stance thigh w.r.t. forward momentum
        CdZ = Cd[2] * flip
        # print phase_t, CdZ
        if CdZ < -0.20:
            (a, b) = (i.next(), i.next())
            st_heel2_offset = a - b * (CdZ + 0.20)
            st_heel2_offset *= flip
            # print phase_t, CdZ, st_heel2_offset
            # st_heel2_offset = 0.5
            q['j_heel_%s_2' % stance] -= st_heel2_offset

        # Adjust the lateral swing heel
        swFT = skel.body('h_toe_%s' % swing).T
        (ax, ay, az) = self.mat2euler(swFT)
        # print az
        q['j_heel_%s_2' % swing] += -5.0 * az

        return q

    def mat2euler(self, _M, cy_thresh=None):
        M = np.asarray(_M)
        M = M[:3, :3]
        if cy_thresh is None:
            try:
                cy_thresh = np.finfo(M.dtype).eps * 4
            except ValueError:
                cy_thresh = np._FLOAT_EPS_4
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r33 * r33 + r23 * r23)
        if cy > cy_thresh:  # cos(y) not close to zero, standard form
            z = math.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21, r22)
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = 0.0
        return z, y, x
