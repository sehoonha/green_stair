import logging
import numpy as np
from parameterized_motion import ParameterizedMotion


class StepOffsetMotion(ParameterizedMotion):
    def __init__(self, skel, ref):
        self.logger = logging.getLogger(__name__)
        super(StepOffsetMotion, self).__init__(skel, ref)
        # self.set_params([0.0, 0.05, 0.10, -0.11,
        #                  0.1, 0.0, 0.10, -0.3,
        #                  1.0, -0.2, 0.20, -0.3,
        #                  -0.33, -0.35])
        self.set_params([0.0, 0.05, 0.10, -0.0,
                         0.1, 0.0, 0.10, -0.3,
                         1.0, -0.2, 0.20, -0.3,
                         -0.43, -0.45])

    def num_params(self):
        return 14

    def parameterized_pose_at_frame(self, frame_index):
        q = self.pose_at_frame(frame_index, isRef=True)
        t = float(frame_index) * self.h

        # Build offsets
        offset = []
        offset.append(self.params[0:4])
        offset.append(self.params[4:8])
        offset.append(self.params[8:12])
        for i in range(3):
            offset.append(offset[0])
        swing_shin = self.params[12]
        swing_heel = self.params[13]

        for i, H in enumerate(np.arange(0.0, 5.0, 1.0)):
            swing = 'left' if i % 2 == 0 else 'right'
            stance = 'right' if i % 2 == 0 else 'left'
            o = offset[i]
            if H <= t and t <= H + 0.5:
                q['j_shin_%s' % swing] += swing_shin
                q['j_heel_%s_1' % swing] += swing_heel
            if H + 0.5 <= t and t <= H + 0.7:
                q['j_shin_%s' % swing] += 0.1

            if H + 0.6 <= t and t <= H + 1.0:
                q['j_thigh_left_z'] += o[3]
                q['j_thigh_right_z'] += o[3]

            if H + 0.4 <= t and t <= H + 1.0:
                q['j_thigh_%s_z' % swing] += o[0]
                q['j_shin_%s' % swing] += o[1]
                q['j_heel_%s_1' % swing] += o[2]
                q['j_heel_%s_1' % stance] += o[2]

            # Lateral balance
            if H + 0.0 <= t and t <= H + 0.4:
                delta = 0.1
                q['j_thigh_%s_x' % stance] += delta
                q['j_heel_%s_2' % stance] += delta
                # q['j_heel_%s_2' % swing] += delta
        return q
