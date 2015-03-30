import logging
import numpy as np
from parameterized_motion import ParameterizedMotion


class StepOffsetMotion(ParameterizedMotion):
    def __init__(self, skel, ref):
        super(StepOffsetMotion, self).__init__(skel, ref)
        self.logger = logging.getLogger(__name__)
        # self.set_params([0.0, 0.05, 0.10, -0.11,
        #                  0.1, 0.0, 0.10, -0.3,
        #                  1.0, -0.2, 0.20, -0.3,
        #                  -0.33, -0.35])
        x = [0.0112808523856, 0.10563409698, 0.0433482286517, 0.18999635747,
             0.182087334932, -0.166626233605, -0.170814864498, 0.114218210145,
             -0.199507253573, 0.306833789884, -0.183338383132, -0.062639245908,
             -0.199507253573, 0.306833789884, -0.183338383132, -0.062639245908,
             -0.381200717721, -0.354133232098]
        # x = [0.0507428153966, 0.104819922863, 0.200986650035, 0.240743314868,
        #      0.352322688635, 0.211605666118, -0.181478472392, 0.0870507587992,
        #      -0.239603273668, 0.368922819081, -0.42845834124, 0.163793369655,
        #      -0.374873211191, -0.407907423573]
        self.set_params(x)

    def num_params(self):
        return 18

    def parameterized_pose_at_frame(self, frame_index):
        q = self.pose_at_frame(frame_index, isRef=True)
        t = float(frame_index) * self.h

        # Build offsets
        offset = []
        offset.append(self.params[0:4])
        offset.append(self.params[4:8])
        offset.append(self.params[8:12])
        offset.append(self.params[12:16])
        for i in range(3):
            offset.append(offset[0])
        swing_shin = self.params[-2]
        swing_heel = self.params[-1]

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
                delta = 0.0
                q['j_thigh_%s_x' % stance] += delta
                q['j_heel_%s_2' % stance] += delta
                # q['j_heel_%s_2' % swing] += delta
        return q
