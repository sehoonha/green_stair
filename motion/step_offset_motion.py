import logging
import numpy as np
from parameterized_motion import ParameterizedMotion


class StepOffsetMotion(ParameterizedMotion):
    def __init__(self, skel, ref):
        super(StepOffsetMotion, self).__init__(skel, ref)
        self.logger = logging.getLogger(__name__)

        # Normal stair
        x = [-0.06979409103863092, -0.0062333211920146611, 0.12821950860139653,
             -0.040927784290258395, 0.11082182983765484, -0.045798505123429083,
             0.12428463188061352, -0.26435682810407191, 0.088902665358672056,
             0.029461925769706798, -0.36846259560886868, -0.30061781977860541,
             -0.18754006854554023, 0.015524538336520361, 0.11532298172260619,
             0.045703546214312232, -0.18463533733584606, -0.21381776491423959,
             0.0]

        # Spring stair
        x = [0.160700818946, 0.0774264506179, 0.0142236617455, -0.105206379139,
             -0.00315864311922, 0.0521894743191, 0.178763066467, 0.00481455685708,
             0.0947441420027, -0.0218107775943, 0.274129502408, 0.253947235854,
             0.13945311051, 0.20092042782, -0.143099631998, 0.100214020591,
             -0.507101639726, 0.123078975583, 0.084933699163]
        self.set_params(x)

    def num_params(self):
        return 19

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
        swing_shin = self.params[-3]
        swing_heel = self.params[-2]

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
