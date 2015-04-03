import logging
import numpy as np
from parameterized_motion import ParameterizedMotion


class StepOffsetMotion(ParameterizedMotion):
    def __init__(self, skel, ref, stair):
        super(StepOffsetMotion, self).__init__(skel, ref)
        self.logger = logging.getLogger(__name__)
        self.set_stair_info(stair)

        # Normal stair
        x = [-0.06979409103863092, -0.0062333211920146611, 0.12821950860139653,
             -0.040927784290258395, 0.11082182983765484, -0.045798505123429083,
             0.12428463188061352, -0.26435682810407191,
             0.4461699303186, -0.0611590225684, 0.1234956071055, -0.1084959537572,
             -0.18463533733584606, -0.21381776491423959, 0.0]

        # # Spring stair
        # x = [0.160700818946, 0.0774264506179, 0.0142236617455, -0.105206379139,
        #      -0.00315864311922, 0.0521894743191, 0.178763066467, 0.00481455685708,
        #      0.2247441420027, -0.0218107775943, 0.074129502408, 0.253947235854,
        #      -0.507101639726, 0.123078975583, 0.084933699163]
        self.set_params(x)

    def set_stair_info(self, stair):
        self.step_duration = stair.step_duration

    def num_params(self):
        return 11

    def parameterized_pose_at_frame(self, frame_index):
        q = self.pose_at_frame(frame_index, isRef=True)
        t = float(frame_index) * self.h

        # Build offsets
        offset = []
        offset.append(self.params[0:4])
        offset.append(self.params[4:8])
        # offset.append(np.array(self.params[8:12]) * 2.0)
        for i in range(3):
            offset.append(offset[0])
        swing_shin = self.params[-3]
        swing_heel = self.params[-2]

        T = self.step_duration
        for i, H in enumerate(np.arange(0.0, T * 5, T)):
            swing = 'left' if i % 2 == 0 else 'right'
            stance = 'right' if i % 2 == 0 else 'left'
            o = offset[i]
            if H + 0.0 * T <= t <= H + 0.5 * T:
                q['j_shin_%s' % swing] += swing_shin
                q['j_heel_%s_1' % swing] += swing_heel
            if H + 0.5 * T <= t <= H + 0.7 * T:
                q['j_shin_%s' % swing] += 0.1

            if H + 0.6 * T <= t <= H + 1.0 * T:
                q['j_thigh_left_z'] += o[3]
                q['j_thigh_right_z'] += o[3]

            if H + 0.4 * T <= t <= H + 1.0 * T:
                q['j_thigh_%s_z' % swing] += o[0]
                q['j_shin_%s' % swing] += o[1]
                q['j_heel_%s_1' % swing] += o[2]
                q['j_heel_%s_1' % stance] += o[2]

            # Lateral balance
            if H + 0.0 * T <= t <= H + 0.4 * T:
                delta = 0.0
                q['j_thigh_%s_x' % stance] += delta
                q['j_heel_%s_2' % stance] += delta
                # q['j_heel_%s_2' % swing] += delta
        return q
