import numpy as np
from numpy.linalg import inv
from jacobian_transpose import JTController


class Controller:
    """ Add damping force to the skeleton """
    def __init__(self, skel, h, ref):
        self.h = h
        self.skel = skel
        self.ref = ref

        # Spring-damper
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        self.qdhat = np.zeros(ndofs)
        # self.Kp = np.diagflat([0.0] * 6 + [450.0] * (ndofs - 6))
        self.Kp = np.diagflat([0.0] * 6 + [500.0] * (ndofs - 6))
        self.Kd = np.diagflat([0.0] * 6 + [40.0] * (ndofs - 6))

        # for i in range(self.skel.ndofs):
        #     if 'right' in self.skel.dof(i).name:
        #         self.Kp[i, i] *= 1.3
        #         self.Kd[i, i] *= 1.1
        # print self.Kp

        # Jacobian transpose
        self.jt = JTController(self.skel)

    # def update_target_by_frame(self, frame_idx):
    #     if frame_idx < self.ref.num_frames:
    #         self.qhat = self.ref.pose_at(frame_idx, skel_id=0)
    #         I = self.skel.dof_indices(['j_heel_left_1', 'j_heel_right_1'])
    #         self.qhat[I] += 0.2

    def compute(self):
        skel = self.skel

        invM = inv(skel.M + self.Kd * self.h)
        p = -self.Kp.dot(skel.q + skel.qdot * self.h - self.qhat)
        d = -self.Kd.dot(skel.qdot - self.qdhat)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.h

        t = self.skel.world.t
        # if 0.12 < t and t < 0.30:
        #     # tau += self.jt.apply('h_heel_left', [0, 500, 0])
        #     Cy = self.skel.C[1]
        #     Cy_hat = 0.30
        #     Cy_dot = self.skel.Cdot[1]
        #     f = (Cy_hat - Cy) * 1000.0 - Cy_dot * 200.0
        #     tau += self.jt.apply('h_toe_right', [0, -f, 0])

        m = self.skel.m
        g = 9.81
        if t < 1.0:
            tau += self.jt.apply('h_heel_right', [0, -m * g, 0])
        else:
            tau += self.jt.apply('h_heel_left', [0, -m * g, 0])

        # if 0.3 < t and t < 0.40:
        #     tau += self.jt.apply('h_shin_left', [0, 500, 0])
        #     tau += self.jt.apply('h_heel_left', [0, 500, 0])

        # Make sure the first six are zero
        tau[:6] = 0
        return tau
