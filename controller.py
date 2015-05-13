import numpy as np
from numpy.linalg import inv
from jacobian_transpose import JTController


class Controller:
    """ Add damping force to the skeleton """
    def __init__(self, sim, skel, h, motion):
        self.sim = sim
        self.h = h
        self.skel = skel
        self.motion = motion

        # Spring-damper
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        self.qdhat = np.zeros(ndofs)
        self.q_ref = np.zeros(ndofs)
        # self.Kp = np.diagflat([0.0] * 6 + [450.0] * (ndofs - 6))
        # self.Kp = np.diagflat([0.0] * 6 + [520.0] * (ndofs - 6))
        # self.Kd = np.diagflat([0.0] * 6 + [40.0] * (ndofs - 6))
        self.Kp = np.diagflat([0.0] * 6 + [500.0] * 6 +
                              [300.0] * (ndofs - 12))
        self.Kd = np.diagflat([0.0] * 6 + [30.0] * (ndofs - 6))

        # for i in range(self.skel.ndofs):
        #     if 'right' in self.skel.dof(i).name:
        #         self.Kp[i, i] *= 1.3
        #         self.Kd[i, i] *= 1.1
        # print self.Kp

        # Jacobian transpose
        self.jt = JTController(self.skel)
        self.reset()

    # def update_target_by_frame(self, frame_idx):
    #     if frame_idx < self.ref.num_frames:
    #         self.qhat = self.ref.pose_at(frame_idx, skel_id=0)
    #         I = self.skel.dof_indices(['j_heel_left_1', 'j_heel_right_1'])
    #         self.qhat[I] += 0.2

    def reset(self):
        self.history = dict()
        self.history['tau'] = list()
        self.history['q'] = list()
        self.history['q_ref'] = list()
        self.history['qdot'] = list()
        self.history['toe_contact'] = list()

    def compute(self):
        skel = self.skel

        m = self.skel.m
        g = 9.81
        T = 0.8
        t = self.sim.get_time()
        i = int(t / T)
        phase_t = t % T
        swing = 'left' if i % 2 == 0 else 'right'
        stance = 'right' if i % 2 == 0 else 'left'

        for ii, dof in enumerate(skel.dofs):
            if 'heel' in dof.name:
                if swing in dof.name and 0.6 < phase_t:
                    self.Kp[ii, ii] = 100.0
                elif stance in dof.name and phase_t < 0.1:
                    self.Kp[ii, ii] = 100.0
                else:
                    self.Kp[ii, ii] = 300.0
        # np.set_printoptions(threshold='nan')

        invM = inv(skel.M + self.Kd * self.h)
        p = -self.Kp.dot(skel.q + skel.qdot * self.h - self.qhat)
        d = -self.Kd.dot(skel.qdot - self.qdhat)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.h

        # t = self.skel.world.t
        # if 0.12 < t and t < 0.30:
        #     # tau += self.jt.apply('h_heel_left', [0, 500, 0])
        #     Cy = self.skel.C[1]
        #     Cy_hat = 0.30
        #     Cy_dot = self.skel.Cdot[1]
        #     f = (Cy_hat - Cy) * 1000.0 - Cy_dot * 200.0
        #     tau += self.jt.apply('h_toe_right', [0, -f, 0])

        tau += self.jt.apply('h_heel_%s' % stance, [0, -m * g, 0])
        tau += self.jt.apply('h_heel_%s' % swing, [0, 10.0 * g, 0])

        # Locate a swing foot step
        if 0.0 < phase_t:
            Foffset = np.array([0.0, 0.0, -0.08])
            if swing == 'left':
                Fhat = self.motion.ref_lfoot_at_frame((i + 1) * 800)
                Fhat = np.array(Fhat) + Foffset
            else:
                Fhat = self.motion.ref_rfoot_at_frame((i + 1) * 800)
                Fhat = np.array(Fhat) - Foffset
            F = skel.body('h_toe_%s' % swing).C
            dF = skel.body('h_toe_%s' % swing).Cdot
            force = -2000.0 * (F - Fhat) - 20.0 * dF
            if phase_t < 0.6:
                force[0] = 0.0
                force[1] = 0.0
            # print t, phase_t, F, Fhat, force
            tau += self.jt.apply('h_toe_%s' % swing, force)
            tau += self.jt.apply('h_heel_%s' % swing, force)

        # Locate a head
        if 0.0 < phase_t:
            H = skel.body('h_head').C
            dH = skel.body('h_head').Cdot
            Hhat = self.motion.ref_head_at_frame((i + 1) * 800)
            force = -2000.0 * (H - Hhat) - 20.0 * dH
            tau += self.jt.apply('h_head', force)

        # # Lateral balance
        # if 0.0 < phase_t:
        #     C = skel.C
        #     dC = skel.Cdot
        #     force = -2000.0 * C - 200.0 * dC
        #     force[0] = 0.0
        #     force[1] = 0.0
        #     # force[2] = 0.0
        #     tau += self.jt.apply('h_head', force)
        #     # tau += self.jt.apply('h_heel_%s' % swing, -force)

        # if 0.3 < t and t < 0.40:
        #     tau += self.jt.apply('h_shin_left', [0, 500, 0])
        #     tau += self.jt.apply('h_heel_left', [0, 500, 0])

        # Make sure the first six are zero
        tau[:6] = 0
        self.history['q'].append(skel.q)
        self.history['q_ref'].append(self.q_ref)
        self.history['qdot'].append(skel.qdot)
        self.history['tau'].append(tau)
        toe_contact = 'h_toe_left' in skel.contacted_body_names() or \
                      'h_heel_left' in skel.contacted_body_names()
        self.history['toe_contact'].append(toe_contact)
        return tau
