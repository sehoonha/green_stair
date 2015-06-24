import numpy as np
from numpy.linalg import norm
import cma
import logging
from parameterized_motion import ParameterizedMotion
from pydart import SkelVector
import threading


def optimizer_worker(motion):
    motion.solve()


class Sample(object):
    def __init__(self, motion, t0, dt, prev, params=None):
        self.logger = logging.getLogger(__name__)
        self.motion = motion
        self.skel = self.motion.skel
        self.t0 = t0
        self.dt = dt
        self.prev = prev
        self.params = params
        # Init variables
        self.score = None
        self.final_state = None
        self.init_weights()

    def init_weights(self):
        ndofs = self.skel.ndofs
        w = SkelVector(0.3 * np.ones(ndofs), self.skel)
        w[:6] = 0
        for i in range(ndofs):
            name = self.skel.dofs[i].name
            if ('bicep' in name) or ('forearm' in name) or ('hand' in name):
                w[i] = 0.5
            if ('thigh' in name) and ('_z' in name):
                w[i] = 2.0
            if ('shin' in name):
                w[i] = 2.0
        self.w = w

    def initial_state(self):
        if self.prev is None:
            return self.motion.init_state
        return self.prev.final_state

    def prev_score(self):
        if self.prev is None:
            return 0.0
        return self.prev.score

    def set_params(self, params):
        self.params = params

    def num_params(self):
        return sum([1 if w_i > 0.0001 else 0 for w_i in self.w])

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

    def index0(self):
        t = self.t0
        world = self.motion.sim.world
        h = world.time_step()
        frame = int(t / h)
        return frame

    def cost(self):
        self.motion.current_sample = self
        sim = self.motion.sim
        skel = sim.skel
        world = sim.world
        sim.reset()
        sim.begin_time = self.t0
        world.x = self.initial_state()

        v = dict()
        for key in ['x', 'q', 'qdot', 'C', 'Cd', 'FL', 'FR', 'H']:
            v[key] = 0.0

        v['x'] = 0.5 * norm(self.params) ** 2

        # Use a local time t
        t = self.t0
        step_duration = 0.8
        h = world.time_step()
        frame = self.index0()
        while t < self.t0 + self.dt:
            sim.step()
            phase_t = t % step_duration

            # Measure v['q']
            q = skel.q
            qhat = self.motion.ref_pose_at_frame(frame)
            q_diff = q - qhat
            # q_diff[:6] = 0.0
            v['q'] += (2.0 / (skel.ndofs)) * 0.5 * norm(q_diff) ** 2

            # Measure v['qdot']
            qdot = skel.qdot
            v['qdot'] += (0.02 / (skel.ndofs)) * 0.5 * norm(qdot) ** 2

            # Measure v['C']
            C = skel.C
            Chat = self.motion.ref_com_at_frame(frame)
            v['C'] += 0.5 * 0.5 * (norm(C - Chat) ** 2)

            # Measure v['Cd']
            Cd = skel.Cdot
            Cdhat = np.array(self.motion.ref_com_dot_at_frame(frame))
            if 0.5 <= phase_t <= 0.8:
                Cdhat[0] *= 0.7
            else:
                Cdhat[0] *= 0.9
            Cdiff = (Cd - Cdhat) * np.array([1.0, 0.3, 5.0])
            v['Cd'] += 3.0 * 0.5 * (norm(Cdiff) ** 2)

            # Measure v['FL']
            w_fl = 100.0 if (0.6 <= t <= 1.6) else 2.0
            FL = skel.body('h_toe_left').C
            FLhat = self.motion.ref_lfoot_at_frame(frame)
            v['FL'] += w_fl * 0.5 * norm(FL - FLhat) ** 2
            if (0.4 <= t <= 0.6) and FL[1] < FLhat[1]:
                v['FL'] += 1.0
            if (0.0 <= t <= 0.2) and 0.16 < FL[0]:
                v['FL'] += 1.0

            # Measure v['FR']
            w_fr = 100.0 if (0.0 <= t <= 0.8) or (1.4 <= t <= 2.4) else 2.0
            FR = skel.body('h_toe_right').C
            FRhat = self.motion.ref_rfoot_at_frame(frame)
            v['FR'] += w_fr * 0.5 * norm(FR - FRhat) ** 2
            if (1.2 <= t <= 1.4) and FR[1] < FRhat[1]:
                v['FR'] += 1.0

            # Measure v['H']
            H = skel.body('h_head').C
            Hhat = self.motion.ref_head_at_frame(frame)
            v['H'] += 0.5 * norm(H - Hhat) ** 2

            # Increase the time and frame index
            t += h
            frame += 1

        # Update the final state
        self.final_state = world.x

        self.score = 1.0 * self.prev_score()
        self.logger.info('')
        self.logger.info('time window starting at %.2f' % self.t0)
        for key, val in v.iteritems():
            self.logger.info('  > %s --> %.4f' % (key, val))
            self.score += val
        self.logger.info('total --> %.4f' % self.score)
        self.motion.current_sample = None
        return self.score


class GlobalWindowedMotion(ParameterizedMotion):
    def __init__(self, skel, ref, stair):
        super(GlobalWindowedMotion, self).__init__(skel, ref)
        self.set_stair_info(stair)
        self.logger = logging.getLogger(__name__)
        self.sequence = []
        self.T = 0.8
        self.dt = 0.2
        self.current_sample = None

        self.logger.info('GlobalWindowsMotion is constructed')
        self.logger.info('dim = %s' % self.num_params())

    def set_stair_info(self, stair):
        self.step_duration = stair.step_duration

    def num_params(self):
        return sum([s.num_params() for s in self.sequence])

    def set_params(self, params):
        self.params = params

        # Reconstruct the samples
        self.sequence = []
        prev = None
        for t0 in np.arange(0.0, self.T, self.dt):
            s = Sample(self, t0, self.dt, prev)
            self.sequence.append(s)
            prev = s

        sample_num_params = [ss.num_params() for ss in self.sequence]
        print sample_num_params
        cum_num_params = np.cumsum(sample_num_params)
        print cum_num_params
        splitted = np.split(params, cum_num_params)
        print splitted
        for s, p in zip(self.sequence, splitted):
            self.logger.info('set params: %f / %s' % (s.t0, p))
            s.set_params(p)

    def launch(self, sim):
        self.logger.info('launch the solver as the thread')
        self.sim = sim
        self.thread = threading.Thread(target=optimizer_worker, args=(self,))

        self.logger.info('thread initialized')
        self.thread.start()
        self.logger.info('thread started')

    def solve(self):
        self.init_state = self.sim.world.x
        T = self.T
        dt = self.dt
        prev_samples = [None]
        all_samples = list()
        n_iter_samples = 1000
        n_saved_samples = 100
        for t0 in np.arange(0.0, T, dt):
            next_samples = []
            if len(prev_samples) > 1:
                prev_values = [100.0 / s.score for s in prev_samples]
                sum_prev_values = sum(prev_values)
                prev_values = np.array(prev_values) / sum_prev_values
            else:
                prev_values = [1.0]
            # 1. Generate new samples
            while len(next_samples) < n_iter_samples:
                prev = np.random.choice(prev_samples, p=prev_values)
                s = Sample(self, t0, dt, prev)
                dim = s.num_params()
                # params = 0.6 * (np.random.rand(dim) - 0.5)
                m = np.zeros(dim)
                C = 0.15 * np.identity(dim)
                params = np.random.multivariate_normal(m, C)
                s.set_params(params)
                next_samples.append(s)
                self.logger.info('')
                self.logger.info('>' * 50)
                self.logger.info('Evaluating a new sample: %d/%d' %
                                 (len(next_samples), n_iter_samples))
                self.logger.info('  params = %s' % s.params)
                self.logger.info('  prev score = %f' % s.prev_score())
                s.cost()

            # 2. Select the good and various samples
            all_samples.append(prev_samples)
            prev_samples = list()
            next_samples.sort(key=lambda s: s.score)
            prev_samples += next_samples[:n_saved_samples / 2]
            next_samples = next_samples[n_saved_samples / 2:]
            while len(prev_samples) < n_saved_samples:
                s = np.random.choice(next_samples)
                prev_samples.append(s)

            values = [ss.score for ss in prev_samples]
            self.logger.info('')
            self.logger.info('----------------------------')
            self.logger.info('# saved samples = %d' % len(prev_samples))
            self.logger.info('best cost = %f' % min(values))
            self.logger.info('worst cost = %f' % max(values))
            self.logger.info('----------------------------')
            self.logger.info('')

        # Reconstruct samples
        self.logger.info('reconstructing.....')
        self.sequence = []
        opt = prev_samples[0]
        while opt is not None:
            self.logger.info('opt sample.score = %f' % opt.score)
            self.sequence = [opt] + self.sequence
            opt = opt.prev
        self.logger.info('reconstructing..... Done')
        self.logger.info('# sequence = %d' % len(self.sequence))
        self.logger.info('merging parameters..... ')
        for s in self.sequence:
            self.logger.info(str(repr(s.params)))
        params = np.concatenate([s.params for s in self.sequence])
        self.set_params(params)
        self.logger.info('merging parameters..... done')

        self.sim.begin_time = 0.0

    def swing_thigh_offset(self, t):
        step_duration = 0.8
        phase_t = t % step_duration
        ret = SkelVector(np.zeros(self.skel.ndofs), self.skel)
        if not (0.1 <= phase_t <= 0.35):
            return ret
        step_counter = int(t / step_duration)
        swing = 'left' if step_counter % 2 == 0 else 'right'
        stance = 'right' if step_counter % 2 == 0 else 'left'
        ret['j_thigh_%s_z' % swing] += 0.2
        ret['j_thigh_%s_z' % stance] -= 0.2
        ret['j_heel_%s_1' % stance] -= 0.2
        # print t, phase_t, ret
        return ret

    def parameterized_pose_at_frame(self, frame_index):
        if self.current_sample is None:
            # Fetch time information
            t = float(frame_index) * self.h
            win_index = int(t / self.dt)

            # Setup the initial target pose
            q0 = self.pose_at_frame(frame_index, isRef=True)
            q1 = q0 + self.swing_thigh_offset(t)

            if win_index >= len(self.sequence):
                return q1
            s = self.sequence[win_index]
            q2 = q1 + s.delta()
            return q2
        else:
            s = self.current_sample
            t = float(frame_index) * self.h
            q0 = self.pose_at_frame(frame_index, isRef=True)
            q = q0 + s.delta() + self.swing_thigh_offset(t)
            # print '>', self.sim.get_time(), win_frame, t, win.t0
            return q
