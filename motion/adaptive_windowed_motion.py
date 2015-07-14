import numpy as np
from numpy.linalg import norm
import math
import logging
from parameterized_motion import ParameterizedMotion
from pydart import SkelVector
import threading


def optimizer_worker(motion):
    motion.solve()


class Sample(object):
    def __init__(self, motion, t0, dt):
        self.motion = motion
        self.skel = self.motion.skel
        self.t0 = t0
        self.dt = dt
        self.curr_score = None

    def set_params(self, params):
        self.params = params

    def num_params(self):
        return self._num_params

    @property
    def i0(self):
        t = self.t0
        world = self.motion.sim.world
        h = world.time_step()
        frame = int(t / h)
        return frame

    def simulate(self, init_state, task):
        # print 'init_state = ', init_state
        self.motion.current_sample = self
        # Fetch variables
        sim = self.motion.sim
        world = sim.world
        stair = sim.stair

        # Reset the simulation
        sim.begin_time = self.t0
        stair.set_activation(task)
        sim.reset()
        world.x = init_state

        # Initialize the value dictionary
        v = self.v = dict()
        for key in ['x', 'q', 'qdot', 'C', 'Cd', 'FL', 'FR', 'H']:
            v[key] = 0.0
        v['x'] = 0.5 * norm(self.params) ** 2

        # Initiliaze the time variables
        t = self.t0
        frame = self.i0
        h = world.time_step()
        # During the time_window
        while t < self.t0 + self.dt:
            # Step
            # print '---', t, frame, self.skel.C, self.skel.Cdot
            sim.step()
            # Evaluate
            self.evaluate_frame(t, frame)
            # Increase the time and frame index
            t += h
            frame += 1
        # Update the current score
        sim_score = 0.0
        for key, val in v.iteritems():
            sim_score += val
        # Build debug info
        info = dict()
        skel = self.skel
        info['C'] = skel.C
        info['Cd'] = skel.Cdot

        # Return the score and the final state
        # print 'final_state = ', world.x
        return (sim_score, world.x, info)

    def evaluate_frame(self, t, frame):
        v = self.v
        skel = self.skel
        step_duration = 0.8
        phase_t = t % step_duration

        # Measure v['q']
        q = skel.q
        qhat = self.motion.ref_pose_at_frame(frame)
        q_diff = q - qhat
        # q_diff[:6] = 0.0
        v['q'] += (2.0 / (skel.ndofs)) * 0.5 * norm(q_diff) ** 2

        # Measure v['qdot']
        qdot = skel.qdot
        qdot_hat = self.motion.ref_velocity_at_frame(frame)
        qdot_diff = qdot - qdot_hat
        v['qdot'] += (0.02 / (skel.ndofs)) * 0.5 * norm(qdot_diff) ** 2

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
        v['Cd'] += 3.0 * 0.5 * (Cd[1] ** 2)

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
        # if (1.0 <= t <= 1.2) and (0.16 + 0.295) < FR[0]:
        #     v['FR'] += 1.0

        # Measure v['H']
        H = skel.body('h_head').C
        Hhat = self.motion.ref_head_at_frame(frame)
        v['H'] += 0.5 * norm(H - Hhat) ** 2

    def log_values(self, logger):
        v = self.v
        for key, val in v.iteritems():
            logger.info('  > %s --> %.4f' % (key, val))
        if hasattr(self, 'info'):
            for key, val in self.info.iteritems():
                logger.info('  [info] %s --> %s' % (str(key), str(val)))


class FFSample(Sample):
    def __init__(self, motion, t0, dt, prev_fb_sample, task):
        super(FFSample, self).__init__(motion, t0, dt)
        self.logger = logging.getLogger(__name__)
        self.prev_fb_sample = prev_fb_sample
        self.task = task
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
        self._num_params = np.count_nonzero(self.w)

    def delta(self, t):
        """ Ignore time t - only for FBSample """
        ndofs = self.skel.ndofs
        ret = np.zeros(ndofs)
        j = 0
        for i in range(ndofs):
            if self.w[i] < 0.0001:
                continue
            else:
                ret[i] += self.w[i] * self.params[j]
                j += 1

        T = 0.8
        step_counter = int(t / T)
        # Determine swing and stance foot
        swing = 'left' if step_counter % 2 == 0 else 'right'
        ret = SkelVector(ret, self.skel)
        # Adjust the lateral swing heel
        swFT = self.skel.body('h_toe_%s' % swing).T
        (ax, ay, az) = self.mat2euler(swFT)
        ret['j_heel_%s_2' % swing] += -5.0 * az
        return ret

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

    def prev_final_state(self):
        if self.prev_fb_sample is None:
            return self.motion.init_state
        return self.prev_fb_sample.task_final_state(self.task)

    def prev_score(self):
        if self.prev_fb_sample is None:
            return 0.0
        return self.prev_fb_sample.task_score(self.task)

    def score(self):
        return self.prev_score() + self.curr_score

    def evaluate(self):
        init_state = self.prev_final_state()
        (sim_score, final_state, info) = self.simulate(init_state, self.task)
        self.curr_score = sim_score
        self.final_state = final_state
        self.info = info
        self.log_values(self.logger)
        self.logger.info('    score : %.4f' %
                         self.score())
        # # Repeat for debuging
        # (sim_score, final_state, info) = self.simulate(init_state, self.task)
        # self.curr_score = sim_score
        # self.final_state = final_state
        # self.info = info
        # self.log_values(self.logger)
        # self.logger.info('    score : %.4f' %
        #                  self.score())
        # # Repeat for debuging
        # (sim_score, final_state, info) = self.simulate(init_state, self.task)
        # self.curr_score = sim_score
        # self.final_state = final_state
        # self.info = info
        # self.log_values(self.logger)
        # self.logger.info('    score : %.4f' %
        #                  self.score())


class FBSample(Sample):
    def __init__(self, motion, t0, dt, prev_ff_sample):
        super(FBSample, self).__init__(motion, t0, dt)
        self.logger = logging.getLogger(__name__)
        self.prev_ff_sample = prev_ff_sample
        self.tasks = self.motion.tasks
        self.init_weights()

        self.results = dict()
        for task in self.tasks:
            result = dict()
            result['final_state'] = None
            result['curr_score'] = None
            self.results[task] = result

    def init_weights(self):
        self._num_params = 8

    def num_all_params(self):
        return 1 + self.prev_ff_sample.num_params() + self.num_params()

    def get_all_params(self):
        prev_ff = self.prev_ff_sample
        ret = np.concatenate([[prev_ff.task], prev_ff.params, self.params])
        return ret

    def set_all_params(self, params):
        n0 = 1
        n1 = n0 + self.prev_ff_sample.num_params()
        n2 = n1 + self.num_params()
        splitted = np.split(params, [n0, n1, n2])

        prev_ff = self.prev_ff_sample
        prev_ff.task = splitted[0][0]
        prev_ff.set_params(splitted[1])
        self.set_params(splitted[2])

    def delta(self, t):
        ff_delta = self.prev_ff_sample.delta(t)
        ret = SkelVector(ff_delta, self.skel)
        T = 0.8
        step_counter = int(t / T)
        # Determine swing and stance foot
        swing = 'left' if step_counter % 2 == 0 else 'right'
        stance = 'right' if step_counter % 2 == 0 else 'left'
        flip = 1.0 if step_counter % 2 == 0 else -1.0

        # Iterator for parameters
        i = np.nditer(self.params)

        # Fetch skel info
        Cd = self.skel.Cdot

        # Adjust the stance heel w.r.t. forward momentum
        (a, b) = (i.next(), i.next())
        st_heel_offset = a - b * (Cd[0] - 0.38)
        ret['j_heel_%s_1' % stance] += st_heel_offset

        # Adjust the stance thigh w.r.t. forward momentum
        (a, b) = (i.next(), i.next())
        st_thigh_offset = 0.1 * (a - b * (Cd[0] - 0.38))
        ret['j_thigh_%s_z' % stance] += st_thigh_offset

        # Adjust the stance heel w.r.t. lateral momentum
        CdZ = Cd[2] * flip
        # print phase_t, CdZ
        if CdZ < -0.20:
            (a, b) = (i.next(), i.next())
            st_heel2_offset = a - b * (CdZ + 0.20)
            st_heel2_offset *= flip
            ret['j_heel_%s_2' % stance] -= st_heel2_offset

        # Adjust the swing thigh w.r.t. forward momentum
        (a, b) = (i.next(), i.next())
        sw_thigh_offset = 0.1 * (a - b * (Cd[0] - 0.38))
        ret['j_thigh_%s_z' % swing] += sw_thigh_offset

        return ret

    def prev_fb_sample(self):
        if self.prev_ff_sample is None:
            return None
        return self.prev_ff_sample.prev_fb_sample

    def task_prev_final_state(self, task):
        prev_fb = self.prev_fb_sample()
        if prev_fb is None:
            return self.motion.init_state
        return prev_fb.task_final_state(task)

    def task_final_state(self, task):
        return self.results[task]['final_state']

    def task_prev_score(self, task):
        prev_fb = self.prev_fb_sample()
        if prev_fb is None:
            return 0.0
        return prev_fb.task_score(task)

    def task_score(self, task):
        return self.results[task]['curr_score'] + self.task_prev_score(task)

    def prev_score(self):
        prev_task_scores = [self.task_prev_score(t) for t in self.tasks]
        return np.mean(prev_task_scores)

    def score(self):
        task_scores = [self.task_score(t) for t in self.tasks]
        return np.mean(task_scores)

    def evaluate(self):
        for task in self.tasks:
            result = self.results[task]
            init_state = self.task_prev_final_state(task)
            (sim_score, final_state, info) = self.simulate(init_state, task)
            self.info = info
            result['curr_score'] = sim_score
            result['final_state'] = final_state
            self.logger.info('[Task]: %.2f' % task)
            self.log_values(self.logger)
            self.logger.info('    task_prev : %.4f' %
                             self.task_prev_score(task))
            self.logger.info('    curr : %.4f' % sim_score)
            self.logger.info('    task_score : %.4f' %
                             self.task_score(task))
        self.logger.info('    score : %.4f' %
                         self.score())


class AdaptiveWindowedMotion(ParameterizedMotion):
    def __init__(self, skel, ref, stair):
        super(AdaptiveWindowedMotion, self).__init__(skel, ref)
        self.set_stair_info(stair)
        self.logger = logging.getLogger(__name__)
        self.sequence = []
        self.T = 0.8
        self.dt = 0.2
        # self.tasks = [0.0, 0.2]
        self.tasks = [0.0, 0.1, 0.2]
        self.current_sample = None
        self.prev_index = -1
        self.saved_samples = []

        self.logger.info('AdaptiveWindowedMotion is constructed')
        self.logger.info('dim = %s' % self.num_params())

    def set_stair_info(self, stair):
        self.step_duration = stair.step_duration

    def num_params(self):
        return sum([s.num_params() for s in self.sequence])

    def num_all_params(self):
        return sum([s.num_all_params() for s in self.sequence])

    def set_params(self, params):
        self.params = params

        # Reconstruct the samples
        self.sequence = []
        prev_fb = None
        for t0 in np.arange(0.0, self.T, self.dt):
            s_ff = FFSample(self, t0, self.dt, prev_fb, None)
            s_fb = FBSample(self, t0, self.dt, s_ff)
            self.sequence.append(s_fb)
            prev_fb = s_fb

        # Fill the params
        self.logger.info('len(params) = %d' % len(params))
        self.logger.info('num_all_params = %d' % self.num_all_params())
        if len(params) < self.num_all_params():
            n_fill = self.num_all_params() - len(params)
            fill = np.zeros(n_fill)
            params = np.concatenate([params, fill])
            self.params = params
            self.logger.info('# fill = %d' % n_fill)
            self.logger.info('new params = %s' % params)

        sample_num_params = [ss.num_all_params() for ss in self.sequence]
        self.logger.info('sample_num_params = %s' % sample_num_params)
        cum_num_params = np.cumsum(sample_num_params)
        self.logger.info('cum_num_params = %s' % cum_num_params)
        splitted = np.split(params, cum_num_params)
        self.logger.info('splitted = %s' % splitted)
        for s, p in zip(self.sequence, splitted):
            self.logger.info('set params: %f / %s' % (s.t0, p))
            s.set_all_params(p)

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
        n_iter_samples = 4000
        n_saved_samples = 500
        self.saved_samples = []

        for t0 in np.arange(0.0, T, dt):
            # For both feed back and feed forward
            for isFF in [True, False]:
                # 0. Calculate the prev values
                prev_values = self.calculate_prev_values(prev_samples)

                # 1. Generate new samples
                next_samples = []
                curr_min_score = 100000.0
                while len(next_samples) < n_iter_samples:
                    s = self.generate_new_sample(t0,
                                                 dt,
                                                 prev_samples,
                                                 prev_values,
                                                 isFF)
                    self.logger.info('')
                    self.logger.info('>' * 50)
                    phase = 'Feed-Forward' if isFF else 'Feed-Back'
                    self.logger.info('Phase = %s' % phase)
                    self.logger.info('Evaluating a new sample: %d/%d' %
                                     (len(next_samples), n_iter_samples))
                    self.logger.info('  t0/dt = %f/%f' % (s.t0, s.dt))
                    if isFF:
                        self.logger.info('  task = %f' % s.task)
                    else:
                        self.logger.info('  prev_ff_params = %s'
                                         % s.prev_ff_sample.params)
                    self.logger.info('  params = %s' % s.params)
                    print type(s)
                    self.logger.info('  prev score = %f' % s.prev_score())
                    s.evaluate()
                    next_samples.append(s)
                    curr_min_score = min(curr_min_score, s.score())
                    self.logger.info('  curr_min_score = %f' % curr_min_score)

                # 2. Select the good and various samples
                all_samples.append(prev_samples)
                selected_samples = self.select_samples(next_samples,
                                                       n_saved_samples)
                prev_samples = selected_samples

                # 3. Debug prints
                values = [ss.score() for ss in prev_samples]
                self.logger.info('')
                self.logger.info('----------------------------')
                self.logger.info('# saved samples = %d' % len(prev_samples))
                self.logger.info('best cost = %f' % min(values))
                self.logger.info('worst cost = %f' % max(values))
                self.logger.info('----------------------------')
                self.logger.info('')
            self.saved_samples.append(prev_samples)

        for i, iter_samples in enumerate(self.saved_samples):
            self.logger.info('Iter %d: %d samples' % (i, len(iter_samples)))

        n_windows = len(self.saved_samples)
        self.set_solution(n_windows - 1, 0)

        # Reconstruct samples
        self.logger.info('set current_sample as None')
        self.current_sample = None
        self.sim.begin_time = 0.0

    def calculate_prev_values(self, prev_samples):
        if len(prev_samples) > 1:
            prev_values = [100.0 / s.score() for s in prev_samples]
            prev_values = np.array(prev_values) / np.sum(prev_values)
            prev_values = np.array(prev_values) / np.sum(prev_values)
            prev_values = np.array(prev_values) / np.sum(prev_values)
        else:
            prev_values = [1.0]
        return prev_values

    def generate_new_sample(self, t0, dt,
                            prev_samples, prev_values, isFF=True):
        try:
            prev = np.random.choice(prev_samples, p=prev_values)
        except:
            self.logger.error('Sum of prev_values = %.12f' %
                              np.sum(prev_values))
            return
        if isFF:
            task = np.random.choice(self.tasks)
            s = FFSample(self, t0, dt, prev, task)
            dim = s.num_params()
            m = np.zeros(dim)
            C = 0.10 * np.identity(dim)
            params = np.random.multivariate_normal(m, C)
            s.set_params(params)
        else:
            s = FBSample(self, t0, dt, prev)
            dim = s.num_params()
            m = np.zeros(dim)
            C = 0.05 * np.identity(dim)
            params = np.random.multivariate_normal(m, C)
            s.set_params(params)
            # s.set_params(m)
        return s

    def select_samples(self, samples, n_saved_samples):
        samples.sort(key=lambda s: s.score())
        ret = list()
        ret += samples[:n_saved_samples / 2]
        while len(ret) < n_saved_samples:
            s = np.random.choice(samples[n_saved_samples / 2:])
            ret.append(s)
        return ret

    def num_solutions(self):
        nx = len(self.saved_samples)
        if nx == 0:
            return (0, 0)
        else:
            ny = len(self.saved_samples[0])
            return (nx - 1, ny - 1)

    def set_solution(self, x, y):
        opt = self.saved_samples[x][y]
        self.logger.info('reconstructing.....')
        self.logger.info(' x, y = %d, %d' % (x, y))
        self.logger.info(' sample value = %.4f' % opt.score())
        self.sequence = []
        # opt = prev_samples[0]
        while opt is not None:
            self.logger.info('opt sample.score = %f' % opt.score())
            self.sequence = [opt] + self.sequence
            opt = opt.prev_fb_sample()
        self.logger.info('reconstructing..... Done')
        self.logger.info('# sequence = %d' % len(self.sequence))
        self.logger.info('merging parameters..... ')
        # for s in self.sequence:
        #     self.logger.info(repr(s.get_all_params()))
        params = np.concatenate([s.get_all_params() for s in self.sequence])
        self.set_params(params)
        self.logger.info('merging parameters..... done')

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
        ret['j_heel_%s_1' % stance] -= 0.3
        # print t, phase_t, ret
        return ret

    def parameterized_pose_at_frame(self, frame_index):
        # print self.skel.body('h_toe_right').C
        if (frame_index % 200) == 0:
            reset_state = np.array(self.skel.world.x)
            # self.skel.world.reset()
            self.skel.world.x = reset_state
            # print '============ manual reset', frame_index

        if self.current_sample is None:
            # Fetch time information
            t = float(frame_index) * self.h
            win_index = int(t / self.dt)

            # Setup the initial target pose
            q0 = self.pose_at_frame(frame_index, isRef=True)
            q1 = q0 + self.swing_thigh_offset(t)

            if win_index >= len(self.sequence):
                # if (frame_index % 200) < 3:
                #     print '---', t, frame_index, self.skel.C, self.skel.Cdot
                return q1
            s = self.sequence[win_index]
            q2 = q1 + s.delta(t)

            # if (frame_index % 200) < 3:
            #     print '---', t, frame_index, self.skel.C, self.skel.Cdot
            #     print '))', s.params
            #     print '))))', s.prev_ff_sample.task
            #     print '))))', s.prev_ff_sample.params
            #     print 'delta:', s.delta(t)
            #     print 'ret:', q2
            #     print self.sim.world.x
            self.prev_index = frame_index
            return q2
        else:
            s = self.current_sample
            t = float(frame_index) * self.h
            q0 = self.pose_at_frame(frame_index, isRef=True)
            q = q0 + s.delta(t) + self.swing_thigh_offset(t)
            # if (frame_index % 200) < 3:
            #     print '---', t, frame_index, self.skel.C, self.skel.Cdot
            #     print '))', s.params
            #     if isinstance(s, FBSample):
            #         print '))))', s.prev_ff_sample.task
            #         print '))))', s.prev_ff_sample.params
            #     print 'delta:', s.delta(t)
            #     print 'ret:', q
            #     print self.sim.world.x
            self.prev_index = frame_index
            return q
