import numpy as np
from numpy.linalg import norm
import cma
import logging
from parameterized_motion import ParameterizedMotion
from pydart import SkelVector
import threading


def optimizer_worker(motion):
    motion.solve()


class Window(object):
    def __init__(self, motion, t0, dt):
        self.logger = logging.getLogger(__name__)
        self.motion = motion
        self.skel = self.motion.skel
        self.t0 = t0
        self.dt = dt

        # Set weights
        ndofs = self.skel.ndofs
        w = SkelVector(0.3 * np.ones(ndofs), self.skel)
        w[:6] = 0
        for i in range(ndofs):
            name = self.skel.dofs[i].name
            if ('bicep' in name) or ('forearm' in name) or ('hand' in name):
                w[i] = 0
            if ('thigh' in name):
                w[i] = 1.0
            if ('shin' in name):
                w[i] = 2.0
        self.w = w

        # Set temporary initial state
        self.init_state = None
        self.final_state = None

        # Set parameters
        p0 = np.random.rand(self.num_params()) - 0.5
        self.set_params(p0)

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

    def optimize(self, sim):
        self.sim = sim
        self.eval_counter = 0

        opts = cma.CMAOptions()
        opts.set('verb_disp', 1)
        opts.set('ftarget', 1.0)
        opts.set('popsize', 32)
        opts.set('maxiter', 200)
        self.opts = opts
        x0 = self.params
        self.logger.info('x0: %s' % x0)
        self.logger.info('------------------ CMA-ES ------------------')
        if self.t0 < -0.399:
            self.logger.info('skip the optimization')
            self.logger.info('--------------------------------------------')
        else:
            res = cma.fmin(self.cost, x0, 0.2, opts)
            self.logger.info('--------------------------------------------')
            self.logger.info('solution: %s' % repr(res[0]))
            self.logger.info('value: %.6f' % res[1])
            self.logger.info('--------------------------------------------')
            self.set_params(res[0])
        self.cost(self.params, write_final=True)

    def index0(self):
        t = self.t0
        world = self.sim.world
        h = world.time_step()
        frame = int(t / h)
        return frame

    def cost(self, x, write_final=False):
        self.eval_counter += 1
        skel = self.sim.skel
        world = self.sim.world
        self.sim.reset()
        self.sim.begin_time = self.t0
        world.x = self.init_state

        # Set the parameters
        self.set_params(x)

        v = dict()
        for key in ['x', 'q', 'C', 'Cd', 'FL', 'FR', 'H']:
            v[key] = 0.0

        v['x'] = 0.5 * norm(x) ** 2

        # Use a local time t
        t = self.t0
        h = world.time_step()
        frame = self.index0()
        while t < self.t0 + self.dt:
            self.sim.step()

            # Measure v['q']
            q = skel.q
            qhat = self.motion.ref_pose_at_frame(frame)
            q_diff = q - qhat
            # q_diff[:6] = 0.0
            v['q'] += (2.0 / (skel.ndofs)) * 0.5 * norm(q_diff) ** 2

            # Measure v['C']
            C = skel.C
            Chat = self.motion.ref_com_at_frame(frame)
            v['C'] += 0.5 * norm(C - Chat) ** 2

            # Measure v['Cd']
            v['Cd'] += 5.0 * 0.5 * (skel.Cdot[2] ** 2)

            # Measure v['FL']
            w_fl = 100.0 if (0.6 <= t <= 1.6) else 2.0
            FL = skel.body('h_toe_left').C
            FLhat = self.motion.ref_lfoot_at_frame(frame)
            v['FL'] += w_fl * 0.5 * norm(FL - FLhat) ** 2
            if (0.4 <= t <= 0.6) and FL[1] < FLhat[1]:
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
        if write_final:
            self.final_state = world.x

        ret = 0.0
        self.logger.info('')
        self.logger.info('time window starting at %.2f/ %d' %
                         (self.t0, self.eval_counter))
        for key, val in v.iteritems():
            self.logger.info('  > %s --> %.4f' % (key, val))
            ret += val
        self.logger.info('total --> %.4f' % ret)
        return ret


class WindowedMotion(ParameterizedMotion):
    def __init__(self, skel, ref, stair):
        super(WindowedMotion, self).__init__(skel, ref)
        self.set_stair_info(stair)
        self.logger = logging.getLogger(__name__)

        # Build windows
        T = 1.6
        dt = 0.2
        self.dt = dt
        self.windows = []
        for t0 in np.arange(0.0, T, dt):
            win = Window(self, t0, dt)
            self.windows.append(win)

        # Set parameter
        # p0 = 0.1 * (np.random.rand(self.num_params()) - 0.5)
        p0 = np.zeros(self.num_params())
        self.set_params(p0)

        # Print info
        self.logger.info('dim = %s' % self.num_params())
        self.logger.info('# windows = %s' % len(self.windows))
        self.current_win = None

    def set_stair_info(self, stair):
        self.step_duration = stair.step_duration

    def num_params(self):
        return sum([w.num_params() for w in self.windows])

    def set_params(self, params):
        self.params = params
        win_num_params = [w.num_params() for w in self.windows]
        cum_num_params = np.cumsum(win_num_params)
        splitted = np.split(params, cum_num_params)
        for w, p in zip(self.windows, splitted):
            print 'set params:', w.t0, p
            w.set_params(p)

    def reconstruct_params(self):
        win_num_params = [w.num_params() for w in self.windows]
        cum_num_params = [0] + list(np.cumsum(win_num_params))
        for i in range(len(self.windows)):
            win = self.windows[i]
            i0 = cum_num_params[i]
            i1 = cum_num_params[i + 1]
            self.params[i0:i1] = win.params

    def launch(self, sim):
        self.logger.info('launch the solver as the thread')
        self.sim = sim
        self.thread = threading.Thread(target=optimizer_worker, args=(self,))

        self.logger.info('thread initialized')
        self.thread.start()
        self.logger.info('thread started')

    def solve(self):
        init_state = self.sim.world.x
        for i, win in enumerate(self.windows):
            self.logger.info('Optimizing the window #%d' % i)
            self.logger.info('  time starts from %.4f' % win.t0)
            self.logger.info('  params %s' % win.params)

            self.current_win = win
            win.init_state = init_state
            win.optimize(self.sim)
            init_state = win.final_state
        self.current_win = None
        for i, win in enumerate(self.windows):
            self.logger.info('%d at %.2f' % (i, win.t0))
            self.logger.info('  solution: %s' % (repr(win.params)))
            self.logger.info('  final: %s' % (win.final_state))
        self.sim.begin_time = 0.0
        self.reconstruct_params()
        self.logger.info('params: %s' % self.params)

    def swing_thigh_offset(self, t):
        step_duration = 0.8
        phase_t = t % step_duration
        ret = SkelVector(np.zeros(self.skel.ndofs), self.skel)
        if not (0.1 <= phase_t <= 0.5):
            return ret
        step_counter = int(t / step_duration)
        swing = 'left' if step_counter % 2 == 0 else 'right'
        ret['j_thigh_%s_z' % swing] += 0.2
        return ret

    def parameterized_pose_at_frame(self, frame_index):
        if self.current_win is None:
            t = float(frame_index) * self.h
            win_index = int(t / self.dt)
            if win_index >= len(self.windows):
                return self.pose_at_frame(frame_index, isRef=True)
            win = self.windows[win_index]
            # print self.sim.get_time(), frame_index, t, win.t0
            q0 = self.pose_at_frame(frame_index, isRef=True)
            q = q0 + win.delta() + self.swing_thigh_offset(t)
            return q
        else:
            win = self.current_win
            t = float(frame_index) * self.h
            q0 = self.pose_at_frame(frame_index, isRef=True)
            q = q0 + win.delta() + self.swing_thigh_offset(t)
            # print '>', self.sim.get_time(), win_frame, t, win.t0
            return q
