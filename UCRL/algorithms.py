"""
Note that this implementation is modified from the UCRL implementation
by Ronan Fruit that can be found at: https://github.com/RonanFR/UCRL
"""

from .cython.max_proba import maxProba
from .cython.ExtendedValueIteration import extended_value_iteration
from .evi.evi import EVI
from .logging import default_logger
from . import bounds as bounds

import numbers
import math as m
import numpy as np
import time

class EVIException(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class AbstractUCRL(object):

    def __init__(self, env,
                 r_max, random_state,
                 alpha_r=None, alpha_p=None,
                 solver=None,
                 verbose = 0,
                 logger=default_logger,
                 bound_type_p="bernstein",
                 bound_type_rew="bernstein"):
        self.env = env
        self.r_max = float(r_max)

        if alpha_r is None:
            self.alpha_r = 1
        else:
            self.alpha_r = alpha_r
        if alpha_p is None:
            self.alpha_p = 1
        else:
            self.alpha_p = alpha_p

        if solver is None:
            # create solver for optimistic model
            self.opt_solver = EVI(nb_states=self.env.num_poses,
                                  actions_per_state=[range(self.env.num_arms) for _ in range(self.env.num_poses)],
                                  bound_type= bound_type_p,
                                  random_state = random_state,
                                  gamma=1.
                                  )
        else:
            self.opt_solver = solver

        # initialize matrices
        self.policy = np.zeros((self.env.num_poses,), dtype=np.int_)
        self.policy_indices = np.zeros((self.env.num_poses,), dtype=np.int_)

        # initialization
        self.total_reward = 0
        self.total_time = 0
        self.regret = [0]  # cumulative regret of the learning algorithm
        self.regret_unit_time = [0]
        self.unit_duration = [1]  # ratios (nb of time steps)/(nb of decision steps)
        self.span_values = []
        self.span_times = []
        self.iteration = 0
        self.episode = 0
        self.delta = 1.  # confidence
        self.bound_type_p = bound_type_p
        self.bound_type_rew = bound_type_rew

        self.verbose = verbose
        self.logger = logger
        self.random_state = random_state
        self.local_random = np.random.RandomState(seed=random_state)

    def clear_before_pickle(self):
        del self.opt_solver

    def description(self):
        desc = {
            "alpha_p": self.alpha_p,
            "alpha_r": self.alpha_r,
            "r_max": self.r_max,
        }
        return desc

class UcrlMdp(AbstractUCRL):
    """
    Implementation of Upper Confidence Reinforcement Learning (UCRL) algorithm for toys state/actions MDPs with
    positive bounded rewards.
    """

    def __init__(self, env, r_max=1, alpha_r=None, alpha_p=None, solver=None,
                 bound_type_p="chernoff", bound_type_rew="chernoff", verbose = 0,
                 logger=default_logger, random_state=None):
        """
        :param environment: an instance of any subclass of abstract class Environment which is an MDP
        :param r_max: upper bound
        :param alpha_r: multiplicative factor for the concentration bound on rewards (default is r_max)
        :param alpha_p: multiplicative factor for the concentration bound on transition probabilities (default is 1)
        """

        assert bound_type_p in ["chernoff",  "bernstein"]
        assert bound_type_rew in ["chernoff",  "bernstein"]

        super(UcrlMdp, self).__init__(env=env,
                                      r_max=r_max, alpha_r=alpha_r,
                                      alpha_p=alpha_p, solver=solver,
                                      verbose=verbose,
                                      logger=logger, bound_type_p=bound_type_p,
                                      bound_type_rew=bound_type_rew,
                                      random_state=random_state)
        num_poses = self.env.num_poses
        num_arms = self.env.num_arms
        self.P_counter = np.zeros((num_poses, num_arms, num_poses), dtype=np.int64)
        self.P = np.ones((num_poses, num_arms, num_poses)) / num_poses
        self.visited_sa = set()
        self.estimated_rewards = np.ones((num_poses, num_arms))# env.arm_means

        self.variance_proxy_reward = np.zeros((num_poses, num_arms))
        self.estimated_holding_times = np.ones((num_poses, num_arms))

        self.nb_observations = np.zeros((num_poses, num_arms), dtype=np.int64)
        self.nu_k = np.zeros((num_poses, num_arms), dtype=np.int64)
        self.tau = 0.9
        self.tau_max = 1
        self.tau_min = 1
        self.rewards = []
        self.poses = []


    def set_prior(self, prior_means):
        self.estimated_rewards = prior_means # TODO: figure out how to use S...

    def learn(self, duration, regret_time_step, render=False):
        """ Run UCRL on the provided environment
        Args:
            duration (int): the algorithm is run until the number of time steps
                            exceeds "duration"
            regret_time_step (int): the value of the cumulative regret is stored
                                    every "regret_time_step" time steps
            render (flag): True for rendering the domain, False otherwise
        """
        if self.total_time >= duration:
            return
        threshold = self.total_time + regret_time_step
        threshold_span = threshold

        self.solver_times = []
        self.simulation_times = []

        t_star_all = time.perf_counter()
        self.env.reset()
        curr_state = self.env.pose
        while self.total_time < duration:
            self.episode += 1

            # initialize the episode
            self.nu_k.fill(0)
            self.delta = 1 / m.sqrt(self.iteration + 1)

            if self.verbose > 0:
                self.logger.info("{}/{} = {:3.2f}%".format(self.total_time, duration, self.total_time / duration *100))

            # solve the optimistic (extended) model
            t0 = time.time()
            span_value = self.solve_optimistic_model()
            t1 = time.time()

            span_value *= self.tau / self.r_max
            if self.verbose:
                self.logger.info("span({}): {:.9f}".format(self.episode, span_value))
                self.logger.info("evi time: {:.4f} s".format(t1-t0))

            if self.total_time > threshold_span:
                self.span_values.append(span_value)
                self.span_times.append(self.total_time)
                threshold_span = self.total_time + regret_time_step

            # execute the recovered policy
            t0 = time.perf_counter()
            self.visited_sa.clear()
            curr_act = self.sample_action(curr_state)  # sample action from the policy

            while np.sum(self.nu_k/(max(1, self.nb_observations[curr_state][curr_act]))) <= 1 \
                    and self.total_time < duration:
                self.update(curr_state=curr_state, curr_act=curr_act)
                if self.total_time > threshold:
                    self.regret_unit_time.append(self.total_time)
                    self.unit_duration.append(self.total_time/self.iteration)
                    threshold = self.total_time + regret_time_step

                # sample a new action
                curr_state = self.env.pose
                curr_act = self.sample_action(curr_state)  # sample action from the policy

            self.nb_observations += self.nu_k

            for (s,a) in self.visited_sa:
                self.P[s,a] = self.P_counter[s,a] / self.nb_observations[s,a]

            t1 = time.perf_counter()
            self.simulation_times.append(t1-t0)
            if self.verbose > 0:
                self.logger.info("expl time: {:.4f} s".format(t1-t0))

        t_end_all = time.perf_counter()
        self.speed = t_end_all - t_star_all
        if self.verbose: 
            self.logger.info("TIME: %.5f s" % self.speed)

        return self.rewards, self.poses

    def beta_r(self):
        """ Confidence bounds on the reward
        Returns:
            np.array: the vector of confidence bounds on the reward function (|S| x |A|)
        """
        S = self.env.num_poses
        A = self.env.num_arms
        if self.bound_type_rew != "bernstein":
            ci = bounds.chernoff(it=self.iteration, N=self.nb_observations,
                                 range=self.r_max, delta=self.delta,
                                 sqrt_C=3.5, log_C=2 * S * A)
            return self.alpha_r * ci
        else:
            N = np.maximum(1, self.nb_observations)
            Nm1 = np.maximum(1, self.nb_observations - 1)
            var_r = self.variance_proxy_reward / Nm1
            log_value = 2.0 * S * A * (self.iteration + 1) / self.delta
            beta = bounds.bernstein2(scale_a=14 * var_r / N,
                                     log_scale_a=log_value,
                                     scale_b=49.0 * self.r_max / (3.0 * Nm1),
                                     log_scale_b=log_value,
                                     alpha_1=m.sqrt(self.alpha_r), alpha_2=self.alpha_r)
            return beta

    def beta_tau(self):
        """ Confidence bounds on holding times
        Returns:
            np.array: the vecor of confidence bounds on the holding times (|S| x |A|)
        """
        return np.zeros((self.env.num_poses, self.env.num_arms))

    def beta_p(self):
        """ Confidence bounds on transition probabilities
        Returns:
            np.array: the vector of confidence bounds on the transition matrix (|S| x |A|)
        """
        S = self.env.num_poses
        A = self.env.num_arms
        if self.bound_type_p != "bernstein":
            beta = bounds.chernoff(it=self.iteration, N=self.nb_observations,
                                   range=1., delta=self.delta,
                                   sqrt_C=14*S, log_C=2*A)
            return self.alpha_p * beta.reshape([S, A, 1])
        else:
            N = np.maximum(1, self.nb_observations)
            Nm1 = np.maximum(1, self.nb_observations - 1)
            var_p = self.P * (1. - self.P)
            log_value = 2.0 * S * A * (self.iteration + 1) / self.delta
            beta = bounds.bernstein2(scale_a=14 * var_p / N[:, :, np.newaxis],
                                     log_scale_a=log_value,
                                     scale_b=49.0 / (3.0 * Nm1[:, :, np.newaxis]),
                                     log_scale_b=log_value,
                                     alpha_1=m.sqrt(self.alpha_p), alpha_2=self.alpha_p)

            return beta

    def update(self, curr_state, curr_act, **kwargs):
        # execute the action
        s = self.env.pose
        r = self.env.step(curr_act)
        self.rewards.append(r)
        self.poses.append(s)
        s2 = self.env.pose  # new state
        t = 1

        # updated observations
        scale_f = self.nb_observations[s][curr_act] + self.nu_k[s][curr_act]

        # update reward and variance estimate
        old_estimated_reward = self.estimated_rewards[s, curr_act]
        self.estimated_rewards[s, curr_act] *= scale_f / (scale_f + 1.)
        self.estimated_rewards[s, curr_act] += r / (scale_f + 1.)
        self.variance_proxy_reward[s, curr_act] += (r - old_estimated_reward) * (r - self.estimated_rewards[s, curr_act])

        self.P_counter[s, curr_act, s2] += 1
        self.visited_sa.add((s,curr_act))
        self.nu_k[s][curr_act] += 1
        self.total_reward += r
        self.total_time += t
        self.iteration += 1

    def sample_action(self, s):
        """
        Args:
            s (int): a given state index
        Returns:
            action_idx (int): index of the selected action
            action (int): selected action
        """
        if len(self.policy.shape) > 1: # Not supported for now?
            # this is a stochastic policy
            print("Stochastic policy unsupported")
            assert(False)
            action_idx = self.local_random.choice(self.policy_indices[s], p=self.policy[s])
            action = self.env.state_actions[s][action_idx]
        else:
            action = self.policy[s]

        return action

    def solve_optimistic_model(self):

        beta_r = self.beta_r()  # confidence bounds on rewards
        beta_tau = self.beta_tau()  # confidence bounds on holding times
        beta_p = self.beta_p()  # confidence bounds on transition probabilities

        t0 = time.perf_counter()
        span_value_new = self.opt_solver.run(
            self.policy_indices, self.policy,
            self.P, #self.estimated_probabilities,
            self.estimated_rewards,
            self.estimated_holding_times,
            beta_r, beta_p, beta_tau, self.tau_max,
            self.r_max, self.tau, self.tau_min,
            self.r_max / m.sqrt(self.iteration + 1)
        )
        t1 = time.perf_counter()
        tn = t1 - t0
        self.solver_times.append(tn)
        if self.verbose > 1:
            self.logger.info("[%d]NEW EVI: %.3f seconds" % (self.episode, tn))
            if self.verbose > 2:
                self.logger.info(self.policy_indices)

        span_value = span_value_new ##############

        if span_value < 0:
            raise EVIException(error_value=span_value)

        if self.verbose > 1:
            print("{:.2f} / {:.2f}".format(span_value, span_value_new))
        assert np.abs(span_value - span_value_new) < 1e-8

        return span_value


    def extended_value_iteration(self, beta_r, beta_p, beta_tau, epsilon):
        """
        Use compiled .so file for improved speed.
        :param beta_r: confidence bounds on rewards
        :param beta_p: confidence bounds on transition probabilities
        :param beta_tau: confidence bounds on holding times
        :param epsilon: desired accuracy
        """
        u1 = np.zeros(self.env.num_poses)
        sorted_indices = np.arange(self.env.num_poses)
        u2 = np.zeros(self.env.num_poses)
        p = self.estimated_probabilities
        counter = 0
        while True:
            counter += 1
            for s in range(self.env.num_poses):
                first_action = True
                for c in range(self.env.num_arms):
                    vec = self.max_proba(p[s][c], sorted_indices, beta_p[s][c])  # python implementation: slow
                    vec[s] -= 1
                    r_optimal = min(self.tau_max*self.r_max,
                                    self.estimated_rewards[s][c] + beta_r[s][c])
                    v = r_optimal + np.dot(vec, u1) * self.tau
                    tau_optimal = min(self.tau_max, max(max(self.tau_min, r_optimal/self.r_max),
                                  self.estimated_holding_times[s][c] - np.sign(v) * beta_tau[s][c]))
                    if first_action or v/tau_optimal + u1[s] > u2[s] or m.isclose(v/tau_optimal + u1[s], u2[s]):  # optimal policy = argmax
                        u2[s] = v/tau_optimal + u1[s]
                        self.policy_indices[s] = c
                        self.policy[s] = c
                    first_action = False
            if max(u2-u1)-min(u2-u1) < epsilon:  # stopping condition of EVI
                print("---{}".format(counter))
                return max(u1) - min(u1), u1, u2
            else:
                u1 = u2
                u2 = np.empty(self.env.num_poses)
                sorted_indices = np.argsort(u1)

    def max_proba(self, p, sorted_indices, beta):
        """
        Use compiled .so file for improved speed.
        :param p: probability distribution with toys support
        :param sorted_indices: argsort of value function
        :param beta: confidence bound on the empirical probability
        :return: optimal probability
        """
        n = np.size(sorted_indices)
        min1 = min(1, p[sorted_indices[n-1]] + beta/2)
        if min1 == 1:
            p2 = np.zeros(self.env.num_poses)
            p2[sorted_indices[n-1]] = 1
        else:
            sorted_p = p[sorted_indices]
            support_sorted_p = np.nonzero(sorted_p)[0]
            restricted_sorted_p = sorted_p[support_sorted_p]
            support_p = sorted_indices[support_sorted_p]
            p2 = np.zeros(self.env.num_poses)
            p2[support_p] = restricted_sorted_p
            p2[sorted_indices[n-1]] = min1
            s = 1 - p[sorted_indices[n-1]] + min1
            s2 = s
            for i, proba in enumerate(restricted_sorted_p):
                max1 = max(0, 1 - s + proba)
                s2 += (max1 - proba)
                p2[support_p[i]] = max1
                s = s2
                if s <= 1: break
        return p2
