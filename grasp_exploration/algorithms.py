"""
Grasp Exploration Algorithm implementations from the paper.
UCB, Thompson Sampling, and GQCNN policy implementations here.
The abstract class defines the necessary methods for an exploration
algorithm.
"""


import abc
import numpy as np


class ExplorationAlgorithm(abc.ABC):

    def __init__(self, num_arms):
        """
        An abstract exploration algorithm class.

        Parameters
        ------------
        num_arms : tuple of ints
            Number of poses by number of arms.
        """
        self.num_arms = num_arms

    @abc.abstractmethod
    def select_arm(self, ind):
        pass

    def update(self, ind, arm, reward):
        pass

    def set_prior(self, prior):
        pass

    def reset(self):
        pass


class OptimalPolicy(ExplorationAlgorithm):

    def __init__(self, num_arms, env):
        super(OptimalPolicy, self).__init__(num_arms)
        self.best_arms = env.arm_means.argmax(axis=1)
    
    def select_arm(self, ind):
        return self.best_arms[ind]


class UCB(ExplorationAlgorithm):

    def __init__(self, num_arms, strength):
        super(UCB, self).__init__(num_arms)
        self.strength = strength
        self.counts = np.zeros(self.num_arms)
        self.values = np.zeros(self.num_arms)

    # Selects an arm from counts and values
    def select_arm(self, ind):

        counts = self.counts[ind]
        values = self.values[ind]

        # Select arm with classic UCB1 if all counts > 0
        if np.all(counts):
            bonuses = np.sqrt((2 * np.log(counts.sum())) / counts)
            arm = np.argmax(values + bonuses)

        # Otherwise select an arm that hasn't been chosen
        else:
            arm = np.random.choice(np.where(counts == 0)[0])

        return arm

    # Update counts
    def update(self, ind, arm, reward):
        self.counts[ind, arm] += 1
        n = self.counts[ind, arm]
        v = self.values[ind, arm]
        self.values[ind, arm] = (n - 1) / n * v + (1 / n) * reward

    # Sets prior pseudo-counts for arms
    def set_prior(self, prior):

        # Only set prior strength if S is positive
        if self.strength > 0:

            # Check shape of prior and strength
            if not np.isscalar(prior) and prior.shape != self.values.shape:
                raise ValueError('Prior must be scalar or {}'.format(self.values.shape))

            if not np.isscalar(S):
                raise ValueError('Strength must be scalar')

            self.counts = self.strength * np.ones(self.num_arms)
            self.values = prior * np.ones(self.num_arms)

    def reset(self):
        self.counts = np.zeros(self.num_arms)
        self.values = np.zeros(self.num_arms)


class TS(ExplorationAlgorithm):

    def __init__(self, num_arms, strength):

        super(TS, self).__init__(num_arms)
        self.strength = strength
        self.alphas = np.ones(self.num_arms)
        self.betas = np.ones(self.num_arms)
        self.mean_probs = 0.5 * np.ones(self.num_arms)

    # Selects an arm via Thompson Sampling
    def select_arm(self, ind):
        return np.argmax(np.random.beta(self.alphas[ind], self.betas[ind]))

    # Updates counts
    def update(self, ind, arm, reward):
        self.alphas[ind, arm] += reward
        self.betas[ind, arm] += (1 - reward)
        self.mean_probs[ind, arm] = (self.alphas[ind, arm] /
                                        (self.alphas[ind, arm] +
                                        self.betas[ind, arm]))

    # Sets prior given strength for Thompson sampling
    def set_prior(self, prior, delta=0.01):

        if self.strength > 0:
            # Check shape of prior and strength
            if not np.isscalar(prior) and prior.shape != self.alphas.shape:
                raise ValueError('Prior must be scalar or {}'.format(self.alphas.shape))

            # if not np.isscalar(S) and S.shape != self.alphas.shape:
            #     raise ValueError('Strength must be scalar or {}'.format(self.alphas.shape))

            prior = np.clip(prior, delta, 1-delta)
            self.alphas = self.strength * prior
            self.betas = self.strength * (1 - prior)

    def reset(self):
        self.alphas = np.ones(self.num_arms)
        self.betas = np.ones(self.num_arms)
        self.mean_probs = 0.5 * np.ones(self.num_arms)


class GQCNNPolicy(ExplorationAlgorithm):

    def __init__(self, num_arms, eps_greedy=0.1):

        super(GQCNNPolicy, self).__init__(num_arms)
        self.eps_greedy = eps_greedy
        self.q_values = np.zeros(self.num_arms)

    def set_prior(self, prior):
        self.q_values = prior

    # Just choose arm with the highest q_value
    def select_arm(self, ind):
        if np.random.random() < self.eps_greedy:
            return np.random.randint(0, self.q_values.shape[-1])
        else:
            return np.argmax(self.q_values[ind])
