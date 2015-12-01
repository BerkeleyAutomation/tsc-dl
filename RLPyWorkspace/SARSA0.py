__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
			   "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Ray N. Forcement"

from rlpy.Agents.Agent import Agent, DescentAlgorithm
import numpy as np, IPython
from rlpy.Tools import addNewElementForAllActions, count_nonzero

class SARSA0(DescentAlgorithm, Agent):

	def __init__(self, policy, representation, discount_factor, initial_learn_rate=0.1, **kwargs):
		super(SARSA0,self).__init__(policy=policy, representation=representation, discount_factor=discount_factor, **kwargs)
		self.logger.info("Initial learning rate:\t\t%0.2f" % initial_learn_rate)

	def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
		# The previous state could never be terminal
		# (otherwise the episode would have already terminated)
	#	IPython.embed()
		
		prevStateTerminal = False

		# MUST call this at start of learn()
		self.representation.pre_discover(s, prevStateTerminal, a, ns, terminal)

		# Compute feature function values and next action to be taken

		discount_factor = self.discount_factor # 'gamma' in literature
		feat_weights    = self.representation.weight_vec # Value function, expressed as feature weights
		features_s      = self.representation.phi(s, prevStateTerminal) # active feats in state
		features        = self.representation.phi_sa(s, prevStateTerminal, a, features_s) # active features or an (s,a) pair
		features_prime_s= self.representation.phi(ns, terminal)
		features_prime  = self.representation.phi_sa(ns, terminal, na, features_prime_s)
		nnz             = count_nonzero(features_s)  # Number of non-zero elements

		# Compute td-error
		td_error            = r + np.dot(discount_factor * features_prime - features, feat_weights)

		# Update value function (or if TD-learning diverges, take no action)
		if nnz > 0:
			feat_weights_old = feat_weights.copy()
			feat_weights               += self.learn_rate * td_error
			if not np.all(np.isfinite(feat_weights)):
				feat_weights = feat_weights_old
				print "WARNING: TD-Learning diverged, theta reached infinity!"

		# MUST call this at end of learn() - add new features to representation as required.
		expanded = self.representation.post_discover(s, False, a, td_error, features_s)

		# MUST call this at end of learn() - handle episode termination cleanup as required.
		if terminal:
			self.episodeTerminated()


	
