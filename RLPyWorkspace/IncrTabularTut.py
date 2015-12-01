__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
			   "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Ray N. Forcement"

from rlpy.Representations.Representation import Representation
import numpy as np
from copy import deepcopy

class IncrTabularTut(Representation):
	"""
	Tutorial representation: identical to IncrementalTabular

	"""
	hash = None
	def __init__(self, domain, discretization=20):
		self.hash           = {}
		self.features_num   = 0
		self.isDynamic      = True
		super(IncrTabularTut, self).__init__(domain, discretization)    

	def phi_nonTerminal(self, s):
		hash_id = self.hashState(s)
		id  = self.hash.get(hash_id)
		F_s = np.zeros(self.features_num, bool)
		if id is not None:
			F_s[id] = 1
		return F_s  

	def featureType(self):
		return bool
	
	def pre_discover(self, s, terminal, a, sn, terminaln):
		return self._add_state(s) + self._add_state(sn)

	def _add_state(self, s):
		hash_id = self.hashState(s)
		id  = self.hash.get(hash_id)
		if id is None:
			#New State
			self.features_num += 1
			#New id = feature_num - 1
			id = self.features_num - 1
			self.hash[hash_id] = id
			#Add a new element to the feature weight vector
			self.addNewWeight()
			return 1
		return 0
	
	def __deepcopy__(self, memo):
		new_copy = IncrementalTabular(self.domain, self.discretization)
		new_copy.hash = deepcopy(self.hash)
		return new_copy

	
