import sympy as sp 
import numpy as np 
from mpmath import log10

expected = np.array([1, 0, 0], np.float32)
predicted = np.array([0.124513, 0.542434, 0.1235432])

print(-1 * ((expected / predicted) + (1 - expected) / (1 - predicted)))