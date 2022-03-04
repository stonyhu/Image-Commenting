import os
import sys
import numpy as np

samples = [line.strip() for line in open(sys.argv[1])]
np.random.shuffle(samples)
corpus = '\n'.join(samples)
print(corpus)
