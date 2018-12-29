import numpy as np
import pickle

fname = "nuswide_metadata/AllTags81.txt"

with open(fname) as f:
    content = f.readlines()

n = len(content)
relevancy_matrix = np.zeros((n,81), dtype=int)

for line, idx in zip(content, range(n)):
    relevancy_matrix[idx,:] = np.array([int(c) for c in line.split(' ') if c is not '\n'])

pickle.dump(relevancy_matrix, open('pickles/nuswide_metadata/relevancy_matrix.p', 'wb'))
