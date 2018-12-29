import numpy as np
import pickle

fname = "AllTags81.txt"

with open(fname) as f:
    content = f.readlines()

fname = "Concepts81.txt"

with open(fname) as f:
    idx_to_concept = f.readlines()

for idx, line in enumerate(idx_to_concept):
    idx_to_concept[idx] = line.split('\n')[0]

n = len(content)
concept_matrix = [None] * n

for idx, line in enumerate(content):
    concepts = []
    for count, indicator in enumerate(line.split(' ')):
        if indicator != '\n' and int(indicator) == 1:
            concepts.append(idx_to_concept[count])
    concept_matrix[idx] = concepts

pickle.dump(concept_matrix, open('concept_matrix.p', 'wb'))
