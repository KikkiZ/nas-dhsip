import random

import numpy as np

from utils import gene_utils

arr = [0] * 16
for index in range(12):
    if random.randint(1, 100) <= 50:
        arr[index] = 0
    else:
        arr[index] = 1

# for index in range(5):
#     arr[25 + index * 4] = random.randint(0, 4)
#     arr[26 + index * 4] = random.randint(0, 5)
#     arr[27 + index * 4] = random.randint(0, 2)
#     arr[28 + index * 4] = random.randint(0, 2)

arr[12] = random.randint(0, 4)
arr[13] = random.randint(0, 5)
arr[14] = random.randint(0, 2)
arr[15] = random.randint(0, 2)

print(arr)
print(gene_utils.get_upsample_gene(np.asarray(arr)))

gene = gene_utils.get_skip_gene(np.asarray(arr))
print(gene)
# print(gene[4][3], gene[4][4])
# x = 0
# print(len([i for i in range(0, 5) if genes[0][i] == gene[0][i] == 1]))

# gene = np.array(arr, dtype=bool)
# print(type(gene))
# genes = gene_utils.get_upsample_gene(gene)
# print(genes)

print(gene_utils.skip_gene.dtype)
