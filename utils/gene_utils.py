import numpy as np

from utils import upsample_space


def gene_swapping():
    return


def gene_mutation():
    return


def get_upsample_gene(genotype: np.ndarray):
    genotype = genotype[12:]

    return upsample_space.Upsample(primitive=genotype[0],
                                   conv=genotype[1],
                                   kernel=genotype[2],
                                   activation=genotype[3])


def get_skip_gene(genotype: np.ndarray):
    genotype = genotype[0:12]
    gene = np.zeros(shape=(5, 5), dtype=np.int32)

    index = 0
    for i in range(5):
        for j in range(5):
            if skip_gene[i][j] == 1:
                gene[i][j] = genotype[index]
                index = index + 1

    return gene


skip_gene = np.array([[1, 0, 1, 0, 0],
                      [1, 1, 0, 1, 0],
                      [0, 1, 1, 0, 1],
                      [0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 1]])
