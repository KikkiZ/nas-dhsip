import random

import numpy as np

from utils import upsample_space


def gene_swapping(genome: list):
    """ 基因互换

    :param genome: 输入可能产生互换的基因组
    :return: 返回互换完成的基因组
    """
    # 需要互换的基因组的长度
    length = round(len(genome) * 0.95)
    if length % 2 != 0:
        length = length + 1

    new_genome = []
    while length > 0:
        temp_1 = genome.pop(random.randrange(0, length))
        temp_2 = genome.pop(random.randrange(0, length - 1))

        start = random.randrange(0, len(temp_1))
        end = random.randrange(start, len(temp_1)) + 1

        new_genotype_1 = temp_1[0:start]
        new_genotype_1.extend(temp_2[start:end])
        new_genotype_1.extend(temp_1[end:len(temp_1)])

        new_genotype_2 = temp_2[0:start]
        new_genotype_2.extend(temp_1[start:end])
        new_genotype_2.extend(temp_2[end:len(temp_2)])

        new_genome.append(new_genotype_1)
        new_genome.append(new_genotype_2)
        length = length - 2

    return new_genome


def gene_mutation(genome: list):
    """ 基因突变

    :param genome: 输入可能产生突变的基因组
    :return: 返回突变完成的基因组
    """
    times = round(len(genome) * 0.10)

    while times >= 0:
        select = random.randrange(0, len(genome))
        index = random.randrange(0, len(genome[select]))

        if 0 <= index < 12:
            genome[select][index] = 0 if genome[select][index] == 1 else 1
        elif index == 12:
            genome[select][index] = random.randrange(0, len(upsample_space.UPSAMPLE_PRIMITIVE))
        elif index == 13:
            genome[select][index] = random.randrange(0, len(upsample_space.UPSAMPLE_CONV))
        elif index == 14:
            genome[select][index] = random.randrange(0, len(upsample_space.KERNEL_SIZE))
        else:
            genome[select][index] = random.randrange(0, len(upsample_space.ACTIVATION))

        times = times - 1

    return genome


def get_upsample_gene(genotype: np.ndarray):
    """ 获取上采样的组成方式

    :param genotype: 输入需要解析的基因型
    :return: 返回解析完成的数据
    """
    genotype = genotype[12:]

    return upsample_space.Upsample(primitive=genotype[0],
                                   conv=genotype[1],
                                   kernel=genotype[2],
                                   activation=genotype[3])


# 跳跃连接的连接位置
skip_gene = np.array([[1, 0, 1, 0, 0],
                      [1, 1, 0, 1, 0],
                      [0, 1, 1, 0, 1],
                      [0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 1]])


def get_skip_gene(genotype: np.ndarray):
    """ 获取跳跃连接的位置

    :param genotype: 输入需要解析的基因型
    :return: 返回解析完成的数据
    """
    genotype = genotype[0:12]
    gene = np.zeros(shape=(5, 5), dtype=np.int32)

    index = 0
    for i in range(5):
        for j in range(5):
            if skip_gene[i][j] == 1:
                gene[i][j] = genotype[index]
                index = index + 1

    return gene
