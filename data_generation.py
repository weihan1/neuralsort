import random as random
import torch
from utils import RANDOM_SEED
import numpy as np
def train_test_data_gen(size, number_of_lists, seed=RANDOM_SEED):
    '''
    Generate number_of_lists times random lists of unique integers <= size of size 'size'
    for training and generate the same number of lists for testing.

    Return a tuple (number_of_lists, size), (number_of_lists, size) containing the 
    lists for training and testing.
    '''
    random.seed(seed)
    lst = []
    sorted_lst = []
    for i in range(number_of_lists):
        lst.append(random.sample(range(1, size+1), size))
        sorted_lst.append(sorted(lst[i]))
    
    return torch.tensor(lst, dtype= float,requires_grad=True), \
        torch.tensor(sorted_lst, dtype= float,requires_grad=True)



