import random as random
import torch
from utils import RANDOM_SEED
import numpy as np
from random import sample
def train_test_data_gen(size, number_of_lists, percentage_of_train, seed=RANDOM_SEED):
    '''
    Generate number_of_lists times random lists of unique integers <= size of size 'size'
    for training and generate the same number of lists for testing.

    Return a 4-tuple, X_train, X_test, y_train, y_test
    
    '''
    random.seed(seed)
    lst = []
    sorted_lst = []
    for i in range(number_of_lists):
        lst.append(random.sample(range(1, 1000+1), size))
        sorted_lst.append(sorted(lst[i]))
    X_train = sample(lst, int(percentage_of_train * len(lst)))
    X_test = [sublist for sublist in lst if sublist not in X_train]
    y_train = [sorted(lst) for lst in X_train]
    y_test = [sorted(lst) for lst in X_test]

    return torch.tensor(X_train, dtype=float, requires_grad=True), \
            torch.tensor(X_test, dtype=float, requires_grad=True), \
            torch.tensor(y_train, dtype=float, requires_grad=True), \
            torch.tensor(y_test, dtype=float, requires_grad=True) 




