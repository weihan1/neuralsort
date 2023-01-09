from torch import nn
import torch

def is_increasing(outputs):
    '''
    Check if the output is monotonically increasing.

    Precondition: the outputs is of size (args.number_of_lists x args.size)
    '''
    # print("outputs", outputs.shape)
    incorrect = 0
    for i in range(outputs.shape[0]):
        if not torch.all(torch.eq(torch.sort(outputs[i])[0], outputs[i])):
            incorrect += 1

    return incorrect


def is_permutation(outputs, targets):
    '''
    Check if the output is a permutation of the target.

    Precondition: the outputs is of size (args.number_of_lists x args.size)
    '''
    incorrect = 0
    for i in range(outputs.shape[0]): #for each training example
        if not torch.all(torch.eq(torch.sort(outputs[i])[0], torch.sort(targets[i])[0])):
            incorrect += 1

    return incorrect


def loss_fn(outputs, targets):
    '''
    Loss function for the neural network. Treat the problem as a regression problem.
    '''
    
    return torch.tensor(is_permutation(outputs, targets), dtype=torch.float32, requires_grad =True) + \
        torch.tensor(is_increasing(outputs), dtype=torch.float32, requires_grad=True)
