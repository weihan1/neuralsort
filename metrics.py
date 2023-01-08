from torch import nn
def loss_fn(outputs, targets):
    '''
    Loss function for the neural network. Treat the problem as a regression problem.
    '''
    return nn.MSELoss()(outputs, targets)

