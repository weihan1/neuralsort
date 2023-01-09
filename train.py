import torch
from utils import RANDOM_SEED
def train(model, epochs, X_train, X_test, y_train, y_test, loss, optimizer, device, path,seed=RANDOM_SEED):
    '''
    Train the model on the training data and return the trained model
    '''
    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    model.train()
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss_value = loss(y_pred, y_train)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        
    
        model.eval()
        with torch.inference_mode():
            test_pred = model(X_test)
            test_loss = loss(test_pred, y_test.type(torch.FloatTensor))
            
            
            if epoch % 1000 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss_value.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | Training Loss: {loss_value} | Test Loss: {test_loss} ")
    torch.save(model.state_dict(), path)