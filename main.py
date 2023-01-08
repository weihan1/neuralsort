from train import train
from model_class import DummyNNsort
from data_generation import train_test_data_gen
from metrics import loss_fn
import torch
from utils import device
from config.config import get_config




if __name__ == "__main__":
    args = get_config()
    X_train, X_test, y_train, y_test = train_test_data_gen(args.size, args.number_of_lists, args.percentage_of_train)
    model = DummyNNsort(size=100)
    train(model, epochs=args.epochs, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, loss= loss_fn, optimizer=torch.optim.Adam(model.parameters(), lr=0.001), device=device, path=args.path)
    
    model.load_state_dict(torch.load(args.path))
    model.eval()
    print("Input", X_test[0])
    with torch.inference_mode():
            test_pred = model(X_test[0].view(1, -1))
    print(test_pred)