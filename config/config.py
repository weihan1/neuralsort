import argparse

parser = argparse.ArgumentParser(description='Data generation algorithm')
parser.add_argument("--size", type=int, default = 100, help="size of the dataset to generate")
parser.add_argument("--number_of_lists", type=int, default=10000,help="number of lists to generate")
parser.add_argument("--percentage_of_train", type=float, default = 0.8,help="percentage of the dataset to use for training")
parser.add_argument("--epochs", type=int, default = 10000,help="number of epochs to train the model")
parser.add_argument("--path", type=str, default = "model_params/model.pt",help="path to save the model")

def get_config():
    args = parser.parse_args()
    return args