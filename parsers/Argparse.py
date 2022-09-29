import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("height",type=int, default=5, nargs='?')
    parser.add_argument("length",type=int, default=10, nargs='?')
    parser.add_argument("gamma",type=float, default=0.99, nargs='?')
    parser.add_argument("alpha",type=float, default=0.2, nargs='?')
    parser.add_argument("epsilon",type = float, default= 0.2, nargs='?')
    parser.add_argument("algo_name", type = str, default="qlearn", nargs='?')
    args = parser.parse_args()

    return args
