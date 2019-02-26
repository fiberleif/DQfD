import argparse
import matplotlib as plt
import pickle

def map_scores(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data1_path", type=str, default=None)
    parser.add_argument("--data2_path", type=str, default=None)
    args = parser.parse_args()

    with open('./' + args.data1_path, 'rb') as f:
       data1 = pickle.load(f)
    with open('./' + args.data2_path, 'rb') as f:
        data2 = pickle.load(f)

    map_scores(dqfd_scores=data2, ddqn_scores=data1, xlabel='Red: data2         Blue: data1', ylabel='Scores')
