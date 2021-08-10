import numpy as np
from rank.RankNet import RankNet


label_file_pat = "./data/processed/%s_label.npy"
group_file_pat = "./data/processed/%s_group.npy"
feature_file_pat = "./data/processed/%s_feature.npy"


def load_data(type):

    labels = np.load(label_file_pat%type)
    qids = np.load(group_file_pat % type)
    features = np.load(feature_file_pat%type)

    X = {
        "feature": features,
        "label": labels,
        "qid": qids
    }
    return X

def train_ranknet():
    Xtrain, Xvalid = load_data("train"), load_data("vali")
    rankNet = RankNet()
    rankNet.fit(Xtrain['feature'], Xtrain['label'])

if __name__ == '__main__':
    train_ranknet()