import math
import pickle
import sys
from features import *
from dataclasses import dataclass
from typing import Union

ADA_NO_OF_STUMPS = 1
MAX_DEPTH = 6


# -------------Decision Tree----------------
@dataclass
class Node:
    attribute: int
    labels: int
    true_branch: Union["Node", any]
    false_branch: Union["Node", any]


def get_data(file_name):
    data = []
    f = open(file_name, encoding="utf8")
    data = [line.strip() for line in f]
    f.close()
    return data


def split_dataset(attribute, data):
    true_split = []
    false_split = []
    for elem in data:
        if elem[attribute]:
            true_split.append(elem)
        else:
            false_split.append(elem)
    return true_split, false_split


def get_entropy(dataset):
    elements = {elem[-1]: sum(x.count(elem[-1]) for x in dataset) for elem in dataset}
    entropy = [-elements[i] / len(dataset) * math.log(elements[i] / len(dataset), 2) for i in elements]
    return sum(entropy)


def cal_IG(attri, dataset):
    initial_entropy = get_entropy(dataset)
    true_split, false_split = split_dataset(attri, dataset)
    entropy_false = (len(false_split) / len(dataset)) * get_entropy(false_split)
    entropy_true = (len(true_split) / len(dataset)) * get_entropy(true_split)
    return initial_entropy - (entropy_false + entropy_true)


def importance(attributes, dataset):
    info_gain = {a: cal_IG(a, dataset) for a in attributes}
    best_attribute = max(info_gain, key=info_gain.get)
    return best_attribute, info_gain[best_attribute]


def plurality_value(predictions):
    return max(predictions, key=predictions.get)


def get_label_count(dataset):
    count = {elem[-1]: sum(x.count(elem[-1]) for x in dataset) for elem in dataset}
    return count


def is_leaf(node):
    return (not node.true_branch and not node.false_branch)


def buildTree(dataset, depth):
    attri, info_gain = importance([i for i in range(len(dataset[0]) - 1)], dataset)
    if info_gain == 0 or depth == 0:
        return Node(attri, get_label_count(dataset), None, None)
    true_split, false_split = split_dataset(attri, dataset)
    true_branch = buildTree(true_split, depth - 1)
    false_branch = buildTree(false_split, depth - 1)
    return Node(attri, get_label_count(dataset), true_branch, false_branch)


def dt_predict(tuple, node):
    if is_leaf(node):
        return plurality_value(node.labels)
    if tuple[node.attribute]:
        return dt_predict(tuple, node.true_branch)
    else:
        return dt_predict(tuple, node.false_branch)


# ----------------------ADA BOOST------------------------------------

def normalize_weights(weights):
    total = sum(weights)
    norm = [float(i) / total for i in weights]
    return norm


def adaBoost(feature_matrix, n):
    no_of_features = len(feature_matrix[0]) - 1
    english_statements = [feature_matrix[i][-1] == 'en' for i in range(len(feature_matrix))]
    w = [(1 / len(feature_matrix)) for i in range(len(feature_matrix))]
    z = [0 for _ in range(no_of_features)]
    h = [None for _ in range(no_of_features)]
    for k in range(no_of_features):
        h[k] = buildTree(feature_matrix, ADA_NO_OF_STUMPS)
        error = 0
        for j in range(n):
            if dt_predict(feature_matrix, h[k]) is not english_statements[j]:
                error += w[j]
        for j in range(n):
            if dt_predict(feature_matrix, h[k]) is english_statements[j]:
                w[j] *= (error / (1 - error))
        w = normalize_weights(w)
        if error == 0:
            z[k] = float('inf')
        elif error == 1:
            z[k] = 0
        else:
            z[k] = math.log(abs(1 - error) / error)
    hypothesis = [(h[k].attribute, z[k]) for k in range(no_of_features)]
    return hypothesis


def ada_predict(matrix, tree_list):
    nl = en = 0
    for tree in tree_list:
        if matrix[tree[0]]:
            en += tree[1]
        else:
            nl += tree[1]
    if nl > en:
        return "nl"
    return "en"


# -------------------------------UTILITIES--------------------------

def classify(matrix, node):
    if isinstance(node, list):
        return ada_predict(matrix, node)
    else:
        return dt_predict(matrix, node)


def error_message():
    print("Usage: train <examples> <hypothesisOut> <learning-type>")
    print("Usage: predict <hypothesis> <file>")
    sys.exit(1)


def main():
    if len(sys.argv) < 4:
        error_message()
    mode = sys.argv[1]
    hypothesis = None
    if mode == "train":
        if len(sys.argv) != 5:
            error_message()
        file = sys.argv[2]
        hout_file = sys.argv[3]
        type = sys.argv[4]
        input = get_data(file)
        feature_matrix = get_features(input, True)
        if type == 'dt':
            hypothesis = buildTree(feature_matrix, MAX_DEPTH)
        elif type == 'ada':
            hypothesis = adaBoost(feature_matrix, len(input))
        else:
            error_message()
        pickle.dump(hypothesis, open(hout_file, 'wb'))
        print("Training Completed")
    elif mode == "predict":
        if len(sys.argv) != 4:
            error_message()
        hout_file = sys.argv[2]
        file = sys.argv[3]
        input = get_data(file)
        node = pickle.load(open(hout_file, 'rb'))
        test_matrix = get_features(input, False)
        # counter = {"nl":0, "en":0}
        for data in test_matrix:
            x = classify(data, node)
            print(x)
            # counter[x]+=1
        # print(counter)
    else:
        error_message()


if __name__ == "__main__":
    main()
