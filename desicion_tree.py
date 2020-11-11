import math
from collections import Counter
import numpy as np
from sklearn import datasets
import random


class Node:
    def __init__(self, data=''):
        self.data = data
        self.children = []
        self.value = -1

    def set_value(self, value):
        self.value = value

    def is_empty(self):
        return (self.data is None)

    def set_data(self, data):
        self.data = data

    def set_value(self, val):
        self.value = val

    def get_value(self):
        return self.value

    def get_data(self):
        return self.data

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def __repr__(self, level=0):
        representation = "\t" * level
        if self.value >= 0 :
            representation += "value:" + repr(self.value)
        representation+=" |" + "-" * level
        if self.data > 999:
            representation += " Class" + repr(self.data - 1000) + "\n"
        else:
            representation += " attribute" + repr(self.data) + "\n"
        for child in self.children:
            representation += child.__repr__(level + 1)
        return representation


def stop_condition(labels, limit_post_pruning, records=[]):
    x = Counter(labels)
    if len(records) < limit_post_pruning:
        return True
    if len(x) > 1:
        return False
    else:
        return True


""" best_split takes as input 'records' : 2D list of records
    returns attribute with lowest gini split as a numeric value in the range of 0 to # of attributes - 1 """


def best_split(records):
    # n : number of attributes
    # gsplits: list of calculated gini splits for all attributes
    n = len(records[0]) - 1  # number of attribute
    d = len(records)  # number of record
    label_vals = [records[y][-1] for y in range(len(records))]  # get label values
    gsplits = []

    # iterate over all attributes to calculate their gsplits
    for i in range(n):
        att_vals = [records[j][i] for j in range(len(records))]  # get attribute values of ith attribute
        C = Counter(att_vals)  # get counts of all unique attribute values
        IG_vals = []  # list to store gini index values of all values of a particular attribute
        nr_subsets = []  # list to store number of records in each subset of attribute values

        # partition the data by each unique attribute value while getting labels into a list of 2D lists where
        # elements are [att_value, label_value] calculate gini values of each attribute value
        for attribute in C.keys():
            att_subset = [[att_vals[u], label_vals[u]] for u in range(len(att_vals)) if att_vals[u] == attribute]
            nr = len(att_subset)
            nr_subsets.append(nr)
            labels_of_subset = [att_subset[u][1] for u in range(len(att_subset))]
            local_c = Counter(labels_of_subset)
            IG = sum(-(v / nr) * math.log2(v / nr) for v in local_c.values())  # p(label) = v/nr
            IG_vals.append(IG)

        gs = sum((nr_subsets[x] / d) * IG_vals[x] for x in range(len(nr_subsets)))
        gsplits.append(gs)

    return gsplits.index(min(gsplits))


""" Function Build_Tree takes in list of records(type 2D list) and list of 
    attributes(type 1D list) and returns a root of decision tree (type Node) """


def build_tree(records, attributes, limit_post_pruning):
    root = Node()
    labels = [r[-1] for r in records]
    if stop_condition(labels, limit_post_pruning, records):
        root.set_data(1000 + (labels[0]))
        return root

    if attributes.count(0) == 0:  # majority voting
        c = Counter(labels)
        majority = c.most_common(1)[0][0]  # since most_common returns a list of (element,count) tuples
        root.set_data(1000 + (majority))
        return root

    split_attribute = best_split(records)
    root.set_data(split_attribute)
    attributes[split_attribute] = 1
    split_attribute_vals = [records[i][split_attribute] for i in range(len(records))]
    val_counts = Counter(split_attribute_vals)
    for val in val_counts.keys():
        partition = [records[i] for i in range(len(records)) if records[i][split_attribute] == val]
        if len(partition) == 0:
            c = Counter(labels)
            majority = c.most_common(1)[0][0]
            leaf = Node(1000 + (majority))
            leaf.set_value(val)
            root.add_child(leaf)
        else:
            child = build_tree(partition, attributes, limit_post_pruning)
            child.set_value(val)
            root.add_child(child)

    return root


def find_class(tree, record):
    # print(tree.get_data())
    if tree.get_data() > 999:
        return tree.get_data() - 1000
    else:
        children = tree.get_children()
        for child in children:
            if child.get_value() == record[tree.get_data()]:
                return find_class(child, record)


def check_correctness(tree, records):
    correct = 0
    for record in records:
        predicted_class = find_class(tree, record)
        # print(str(record[-1]))#+str(predicted_class))
        if record[-1] == predicted_class:
            correct = correct + 1
    return correct / len(records)


def open_file():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    records = np.stack(( x[:,0],x[:,1],x[:,2],x[:,3],y), axis=1)
    return records


def Hunt(limit=0):
    data = open_file()
    # shuffle data to have all classes in train set and test set
    random.shuffle(data)
    random.shuffle(data)
    l_train = 1 * (len(data) // 3)
    [test_set, train_set] = np.split(data, [l_train])
    n_records = len(train_set)
    n_attributes = len(train_set[0]) - 1
    print("\n")
    print("------Input Configuration------")
    print("\nNumber of records: ", n_records)
    print("\nNumber of attributes: ", n_attributes)
    print("\n")
    print("-------------------------------\n")

    att = [0 for i in range(n_attributes)]
    tree = build_tree(train_set, att, limit)
    print(tree)
    print("correctness of train set :" + str(check_correctness(tree, train_set)))
    correctness = check_correctness(tree, test_set)
    print("correctness of train set :" + str(correctness))
    return correctness

Hunt()

