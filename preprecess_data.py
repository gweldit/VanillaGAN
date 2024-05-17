import os
from pathlib import Path
from random import shuffle
from typing import List
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils.convert import from_networkx


def read_subfolder(path: str, label):
    sequences = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r") as f:
                lines = f.readlines()
                # print(lines)

                for line in lines:
                    line = list(map(int, line.strip().split()))
                    sequences.append(line)
                    labels.append(label)

                    sequences.append(line)
                    labels.append(label)

    return sequences, labels


def read_adfa_data(path: str):
    """
    sub_dir: data folder full path (e.g. "/../.../ADFA/Training_Data_Master")
    Read all files in the data folder and return a list of sequences.
    """
    folder_sequences = []
    folder_labels = []

    # check labels when reading attack data to be able to assign label to 1
    label = 0  # indicates benign
    if "Attack_Data_Master" in path:
        label = 1
        for sub_folder in os.listdir(path):
            sub_folder_path = os.path.join(path, sub_folder)
            if os.path.isdir(sub_folder_path):
                # print("processing folder: ", sub_folder)
                sub_folder_sequences, sub_folder_labels = read_subfolder(
                    sub_folder_path, label=label
                )
            folder_sequences += sub_folder_sequences
            folder_labels.extend(sub_folder_labels)

        # return a list of sequences, and labels for the attack data
        return folder_sequences, folder_labels

    # return a list of sequences, and labels for the benign data
    return read_subfolder(path, label=label)


def sequence_to_graph(L: List, graph_label=None, vocab_size=None):
    """
    Convert a sequence of (integers) to a graph.
    Currently, we are using already encoded set of integers that represent system calls. If raw data is used, it will be necessary to encode the data first using a dictionary.
    """
    # create a graph
    G = nx.DiGraph()
    for i in range(len(L) - 1):
        edge = (L[i], L[i + 1])
        # if edge is not in the graph
        if not G.has_edge(*edge):
            G.add_edge(*edge, weight=1)

        # if edge is in the graph, just update the weight
        else:
            u, v = edge
            G[u][v]["weight"] += 1

    # add node attributes
    node_attr = []
    # convert networkx graph to pyg graph data

    nodes = torch.tensor(list(G.nodes), dtype=torch.long)
    node_attr = [nodes]

    # convert graph to pytorch geometric data
    data = from_networkx(G)
    data.x = node_attr
    if graph_label is not None:
        data.y = graph_label

    # validate the data
    data.validate(raise_on_error=True)

    return G, data


def fetch_graph_data(sequences, labels, vocab_size=342):
    graphs = []
    for i in range(len(sequences)):
        nx_graph_G, pyg_graph_data = sequence_to_graph(
            sequences[i], graph_label=labels[i], vocab_size=vocab_size
        )
        graphs.append(pyg_graph_data)
    return graphs
