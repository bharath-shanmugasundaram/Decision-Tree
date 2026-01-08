from flask import Flask, request, jsonify, render_template
import torch as t
import torch.nn as nn
import torch
import numpy as np
import pandas as pd


app = Flask(__name__)
 

df = pd.read_csv("/Users/bhara-zstch1566/Machine Learning/Decission Tree/play_tennis.csv")
df = df.drop(columns="day")

df["play"] = df["play"].map({"No":0,"Yes":1})

df["wind"] = df["wind"].map({'Weak':0, 'Strong':1})

df["humidity"] = df["humidity"].map({'High':1, 'Normal':0})

df["temp"] = df["temp"].map({'Hot':1, 'Mild':2, 'Cool':3})

df["outlook"] = df["outlook"].map({"Rain":1,"Overcast":2,"Sunny":3})

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

X = X.to_numpy()
Y = Y.to_numpy()

X = torch.tensor(X,dtype=torch.float32)
Y = torch.tensor(Y)

def entropy(y):
    classes, counts = torch.unique(y, return_counts=True)
    probs = counts.float() / counts.sum()
    return -torch.sum(probs * torch.log2(probs))



def information_gain(X, y, feature_index):
    parent_entropy = entropy(y)
    values = torch.unique(X[:, feature_index])

    weighted_entropy = 0.0
    for v in values:
        mask = X[:, feature_index] == v
        y_subset = y[mask]
        weighted_entropy += (len(y_subset) / len(y)) * entropy(y_subset)

    return parent_entropy - weighted_entropy


def best_split(X, y):
    gains = []
    for i in range(X.shape[1]):
        gains.append(information_gain(X, y, i))
    return torch.argmax(torch.tensor(gains)).item()


class Node:
    def __init__(self, feature=None, children=None, value=None):
        self.feature = feature      
        self.children = children   
        self.value = value

def build_tree(X, y):
    if len(torch.unique(y)) == 1:
        return Node(value=y[0].item())

    if X.shape[1] == 0:

        majority = torch.mode(y).values.item()
        return Node(value=majority)

    feature = best_split(X, y)
    node = Node(feature=feature, children={})

    for v in torch.unique(X[:, feature]):
        mask = X[:, feature] == v
        node.children[v.item()] = build_tree(X[mask], y[mask])

    return node

tree = build_tree(X, Y)

def tree_predict(node, x):
    if node.value is not None:
        return node.value

    feature_value = x[node.feature].item()
    return tree_predict(node.children[feature_value], x)



# ---- Routes ----
@app.route("/")
def home():
    return render_template("index.html", input_size=4)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", None)
    if features is None or len(features) != 4:
        return jsonify({"error": f"Expected {4} features"}), 400

    
    x = t.tensor(features, dtype=t.float32)
    pred_class = tree_predict(tree, x)
    
    return jsonify({"probability": 1, "prediction": pred_class})

if __name__ == "__main__":
    app.run(debug=True)

# %%
