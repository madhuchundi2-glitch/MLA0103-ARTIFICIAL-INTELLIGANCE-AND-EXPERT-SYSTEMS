from math import log2
from collections import Counter, defaultdict

# Q3: Play Tennis dataset from the question
data = [
    ("D1", "Sunny",    "Hot",  "High",   "Weak",   "No"),
    ("D2", "Sunny",    "Hot",  "High",   "Strong", "No"),
    ("D3", "Overcast", "Hot",  "High",   "Weak",   "Yes"),
    ("D4", "Rain",     "Mild", "High",   "Weak",   "Yes"),
    ("D5", "Rain",     "Cool", "Normal", "Weak",   "Yes"),
    ("D6", "Rain",     "Cool", "Normal", "Strong", "No"),
    ("D7", "Overcast", "Cool", "Normal", "Strong", "Yes"),
    ("D8", "Sunny",    "Mild", "High",   "Weak",   "No"),
    ("D9", "Sunny",    "Cool", "Normal", "Weak",   "Yes"),
    ("D10","Rain",     "Mild", "Normal", "Weak",   "Yes"),
    ("D11","Sunny",    "Mild", "Normal", "Strong", "Yes"),
    ("D12","Overcast", "Mild", "High",   "Strong", "Yes"),
    ("D13","Overcast", "Hot",  "Normal", "Weak",   "Yes"),
    ("D14","Rain",     "Mild", "High",   "Strong", "No"),
]

FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]
IDX = {"Outlook": 1, "Temperature": 2, "Humidity": 3, "Wind": 4}
TARGET = 5

def entropy(rows):
    counts = Counter(r[TARGET] for r in rows)
    n = len(rows)
    return -sum((c/n) * log2(c/n) for c in counts.values())

def information_gain(rows, feature):
    groups = defaultdict(list)
    for r in rows:
        groups[r[IDX[feature]]].append(r)
    n = len(rows)
    return entropy(rows) - sum(
        len(g)/n * entropy(g) for g in groups.values()
    )

def majority(rows):
    return Counter(r[TARGET] for r in rows).most_common(1)[0][0]

def build_tree(rows, features):
    classes = {r[TARGET] for r in rows}

    if len(classes) == 1:
        return {"leaf": next(iter(classes))}

    if not features:
        return {"leaf": majority(rows)}

    best = max(features, key=lambda f: information_gain(rows, f))
    tree = {"feature": best, "branches": {}}
    remaining = [f for f in features if f != best]

    for value in sorted({r[IDX[best]] for r in rows}):
        subset = [r for r in rows if r[IDX[best]] == value]
        tree["branches"][value] = build_tree(subset, remaining)

    return tree

def print_tree(tree, indent=""):
    if "leaf" in tree:
        print(indent + "-> " + tree["leaf"])
        return

    print(indent + "[" + tree["feature"] + "]")

    for value, branch in tree["branches"].items():
        print(indent + f"  {value}: ", end="")
        if "leaf" in branch:
            print("-> " + branch["leaf"])
        else:
            print()
            print_tree(branch, indent + "    ")

def predict(tree, sample):
    if "leaf" in tree:
        return tree["leaf"]
    feature = tree["feature"]
    value = sample[feature]
    return predict(tree["branches"][value], sample)

# MAIN
print("=" * 60)
print("Q3 - DECISION TREE USING ID3 ALGORITHM")
print("=" * 60)

print(f"\nEntropy(S) = {entropy(data):.4f}")

print("\nInformation Gain:")
for feature in FEATURES:
    print(f"Gain({feature}) = {information_gain(data, feature):.4f}")

tree = build_tree(data, FEATURES)

print("\nDecision Tree:")
print_tree(tree)

print("\nExample Predictions:")
test_samples = [
    {"Outlook":"Sunny", "Temperature":"Cool", "Humidity":"High", "Wind":"Strong"},
    {"Outlook":"Rain", "Temperature":"Mild", "Humidity":"Normal", "Wind":"Weak"},
    {"Outlook":"Overcast", "Temperature":"Hot", "Humidity":"High", "Wind":"Strong"},
]

for i, sample in enumerate(test_samples, 1):
    print(f"Test {i}: {sample} => Play Tennis = {predict(tree, sample)}")
