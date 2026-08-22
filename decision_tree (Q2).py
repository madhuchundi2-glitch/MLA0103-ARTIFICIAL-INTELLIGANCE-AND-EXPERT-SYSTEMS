from math import log2
from collections import Counter

# ============================================================
# DECISION TREE - QUESTION 2
# Dataset taken from the second uploaded image
#
# Instance | a1    | a2  | a3     | Class
#    1     | True  | Hot | High   | No
#    2     | True  | Hot | High   | No
#    3     | False | Hot | High   | Yes
#    4     | False | Cool| Normal | Yes
#    5     | False | Cool| Normal | Yes
#    6     | True  | Cool| High   | No
#    7     | True  | Hot | High   | No
#    8     | False | Hot | Normal | Yes
#    9     | False | Cool| Normal | Yes
#   10     | False | Cool| High   | Yes
# ============================================================

data = [
    (1,  True,  "Hot",  "High",   "No"),
    (2,  True,  "Hot",  "High",   "No"),
    (3,  False, "Hot",  "High",   "Yes"),
    (4,  False, "Cool", "Normal", "Yes"),
    (5,  False, "Cool", "Normal", "Yes"),
    (6,  True,  "Cool", "High",   "No"),
    (7,  True,  "Hot",  "High",   "No"),
    (8,  False, "Hot",  "Normal", "Yes"),
    (9,  False, "Cool", "Normal", "Yes"),
    (10, False, "Cool", "High",   "Yes"),
]

FEATURES = ["a1", "a2", "a3"]
FEATURE_INDEX = {"a1": 1, "a2": 2, "a3": 3}
TARGET_INDEX = 4


def entropy(rows):
    """Calculate entropy of the class column."""
    if not rows:
        return 0.0

    counts = Counter(row[TARGET_INDEX] for row in rows)
    total = len(rows)

    return -sum(
        (count / total) * log2(count / total)
        for count in counts.values()
    )


def information_gain(rows, feature):
    """Calculate Information Gain for a feature."""
    parent_entropy = entropy(rows)
    total = len(rows)
    index = FEATURE_INDEX[feature]

    groups = {}
    for row in rows:
        groups.setdefault(row[index], []).append(row)

    weighted_entropy = sum(
        len(group) / total * entropy(group)
        for group in groups.values()
    )

    return parent_entropy - weighted_entropy


def build_tree(rows, features):
    """Build an ID3 decision tree."""
    classes = {row[TARGET_INDEX] for row in rows}

    if len(classes) == 1:
        return {"leaf": next(iter(classes))}

    if not features:
        majority = Counter(
            row[TARGET_INDEX] for row in rows
        ).most_common(1)[0][0]
        return {"leaf": majority}

    best = max(
        features,
        key=lambda feature: information_gain(rows, feature)
    )

    tree = {"feature": best, "branches": {}}
    index = FEATURE_INDEX[best]

    remaining = [f for f in features if f != best]

    for value in sorted(
        {row[index] for row in rows},
        key=str
    ):
        subset = [row for row in rows if row[index] == value]
        tree["branches"][value] = build_tree(subset, remaining)

    return tree


def print_tree(tree, indent=""):
    if "leaf" in tree:
        print(indent + "-> " + tree["leaf"])
        return

    print(indent + tree["feature"])

    for value, subtree in tree["branches"].items():
        print(indent + f"  {value}: ", end="")
        if "leaf" in subtree:
            print("-> " + subtree["leaf"])
        else:
            print()
            print_tree(subtree, indent + "    ")


def predict(tree, sample):
    if "leaf" in tree:
        return tree["leaf"]

    feature = tree["feature"]
    value = sample[feature]

    if value not in tree["branches"]:
        raise ValueError(
            f"Unknown value '{value}' for {feature}"
        )

    return predict(tree["branches"][value], sample)


# ---------------- MAIN PROGRAM ----------------

print("=" * 60)
print("DECISION TREE - QUESTION 2 (ID3)")
print("=" * 60)

print("\nDataset Entropy:")
print(f"Entropy(S) = {entropy(data):.4f}")

print("\nInformation Gain:")
for feature in FEATURES:
    print(
        f"Gain({feature}) = "
        f"{information_gain(data, feature):.4f}"
    )

tree = build_tree(data, FEATURES)

print("\nDecision Tree:")
print_tree(tree)

print("\nExample Predictions:")

test_samples = [
    {"a1": True,  "a2": "Hot",  "a3": "High"},
    {"a1": False, "a2": "Hot",  "a3": "High"},
    {"a1": False, "a2": "Cool", "a3": "Normal"},
    {"a1": False, "a2": "Cool", "a3": "High"},
]

for i, sample in enumerate(test_samples, 1):
    print(
        f"Test {i}: {sample} "
        f"=> Class = {predict(tree, sample)}"
    )