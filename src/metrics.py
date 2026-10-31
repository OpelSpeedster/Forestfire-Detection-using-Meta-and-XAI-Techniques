def prf1(y_true, y_pred):
    labels = ["smoke", "fire", "no_fire"]
    out = {}
    for l in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == l and b == l)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != l and b == l)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == l and b != l)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f = 2 * p * r / (p + r) if p + r else 0
        out[l] = {"precision": p, "recall": r, "f1": f}
    return out

def accuracy(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0
