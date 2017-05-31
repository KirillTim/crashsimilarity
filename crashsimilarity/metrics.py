import math

def true_positive(labels_true, labels_pred):
    ans = set()
    for i, a in enumerate(labels_true):
        for j, b in enumerate(labels_true):
            if j <= i:
                continue
            if a == b and labels_pred[i] == labels_pred[j] and labels_pred[i] != -1:
                ans.add((min(i,j), max(i,j)))
    return len(ans)

def false_positive(labels_true, labels_pred):
    ans = set()
    for i, a in enumerate(labels_true):
        for j, b in enumerate(labels_true):
            if j <= i:
                continue
            if a == b and labels_pred[i] != labels_pred[j] and labels_pred[i] != -1:
                ans.add((min(i,j), max(i,j)))
    return len(ans)
    
def false_negative(labels_true, labels_pred):
    ans = set()
    for i, a in enumerate(labels_true):
        for j, b in enumerate(labels_true):
            if j <= i:
                continue
            if a != b and labels_pred[i] == labels_pred[j] and labels_pred[i] != -1:
                ans.add((min(i,j), max(i,j)))
    return len(ans)

def my_precision(labels_true, labels_pred):
    return float(true_positive(labels_true, labels_pred)) / (true_positive(labels_true, labels_pred) + false_positive(labels_true, labels_pred))

def my_recall(labels_true, labels_pred):
    return float(true_positive(labels_true, labels_pred)) / (true_positive(labels_true, labels_pred) + false_negative(labels_true, labels_pred))

def my_FMI(labels_pred, labels_true):
    tp = true_positive(labels_pred, labels_true)
    fp = false_positive(labels_pred, labels_true)
    fn = false_negative(labels_pred, labels_true)
    return float(tp) / math.sqrt((tp+fp) * (tp+fn))