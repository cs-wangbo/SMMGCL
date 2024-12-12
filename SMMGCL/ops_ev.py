import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster._supervised import check_clusterings


def get_evaluation_results(labels_true, labels_pred):
    NMI = normalized_mutual_info_score(labels_true, labels_pred)
    ARI = adjusted_rand_score(labels_true, labels_pred)
    ACC = clustering_accuracy(labels_true, labels_pred)
    return ACC, NMI, ARI

def clustering_purity(labels_true, labels_pred):
    y_true = labels_true.copy()
    y_pred = labels_pred.copy()
    if y_true.shape[1] != 1:
        y_true = y_true.T
    if y_pred.shape[1] != 1:
        y_pred = y_pred.T

    n_samples = len(y_true)

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    y_true_temp = np.zeros((n_samples, 1))
    if n_true_classes != max(y_true):
        for i in range(n_true_classes):
            y_true_temp[np.where(y_true == u_y_true[i])] = i + 1
        y_true = y_true_temp

    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)
    y_pred_temp = np.zeros((n_samples, 1))
    if n_pred_classes != max(y_pred):
        for i in range(n_pred_classes):
            y_pred_temp[np.where(y_pred == u_y_pred[i])] = i + 1
        y_pred = y_pred_temp

    u_y_true = np.unique(y_true)
    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)

    n_correct = 0
    for i in range(n_pred_classes):
        incluster = y_true[np.where(y_pred == u_y_pred[i])]

        inclunub = np.histogram(incluster, bins = range(1, int(max(incluster)) + 1))[0]
        if len(inclunub) != 0:
            n_correct = n_correct + max(inclunub)

    Purity = n_correct/len(y_pred)

    return Purity


def clustering_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    from scipy.optimize import linear_sum_assignment as linear_assignment  # 添加as语句不用修改代码中的函数名

    u = linear_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def b3_precision_recall_fscore(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")
    n_samples = len(labels_true)
    true_clusters = {}
    pred_clusters = {}

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score
