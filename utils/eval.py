import logging

import numpy as np
from sklearn import metrics
import torch
from torchmetrics.classification import BinaryF1Score, BinaryPrecision


# vus
def range_convers_new(label):
    '''
    input: arrays of binary values
    output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
    '''
    L = []
    i = 0
    j = 0
    while j < len(label):
        while label[i] == 0:
            i += 1
            if i >= len(label):  # ?
                break  # ?
        j = i + 1
        if j >= len(label):
            if j == len(label):
                L.append((i, j - 1))

            break
        while label[j] != 0:
            j += 1
            if j >= len(label):
                L.append((i, j - 1))
                break
        if j >= len(label):
            break
        L.append((i, j - 1))
        i = j
    return L


def new_sequence(label, sequence_original, window):
    a = max(sequence_original[0][0] - window // 2, 0)
    sequence_new = []
    for i in range(len(sequence_original) - 1):
        if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
            sequence_new.append((a, sequence_original[i][1] + window // 2))
            a = sequence_original[i + 1][0] - window // 2
    sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
    return sequence_new


def sequencing(x, L, window=5):
    label = x.copy().astype(float)
    length = len(label)

    for k in range(len(L)):
        s = L[k][0]
        e = L[k][1]

        x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
        label[x1] += np.sqrt(1 - (x1 - e) / (window))

        x2 = np.arange(max(s - window // 2, 0), s)
        label[x2] += np.sqrt(1 - (s - x2) / (window))

    label = np.minimum(np.ones(length), label)
    return label


# TPR_FPR_window
def RangeAUC_volume_opt(labels_original, score, windowSize, thre=250):
    window_3d = np.arange(0, windowSize + 1, 1)
    P = np.sum(labels_original)
    seq = range_convers_new(labels_original)
    l = new_sequence(labels_original, seq, windowSize)

    score_sorted = -np.sort(-score)

    tpr_3d = np.zeros((windowSize + 1, thre + 2))
    fpr_3d = np.zeros((windowSize + 1, thre + 2))
    prec_3d = np.zeros((windowSize + 1, thre + 1))

    auc_3d = np.zeros(windowSize + 1)
    ap_3d = np.zeros(windowSize + 1)

    tp = np.zeros(thre)
    N_pred = np.zeros(thre)

    for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
        threshold = score_sorted[i]
        pred = score >= threshold
        N_pred[k] = np.sum(pred)

    for window in window_3d:

        labels_extended = sequencing(labels_original, seq, window)
        L = new_sequence(labels_extended, seq, window)

        TF_list = np.zeros((thre + 2, 2))
        Precision_list = np.ones(thre + 1)
        j = 0

        for i in np.linspace(0, len(score) - 1, thre).astype(int):
            threshold = score_sorted[i]
            pred = score >= threshold
            labels = labels_extended.copy()
            existence = 0

            for seg in L:
                labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                if (pred[seg[0]:(seg[1] + 1)] > 0).any():
                    existence += 1
            for seg in seq:
                labels[seg[0]:seg[1] + 1] = 1

            TP = 0
            N_labels = 0
            for seg in l:
                TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                N_labels += np.sum(labels[seg[0]:seg[1] + 1])

            TP += tp[j]
            FP = N_pred[j] - TP

            existence_ratio = existence / len(L)

            P_new = (P + N_labels) / 2
            recall = min(TP / P_new, 1)

            TPR = recall * existence_ratio
            N_new = len(labels) - P_new
            FPR = FP / N_new

            Precision = TP / N_pred[j]

            j += 1
            TF_list[j] = [TPR, FPR]
            Precision_list[j] = Precision

        TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

        tpr_3d[window] = TF_list[:, 0]
        fpr_3d[window] = TF_list[:, 1]
        prec_3d[window] = Precision_list

        width = TF_list[1:, 1] - TF_list[:-1, 1]
        height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
        AUC_range = np.dot(width, height)
        auc_3d[window] = (AUC_range)

        width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
        height_PR = Precision_list[1:]

        AP_range = np.dot(width_PR, height_PR)
        ap_3d[window] = AP_range

    return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)


def generate_curve(label, score, slidingWindow, thre=250):

    tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = RangeAUC_volume_opt(
        labels_original=label, score=score, windowSize=slidingWindow, thre=thre)

    return avg_auc_3d, avg_ap_3d


# auc
def metric_new_auc(label, score):
    if np.sum(label) == 0:
        print('All labels are 0. Label must have groud truth value for calculating AUC score.')
        return None

    if np.isnan(score).any() or score is None:
        print('Score must not be none.')
        return None

    # area under curve
    auc = metrics.roc_auc_score(label, score)

    return auc


# composit f1
def get_events(y_test, outlier=1, normal=0):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
        else:
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events


def get_composite_fscore(pred_labels, true_events, true_labels):
    tp = torch.stack([
        pred_labels[start:end + 1].any()
        for start, end in true_events.values()
    ]).sum()

    fn = len(true_events) - tp
    rec_e = tp/(tp + fn)
    prec_metric = BinaryPrecision().to(true_labels.device)
    prec_t = prec_metric(pred_labels, true_labels)


    if rec_e == 0 and prec_t == 0:
        return 0.0

    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)

    return fscore_c.item()



def get_pointwise_fscore(true_labels, pred_labels):
    metric = BinaryF1Score().to(true_labels.device)
    return metric(pred_labels, true_labels).item()


def collect_metrics(label, score, thresh_=250):

    label_, score_ = label.cpu().numpy(), score.cpu().numpy()
    precision, recall, thresholds = metrics.precision_recall_curve(label_.astype(float), score_.astype(float))

    AUC_ROC = metric_new_auc(label_, score_)
    AUC_PR = metrics.auc(recall, precision)

    true_events = get_events(label)

    event_lengths = [e - b for (b, e) in true_events.values()]
    # print(event_lengths)
    median_len = np.median(event_lengths)
    # vus_sliding_window = max(100, int(median_len // 2))  # 100 for QAD
    vus_sliding_window = min(50, max(5, int(median_len // 2)))  # 50 for ASD
    # print('vus sliding window size', vus_sliding_window)

    VUS_ROC, VUS_PR = generate_curve(label_, score_, vus_sliding_window, thresh_)

    best_f1_p = 0.0
    best_f1_c = 0.0
    best_f1_c_thresh = None

    thresholds = torch.unique(score)
    # print(len(thresholds))
    if thresholds.numel() > 1000:
        thresholds = torch.linspace(score.min(), score.max(), 500, device=score.device)

    for i in thresholds:

        pred_binary = (score >= i).long()

        # point-wise F1
        f1_p_ = get_pointwise_fscore(label, pred_binary)
        best_f1_p = max(best_f1_p, f1_p_)

        # composite F1
        f1_c_ = get_composite_fscore(pred_binary, true_events, label)
        if f1_c_ >= best_f1_c:
            best_f1_c = f1_c_
            best_f1_c_thresh = i

    if best_f1_c_thresh is None:
        best_f1_c_thresh = thresholds[0]


    return VUS_ROC, VUS_PR, AUC_ROC, AUC_PR, best_f1_p, best_f1_c
