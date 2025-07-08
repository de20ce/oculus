from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

def compute_metrics(pred_file, protocol_file):
    preds = []
    labels = []

    with open(protocol_file, 'r') as f1, open(pred_file, 'r') as f2:
        proto_lines = f1.readlines()
        pred_lines = f2.readlines()

    for proto_line, pred_line in zip(proto_lines, pred_lines):
        proto_parts = proto_line.strip().split()
        pred_parts = pred_line.strip().split()

        if len(proto_parts) != 2 or len(pred_parts) != 2:
            print(f" Skipping malformed line:\n  proto: {proto_line.strip()}\n  pred: {pred_line.strip()}")
            continue

        _, true_label = proto_parts
        _, pred_score = pred_parts

        # True label: 1 for live, 0 for attack
        is_live = 1 if true_label == "0_0_0" else 0
        labels.append(is_live)
        preds.append(float(pred_score))

    if not preds or not labels:
        raise ValueError("No valid predictions or labels found.")

    preds = np.array(preds)
    labels = np.array(labels)

    # AUC
    auc = roc_auc_score(labels, preds)

    # APCER, BPCER, ACER
    fpr, tpr, thresholds = roc_curve(labels, preds)
    best_thresh_idx = np.argmax(tpr - fpr)  # Youden's J
    threshold = thresholds[best_thresh_idx]
    predicted_labels = (preds >= threshold).astype(int)

    live_mask = labels == 1
    attack_mask = labels == 0

    bpcer = 1 - (predicted_labels[live_mask] == 1).mean()
    apcer = 1 - (predicted_labels[attack_mask] == 0).mean()
    acer = (bpcer + apcer) / 2

    return {
        "AUC": round(auc, 4),
        "Threshold": round(threshold, 4),
        "BPCER": round(bpcer * 100, 2),
        "APCER": round(apcer * 100, 2),
        "ACER": round(acer * 100, 2)
    }
