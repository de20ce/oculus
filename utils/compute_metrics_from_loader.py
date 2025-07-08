from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

# === Metric function on val_loader ===
def compute_metrics_from_loader(model, val_loader, device):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for images, group_ids, attack_ids, is_live_flags, group_keys in val_loader:
            images = images.to(device)
            is_live_flags = is_live_flags.to(device)

            _, _, live_scores = model(images)

            preds.extend(live_scores.cpu().numpy())
            labels.extend(is_live_flags.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    auc = roc_auc_score(labels, preds)
    fpr, tpr, thresholds = roc_curve(labels, preds)
    best_thresh_idx = np.argmax(tpr - fpr)
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
    }, acer