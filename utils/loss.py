import torch.nn.functional as F

def hierarchical_live_loss(group_logits, attack_logits, live_scores, group_labels, attack_labels, is_live_flags, group_keys, group_to_attacks):
    group_loss = F.cross_entropy(group_logits, group_labels)
    attack_loss = 0
    for i in range(group_labels.size(0)):
        group_key = group_keys[i]
        num_attacks = len(group_to_attacks[group_key])
        attack_logit = attack_logits[i, :num_attacks]
        attack_loss += F.cross_entropy(attack_logit.unsqueeze(0), attack_labels[i].unsqueeze(0))
    binary_loss = F.binary_cross_entropy(live_scores, is_live_flags.float())
    return group_loss + attack_loss / group_labels.size(0) + binary_loss
