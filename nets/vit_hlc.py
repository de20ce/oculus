from cdc import CentralDifferenceConv2d
from frequency import FrequencyBranch
from se_residual import SEBlockResidual, timm, nn, torch

# Final Improved Model
class ViTHierarchicalLiveClassifier(nn.Module):
    def __init__(self, vit_model_name, group_to_attacks, group_label_map, apply_attention_last=True):
        super().__init__()
        self.apply_attention_last = apply_attention_last

        # CDC Frontend
        self.cdc_block = nn.Sequential(
            CentralDifferenceConv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1)
        )

        # ViT Backbone
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        self.vit.head = nn.Identity()
        feature_dim = self.vit.num_features
        self.projector = nn.Linear(feature_dim, 512)

        # Frequency branch
        self.freq_branch = FrequencyBranch(out_features=512)

        # Fusion head after concatenation
        self.fusion_head = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Added post-fusion SE attention block
        self.post_fusion_attention = SEBlockResidual(512)

        self.group_classifier = nn.Linear(512, len(group_to_attacks))
        self.index_to_group_key = {v: k for k, v in group_label_map.items()}

        self.attack_classifiers = nn.ModuleDict()
        for group_key, attack_list in group_to_attacks.items():
            self.attack_classifiers[group_key] = nn.Linear(512, len(attack_list))

        self.binary_classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = self.cdc_block(x)
        spatial_features = self.vit(x)
        spatial_features = self.projector(spatial_features)

        freq_features = self.freq_branch(x)

        fused_features = torch.cat([spatial_features, freq_features], dim=1)
        fused_features = self.fusion_head(fused_features)

        # Apply post-fusion attention here
        fused_features = self.post_fusion_attention(fused_features)

        group_logits = self.group_classifier(fused_features)
        group_probs = F.softmax(group_logits, dim=1)
        predicted_group = torch.argmax(group_probs, dim=1)

        attack_logits = torch.zeros(
            x.size(0),
            max(classifier.out_features for classifier in self.attack_classifiers.values()),
            device=x.device
        )

        for i in range(x.size(0)):
            group_key = self.index_to_group_key[predicted_group[i].item()]
            logits = self.attack_classifiers[group_key](fused_features[i].unsqueeze(0))
            attack_logits[i, :logits.size(1)] = logits

        live_logits = self.binary_classifier(fused_features)
        live_scores = torch.sigmoid(live_logits).squeeze(1)

        return group_logits, attack_logits, live_scores

