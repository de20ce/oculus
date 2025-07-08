import os
import torch
import pickle  # For catching unpickling errors
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from datasets.hierarchical_face_attack_dataset import HierarchicalFaceAttackDataset
from datasets.split_protocol import split_protocol_dataset
from nets.vit_hlc import ViTHierarchicalLiveClassifier
from utils.compute_metrics_from_loader import compute_metrics_from_loader
from utils.loss import hierarchical_live_loss

# === Dataset Paths ===

protocol_root_pairs = [
    ("data/UniAttackData+/Protocol-train.txt",
     "data/UniAttackData+/Data-train"),

    ("data/UniAttackData+/Protocol-val.txt",
     "data/UniAttackData+/Data-val")
]

# === Load full dataset ===
base_dataset = HierarchicalFaceAttackDataset(protocol_root_pairs)

# === Apply balanced 70/30 protocol-wise splitting ===
train_set, val_set = split_protocol_dataset(base_dataset, base_dataset.protocol_entries, train_ratio=0.7)

# === Define Transforms ===
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Inject transforms ===
train_set.dataset.transform = train_transform
val_set.dataset.transform = val_transform

# === DataLoaders ===
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")


# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTHierarchicalLiveClassifier(
    vit_model_name="deit_base_patch16_224", # beit_base_patch16_224, deit_base_patch16_224, convnext_base
    group_to_attacks=base_dataset.group_to_attacks,
    group_label_map=base_dataset.group_label_map
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

start_epoch = 0
best_val_acer = float('inf')

checkpoint_path = "data/UniAttackData+/resume_checkpoint_final.pth"
best_model_path = "data/UniAttackData+/best_model_final.pth"

# === Resume if checkpoint exists ===
if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acer = checkpoint['best_val_acer']
        print(f"Resumed from checkpoint at epoch {start_epoch-1} (Best ACER: {best_val_acer*100:.2f}%)")
    except (pickle.UnpicklingError, RuntimeError) as e:
        print(f"Failed to load checkpoint: {e}\n Starting fresh training.")

epochs = start_epoch + 11

# === Training Loop ===
for epoch in range(start_epoch, epochs):
    model.train()
    total_loss = 0

    for batch_idx, (images, group_ids, attack_ids, is_live_flags, group_keys) in enumerate(train_loader):
        images = images.to(device)
        group_ids = group_ids.to(device)
        attack_ids = attack_ids.to(device)
        is_live_flags = is_live_flags.to(device)

        optimizer.zero_grad()
        group_logits, attack_logits, live_scores = model(images)

        loss = hierarchical_live_loss(
            group_logits, attack_logits, live_scores,
            group_ids, attack_ids, is_live_flags,
            group_keys, base_dataset.group_to_attacks
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}] | Avg Train Loss: {avg_train_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_images, val_group_ids, val_attack_ids, val_is_live_flags, val_group_keys in val_loader:
            val_images = val_images.to(device)
            val_group_ids = val_group_ids.to(device)
            val_attack_ids = val_attack_ids.to(device)
            val_is_live_flags = val_is_live_flags.to(device)

            loss = hierarchical_live_loss(
                *model(val_images),
                val_group_ids, val_attack_ids, val_is_live_flags,
                val_group_keys, base_dataset.group_to_attacks
            )
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    val_metrics, acer = compute_metrics_from_loader(model, val_loader, device)
    print("Validation Metrics:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value}")

    # === Save best model ===
    if acer < best_val_acer:
        best_val_acer = acer
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved: {best_model_path} (ACER: {best_val_acer*100:.2f}%)")

    # === Save checkpoint ===
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_acer': best_val_acer
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

print("Training complete.")
