import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random
from torch.utils.data import Dataset, Subset

class HierarchicalFaceAttackDataset(Dataset):
    def __init__(self, protocol_root_pairs, transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.entries = []
        self.group_label_map = {}
        self.attack_label_map = {}
        self.group_to_attacks = {}
        self.protocol_entries = []

        for proto_file, root_dir in protocol_root_pairs:
            file_entries = []
            with open(proto_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue

                    image_path, label_str = parts
                    parts_label = label_str.split("_")

                    group_key = "0_0" if label_str == "0_0_0" else "_".join(parts_label[:2])

                    if group_key not in self.group_label_map:
                        self.group_label_map[group_key] = len(self.group_label_map)
                        self.group_to_attacks[group_key] = []

                    if label_str not in self.attack_label_map:
                        self.attack_label_map[label_str] = len(self.group_to_attacks[group_key])
                        self.group_to_attacks[group_key].append(label_str)

                    group_id = self.group_label_map[group_key]
                    attack_id = self.attack_label_map[label_str]
                    full_image_path = os.path.join(root_dir, os.path.basename(image_path))

                    entry = {
                        'image': full_image_path,
                        'group_id': group_id,
                        'attack_id': attack_id,
                        'group_key': group_key
                    }
                    file_entries.append(entry)

            self.protocol_entries.append(file_entries)
            self.entries.extend(file_entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image = Image.open(item['image']).convert('RGB')
        image = self.transform(image)
        group_id = item['group_id']
        attack_id = item['attack_id']
        is_live = 1 if item['group_key'] == "0_0" else 0
        return image, group_id, attack_id, is_live, item['group_key']


