import os
import numpy as np
from torch.utils.data import Dataset, Subset
from PIL import Image
import torch
import torchvision.transforms as transforms
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import dataset_benchmark
import random
from torch.utils.data import DataLoader
import io
from torchvision.transforms import RandAugment

class LoRADataset(Dataset):
    
    def __init__(self, data_txt_path, stage='train', transform=None, robust_type=None):
        self.samples = []
        self.model_targets = []
        self.realfake_targets = []
        self.scene_ids = []
        self.prompts = []
        self.stage = stage
        self.transform = transform
        self.robust_type = robust_type

        with open(data_txt_path, "r", encoding="utf-8") as f:
            f.readline()  
            for line in f:
                path, caption, scene_id,_ , model_label, realfake_label = line.strip().split("\t")
                self.samples.append(path)
                self.model_targets.append(int(model_label))
                self.realfake_targets.append(int(realfake_label))
                self.scene_ids.append(int(scene_id))

                content_prompt = caption.strip()

                scene_prompts_map = {
                    0: "an activity or action",
                    1: "a type of animal",
                    2: "a kind of building or structure",
                    3: "a piece of clothing",
                    4: "a type of food or dish",
                    5: "a natural scene or landscape",
                    6: "a common object",
                    7: "a person or human figure",
                    8: "a kind of plant",
                    9: "a type of vehicle",
                }
                scene_prompt = scene_prompts_map.get(int(scene_id), f"{scene_id}")

                real_fake_prompt = "Camera" if int(realfake_label) == 0 else "Deepfake"

                full_prompt = f"{real_fake_prompt},{scene_prompt},{content_prompt}"
                self.prompts.append(full_prompt)

        self.scene_ids = np.array(self.scene_ids)
        self.model_targets = np.array(self.model_targets)
        self.realfake_targets = np.array(self.realfake_targets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        try:
            img_path = self.samples[index]
            img = Image.open(img_path).convert("RGB")
            
            label = self.realfake_targets[index]
            model_label = self.model_targets[index]
            scene_id = self.scene_ids[index]
            prompt = self.prompts[index]

            return img, label, model_label, scene_id, prompt
        except Exception as e:
            print(f"[ERROR] Failed to open image at index={index}, path={self.samples[index]}, error={e}")
            return None

        label = self.realfake_targets[index]  
        model_label = self.model_targets[index]  
        scene_id = self.scene_ids[index]       
        prompt = self.prompts[index]           

        return img, label, model_label, scene_id, prompt

class LoRASubset(Dataset):
    
    def __init__(self, dataset, indices, model_label):
        self.dataset = Subset(dataset, indices)
        self.indices = indices
        self.model_label = model_label

        self.scene_ids = dataset.scene_ids[indices]
        self.model_targets = dataset.model_targets[indices]
        self.realfake_targets = dataset.realfake_targets[indices]
        self.prompts = [dataset.prompts[i] for i in indices]
        self.targets = self.realfake_targets

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        underlying_idx = self.indices[idx]
        item = self.dataset.dataset[underlying_idx]  
        img, label, _, scene_id, prompt = item
        task_label = self.model_label
        return img, label, task_label, scene_id, prompt


def get_transforms():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform, test_transform

def get_lora_scene_scenario(args):
    train_tf, val_tf, test_tf = get_transforms()
    train_ds = LoRADataset(args.train_txt, stage="train", transform=train_tf)
    val_ds = LoRADataset(args.val_txt, stage="val", transform=val_tf)
    test_ds_clean = LoRADataset(args.test_txt, stage="test", transform=test_tf)
    n_models = args.timestamp

    list_train, list_val, list_test = [], [], []

    for m in range(n_models):
        train_idx = torch.where(torch.tensor(train_ds.model_targets) == m)[0]
        test_idx = torch.where(torch.tensor(test_ds_clean.model_targets) == m)[0]
        
        val_indices = []
        for task in range(m + 1):  
            task_indices = torch.where(torch.tensor(val_ds.model_targets) == task)[0]
            val_indices.append(task_indices)
        if len(train_idx) > 0:
            train_sub = LoRASubset(train_ds, train_idx, m) 
            train_set = AvalancheDataset(train_sub, task_labels=m)
            list_train.append(train_set)

        if len(test_idx) > 0:
            test_sub = LoRASubset(test_ds_clean, test_idx, m)
            test_set = AvalancheDataset(test_sub, task_labels=m)
            list_test.append(test_set)
        if val_indices:  
            val_idx = torch.cat(val_indices, dim=0)
            if len(val_idx) > 0:
                val_sub = LoRASubset(val_ds, val_idx, m)  
                val_set = AvalancheDataset(val_sub, task_labels=m)
                list_val.append(val_set)

    scenario = dataset_benchmark(
        train_datasets=list_train,
        test_datasets=list_test,
        val_datasets=list_val,
        train_transform=train_tf,
        eval_transform=test_tf,
    )
    return scenario