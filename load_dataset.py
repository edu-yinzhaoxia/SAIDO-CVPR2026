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

def apply_jpeg_compression_pil(img_pil: Image.Image, quality: int = 50) -> Image.Image:
    """Save+reload via PIL JPEG to simulate compression artifacts (quality 1-100)."""
    bio = io.BytesIO()
    img_pil.save(bio, format='JPEG', quality=int(quality))
    bio.seek(0)
    return Image.open(bio).convert('RGB')

def add_gaussian_noise_pil(img_pil: Image.Image, sigma: float = 4.0) -> Image.Image:
    """Add Gaussian noise with std = sigma (assumes pixel range 0-255)."""
    arr = np.asarray(img_pil).astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=float(sigma), size=arr.shape).astype(np.float32)
    arr_noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_noisy)

def up_down_sample_pil(img_pil: Image.Image) -> Image.Image:
    """Downsample by 2 (nearest) then upsample back to original size (nearest)."""
    w, h = img_pil.size
    w2, h2 = max(1, w // 2), max(1, h // 2)
    small = img_pil.resize((w2, h2), resample=Image.NEAREST)
    up = small.resize((w, h), resample=Image.NEAREST)
    return up

def jpeg_compress_pil(img, quality=50):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def add_gaussian_noise(img, mean=0, std=0.05):
    np_img = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(mean, std, np_img.shape)
    np_img = np.clip(np_img + noise, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

class RobustMixAugment:
    def __init__(self, p_jpeg=0.3, p_noise=0.3, jpeg_quality=50, noise_std=0.05):
        self.p_jpeg = p_jpeg
        self.p_noise = p_noise
        self.jpeg_quality = jpeg_quality
        self.noise_std = noise_std

    def __call__(self, img):
        
        if random.random() < self.p_jpeg:
            img = jpeg_compress_pil(img, quality=self.jpeg_quality)
        
        if random.random() < self.p_noise:
            img = add_gaussian_noise(img, std=self.noise_std)
        return img

class LoRADataset(Dataset):
    """
    Load data from .txt file:
    Each line format: path, caption, scene_id, pred_scene_id, model_label, realfake_label
    Returns: (img, label, model_label, scene_id, prompt)
    """
    def __init__(self, data_txt_path, stage='train', transform=None, robust_type=None):
        self.samples = []
        self.model_targets = []
        self.realfake_targets = []
        self.scene_ids = []
        self.prompts = []
        self.stage = stage
        self.transform = transform
        self.robust_type = robust_type
        #self.robust_mode = robust_mode

        with open(data_txt_path, "r", encoding="utf-8") as f:
            f.readline()  
            for line in f:
                path, caption,scene_id,_ , model_label, realfake_label = line.strip().split("\t")
                path = path.replace("\\", "/")
                path = os.path.normpath(path)
                
                path = path.replace("\\", "/")
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

        print(f"[LoRADataset] {stage} 集合加载完成: {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        try:
            img_path = self.samples[index]
            if not isinstance(img_path, str):
                raise ValueError(f"Invalid path type: {type(img_path)}")
            
            img_path = img_path.replace("\\", "/")
            img_path = os.path.normpath(img_path)
            img_path = img_path.replace("\\", "/")
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            img = Image.open(img_path).convert("RGB")
            
            if self.stage == "test" and self.robust_type is not None:
                if self.robust_type == "jpeg":
                    img = apply_jpeg_compression_pil(img, quality=50)
                elif self.robust_type == "gaussian":
                    img = add_gaussian_noise_pil(img, sigma=4.0)
                elif self.robust_type == "downsample":
                    img = up_down_sample_pil(img)
            #if self.transform:
                #img = self.transform(img)
            label = self.realfake_targets[index]
            model_label = self.model_targets[index]
            scene_id = self.scene_ids[index]
            prompt = self.prompts[index]

            return img, label, model_label, scene_id, prompt
        except Exception as e:
            print(f"[ERROR] Failed to open image at index={index}, path={self.samples[index]}, error={e}")
            raise RuntimeError(f"Failed to load image at index {index}: {e}") from e

class LoRASubset(Dataset):
    """
    Subset for a specific generative model type (model_label),
    returns (img, label, task_label, scene_id, prompt).
    """
    def __init__(self, dataset, indices, model_label):
        self.dataset = Subset(dataset, indices)
        self.indices = indices
        self.model_label = model_label

        self.scene_ids = dataset.scene_ids[indices]
        self.model_targets = dataset.model_targets[indices]
        self.realfake_targets = dataset.realfake_targets[indices]
        self.prompts = [dataset.prompts[i] for i in indices]
        self.targets = self.realfake_targets

        print(f"[LoRASubset] Construction complete: model={model_label}, subset size={len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        underlying_idx = self.indices[idx]
        item = self.dataset.dataset[underlying_idx]  
        img, label, _, scene_id, prompt = item
        task_label = self.model_label
        return img, label, task_label, scene_id, prompt


def get_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    '''
    train_transform = transforms.Compose([
        #RobustMixAugment(p_jpeg=0.5, p_noise=0.5, jpeg_quality=40, noise_std=0.03),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandAugment(num_ops=2, magnitude=9),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    '''
    train_transform = transforms.Compose([
        RobustMixAugment(p_jpeg=0.5, p_noise=0.5, jpeg_quality=50, noise_std=0.0157),
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
    robust_jpeg_ds = LoRADataset(args.test_txt, stage="test", transform=test_tf, robust_type="jpeg")
    robust_gauss_ds = LoRADataset(args.test_txt, stage="test", transform=test_tf, robust_type="gaussian")
    robust_down_ds = LoRADataset(args.test_txt, stage="test", transform=test_tf, robust_type="downsample")
    n_models = args.timestamp

    list_train, list_val, list_test = [], [], []
    jpeg_streams = []
    gaussian_streams = []
    downsample_streams = []


    for m in range(n_models):
        train_idx = torch.where(torch.tensor(train_ds.model_targets) == m)[0]
        test_idx = torch.where(torch.tensor(test_ds_clean.model_targets) == m)[0]
        val_indices = []
        for task in range(m + 1):  
            task_indices = torch.where(torch.tensor(val_ds.model_targets) == task)[0]
            val_indices.append(task_indices)
        if len(train_idx) > 0:
            train_sub = LoRASubset(train_ds, train_idx, m)
            sample = train_sub[0]
            print("Sample structure train_sub:", sample)
            print("Sample length:", len(sample))
            train_set = AvalancheDataset(train_sub, task_labels=m)
            sample = train_sub[0]
            print("Sample structure train_set:", sample)
            print("Sample length:", len(sample))
            list_train.append(train_set)

        if len(test_idx) > 0:
            test_sub = LoRASubset(test_ds_clean, test_idx, m)
            test_set = AvalancheDataset(test_sub, task_labels=m)
            list_test.append(test_set)
            jpeg_sub = LoRASubset(robust_jpeg_ds, test_idx, m)
            gauss_sub = LoRASubset(robust_gauss_ds, test_idx, m)
            down_sub = LoRASubset(robust_down_ds, test_idx, m)

            jpeg_streams.append(AvalancheDataset(jpeg_sub, task_labels=m))
            gaussian_streams.append(AvalancheDataset(gauss_sub, task_labels=m))
            downsample_streams.append(AvalancheDataset(down_sub, task_labels=m))

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

    jpeg_scenario = dataset_benchmark(
        train_datasets=list_train,
        test_datasets=jpeg_streams,
        val_datasets=list_val,
        train_transform=train_tf,
        eval_transform = test_tf
    )
    gaussian_scenario = dataset_benchmark(
        train_datasets=list_train,
        val_datasets=list_val,
        train_transform=train_tf,
        test_datasets=gaussian_streams,
        eval_transform=test_tf
    )
    downsample_scenario = dataset_benchmark(
        train_datasets=list_train,
        val_datasets=list_val,
        train_transform=train_tf,
        test_datasets=downsample_streams,
        eval_transform=test_tf
    )


    return scenario, jpeg_scenario, gaussian_scenario, downsample_scenario

