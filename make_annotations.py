import os
import json
import re
import random
import argparse
import torch
import torch.nn.functional as F
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import warnings
from openai import OpenAI
import base64
from collections import defaultdict
import requests
import hashlib
import shutil
MIN_PAIR = 50  
warnings.filterwarnings('ignore')

from transformers import CLIPModel, CLIPProcessor
from get_images_features import get_clip_model, get_clipcap_model, get_image_features, get_text

error_count = 0

warnings.filterwarnings('ignore')

qwen_client = OpenAI(
    api_key="",   
    base_url="",
)

CLASS_MAPPING = {
    'Activity': 0,
    'Animal': 1,
    'Building': 2,
    'Clothing': 3,
    'Food': 4,
    'Nature': 5,
    'Object': 6,
    'Person': 7,
    'Plant': 8,
    'Vehicle': 9
}

ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
ALL_CLASS_IDS = list(CLASS_MAPPING.values())

MODEL_MAPPING = {
    'ProGAN': 3,
    'SAGAN': 2,
    'BigGAN': 5,
    'ADM': 0,
    'GLIDE': 1,
    'wukong': 6,
    'diffusion1.5': 7,
    'VQDM': 4,
    'midjourney': 8
}

ID_TO_MODEL = {v: k for k, v in MODEL_MAPPING.items()}
ALL_MODEL_IDS = list(MODEL_MAPPING.values())

from openai import BadRequestError

def classify_scene_with_qwen(image_path):
    prompt = (
        f"Please classify the scene in this image.\n"
        f"Available scenes include: {', '.join(CLASS_MAPPING.keys())}.\n"
        f"Do not combine existing scenes, only output one most representative scene name, no explanation.\n"
    )

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        completion = qwen_client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ]
        )
        scene_name = completion.choices[0].message.content.strip()
        print(f'{image_path} : {scene_name}')
        return scene_name

    except BadRequestError as e:
        print(f"Image violates policy or is non-compliant, deleted: {image_path}")
        try:
            os.remove(image_path)
        except Exception as rm_err:
            print(f"Failed to delete: {rm_err}")
        return None

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def move_image_to_category(image_path, scene_name, base_output_dir="./dataset/sorted"):
    target_dir = os.path.join(base_output_dir, scene_name)
    os.makedirs(target_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    new_path = os.path.join(target_dir, filename)

    if os.path.exists(new_path):
        with open(image_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        new_filename = f"{name}_{file_hash}{ext}"
        new_path = os.path.join(target_dir, new_filename)

        counter = 1
        while os.path.exists(new_path):
            new_filename = f"{name}_{file_hash}_{counter}{ext}"
            new_path = os.path.join(target_dir, new_filename)
            counter += 1

    try:
        shutil.move(image_path, new_path)
    except Exception as e:
        print(f"Failed to move file {image_path} -> {new_path}: {e}")
        return image_path

    return new_path

def get_or_add_class_id(scene_name):
    """Dynamically add new category"""
    global CLASS_MAPPING, ID_TO_CLASS, ALL_CLASS_IDS
    if scene_name in CLASS_MAPPING:
        return CLASS_MAPPING[scene_name]
    else:
        new_id = max(CLASS_MAPPING.values()) + 1
        CLASS_MAPPING[scene_name] = new_id
        ID_TO_CLASS[new_id] = scene_name
        ALL_CLASS_IDS.append(new_id)
        print(f"New category added: {scene_name} (ID={new_id})")
        return new_id

def get_image_paths(folder, limit=None, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
    """Recursively get image paths"""
    image_paths = []
    num = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                path = os.path.join(root, f)
                path = os.path.normpath(path).replace("\\", "/")
                image_paths.append(path)
                num += 1
                if limit is not None and num >= limit:
                    return image_paths
    return image_paths

def _tail_for_match(path: str, levels: int = 3) -> str:
    parts = path.replace("\\", "/").lower().split("/")
    if levels <= 0 or levels >= len(parts):
        tail_parts = parts
    else:
        tail_parts = parts[-levels:]
    return "/".join(tail_parts)
def get_content_category_from_path(path):
    path_parts = path.replace("\\", "/").split("/")

    model_index = -1
    for i, part in enumerate(path_parts):
        if part in MODEL_MAPPING:
            model_index = i
            break

    if model_index != -1 and model_index + 1 < len(path_parts):
        category_part = path_parts[model_index + 1]
        for class_name, class_id in CLASS_MAPPING.items():
            if class_name.lower() == category_part.lower():
                return class_id

    return -1


def get_model_type(path):
    path_parts = path.replace("\\", "/").split("/")
    for part in path_parts:
        if part in MODEL_MAPPING:
            return MODEL_MAPPING[part]
    return -1

def split_data_by_category(images, train_ratio=0.7, val_ratio=0.1, min_pair=MIN_PAIR, save_filtered=None):
    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1, "train_ratio and val_ratio must be in (0,1)"
    assert train_ratio + val_ratio < 1, "train_ratio + val_ratio must be less than 1"

    category_model_groups = {cid: {mid: {0: [], 1: []} for mid in ALL_MODEL_IDS} for cid in ALL_CLASS_IDS}
    for img in images:
        path, content_cat, model_type, real_fake = img
        if content_cat in category_model_groups:
            category_model_groups[content_cat][model_type][real_fake].append(img)

    train_set, val_set, test_set = [], [], []
    category_stats = {}
    stats_by_model_cat_sample = {mid: {} for mid in ALL_MODEL_IDS}
    filtered_out_records = []

    for cat_id in tqdm(ALL_CLASS_IDS, desc="Splitting categories"):
        by_model = category_model_groups.get(cat_id, {mid: {0: [], 1: []} for mid in ALL_MODEL_IDS})

        total_real_raw = sum(len(by_model[mid][0]) for mid in ALL_MODEL_IDS)
        total_fake_raw = sum(len(by_model[mid][1]) for mid in ALL_MODEL_IDS)
        raw_pair_count = min(total_real_raw, total_fake_raw)

        for mid in ALL_MODEL_IDS:
            stats_by_model_cat_sample[mid][cat_id] = {'real': 0, 'fake': 0}
        category_stats[cat_id] = {
            'total_pairs': 0, 'train_pairs': 0, 'val_pairs': 0, 'test_pairs': 0,
            'real_count': 0, 'fake_count': 0
        }

        if raw_pair_count < min_pair:
            for mid in ALL_MODEL_IDS:
                r, f = len(by_model[mid][0]), len(by_model[mid][1])
                if r > 0 or f > 0:
                    filtered_out_records.append(
                        f"[GLOBAL] cat={ID_TO_CLASS[cat_id]}, model={ID_TO_MODEL[mid]}, "
                        f"real={r}, fake={f}, pairs={min(r, f)} (<{min_pair})"
                    )
            continue

        for mid in ALL_MODEL_IDS:
            real_m, fake_m = by_model[mid][0], by_model[mid][1]
            model_pair = min(len(real_m), len(fake_m))

            if model_pair < min_pair:
                if len(real_m) > 0 or len(fake_m) > 0:
                    filtered_out_records.append(
                        f"cat={ID_TO_CLASS[cat_id]}, model={ID_TO_MODEL[mid]}, "
                        f"real={len(real_m)}, fake={len(fake_m)}, pairs={model_pair} (<{min_pair})"
                    )
                continue

            
            stats_by_model_cat_sample[mid][cat_id] = {'real': len(real_m), 'fake': len(fake_m)}

            
            random.shuffle(real_m)
            random.shuffle(fake_m)

            
            train_pairs = int(model_pair * train_ratio)
            val_pairs = int(model_pair * val_ratio)
            test_pairs = model_pair - train_pairs - val_pairs

            train_set.extend(real_m[:train_pairs])
            train_set.extend(fake_m[:train_pairs])
            val_set.extend(real_m[train_pairs:train_pairs + val_pairs])
            val_set.extend(fake_m[train_pairs:train_pairs + val_pairs])
            test_set.extend(real_m[train_pairs + val_pairs:train_pairs + val_pairs + test_pairs])
            test_set.extend(fake_m[train_pairs + val_pairs:train_pairs + val_pairs + test_pairs])

            # Update category-level statistics
            category_stats[cat_id]['total_pairs'] += model_pair
            category_stats[cat_id]['train_pairs'] += train_pairs
            category_stats[cat_id]['val_pairs'] += val_pairs
            category_stats[cat_id]['test_pairs'] += test_pairs
            category_stats[cat_id]['real_count'] += len(real_m)
            category_stats[cat_id]['fake_count'] += len(fake_m)

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    
    if save_filtered:
        os.makedirs(os.path.dirname(save_filtered), exist_ok=True)
        with open(save_filtered, "w", encoding="utf-8") as f:
            f.write("\n".join(filtered_out_records))
        print(f"⚠️ Saved filtered category-model records to {save_filtered}")

    return train_set, val_set, test_set, category_stats, stats_by_model_cat_sample


def process_and_save(dataset, save_file, dataset_name, opt,
                     clip_img_model=None, clip_processor=None,
                     clipcap_model=None, tokenizer=None,
                     clip_text_model=None, cls_ids=None, proto_feats=None):
    with open(save_file, "w", encoding="utf-8") as out_f:
        header = "path\tcaption\tscene_id\tscene_name\tmodel_id\treal_fake\n"
        out_f.write(header)

        for path, gt_content, model_type, real_fake in tqdm(dataset, desc=f"保存 {dataset_name}"):
            if opt.shuffle_labels or opt.no_classification:
                caption = "No"
            else:
                image_features = get_image_features(path, clip_img_model, clip_processor, device=opt.device)
                caption = get_text(image_features, tokenizer, clipcap_model, opt.fc_path, False, device=opt.device)

            scene_name = ID_TO_CLASS.get(gt_content, f"Unknown {gt_content}")
            line = f"{path}\t{caption}\t{gt_content}\t{scene_name}\t{model_type}\t{real_fake}\n"
            out_f.write(line)

    print(f"{dataset_name} set saved to {save_file}")

def visualize_model_category_distribution(stats_by_model_cat, save_path, show_plot=False):
    """
    stats_by_model_cat[model][cat] = {'train_pairs': X, 'test_pairs': Y}
    """
    for mid, cat_stats in stats_by_model_cat.items():
        model_name = ID_TO_MODEL[mid]
        categories = [ID_TO_CLASS[cid] for cid in cat_stats.keys()]
        train_vals = [v['train_pairs'] for v in cat_stats.values()]
        test_vals = [v['test_pairs'] for v in cat_stats.values()]

        x = range(len(categories))
        width = 0.35
        plt.figure(figsize=(12, 6))
        plt.bar([i - width / 2 for i in x], train_vals, width, label='Train Pairs')
        plt.bar([i + width / 2 for i in x], test_vals, width, label='Test Pairs')

        plt.title(f'{model_name} - Category Distribution')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.ylabel("Number of Pairs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{model_name}_category_distribution.png"), dpi=300)
        if show_plot:
            plt.show()
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced Caption + CLIP Classification with Visualization')

    parser.add_argument('--model_path', type=str, default='https://www.now61.com/f/Xljmi0/coco_prefix_latest.pt')
    parser.add_argument('--fc_path', type=str, default='https://www.now61.com/f/qwvoH5/fc_parameters.pth')

    parser.add_argument('--images_dir', type=str, default="./dataset",
                        help='Image folder (contains real and fake)')
    parser.add_argument('--save_path', type=str, default='./dataset/captions',
                        help='Result save path')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (None means unlimited)')

    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help="Training set ratio (0-1)")
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help="Validation set ratio (0-1, train_ratio + val_ratio < 1)")
    parser.add_argument("--min_pair", type=int, default=50,
                        help="Minimum real/fake pairs per category, categories below this value will not be written to dataset")

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computing device (cuda or cpu)')
    parser.add_argument('--text_on_cpu', action="store_true",
                        help="Put text encoding on CPU to save GPU memory")

    parser.add_argument('--show_plots', action="store_true",
                        help="Whether to display visualization plots (default: only save, not display)")
    parser.add_argument('--no_visualization', action="store_true",
                        help="Do not save visualization plots")
    parser.add_argument('--no_classification', action="store_true",
                        help="Do not load classification model, only do data splitting and shuffling")
    parser.add_argument('--shuffle_labels', action="store_true",
                        help="Whether to randomly shuffle category labels (caption=No, pred=random category)")

    parser.add_argument('--use_qwen', action="store_true",
                        help="Whether to use Qwen API for scene classification of images and move images")
    parser.add_argument('--sorted_dir', type=str, default=None,
                        help="Base directory for moving images when Qwen classification is enabled (default: create under model directory)")

    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_args()
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(opt.save_path, exist_ok=True)

    image_list = []

    if opt.use_qwen:
        print("Using Qwen API for scene classification and moving images")
        for real_fake in ['real', 'fake']:
            real_fake_flag = 0 if real_fake == 'real' else 1
            base_dir = os.path.join(opt.images_dir, real_fake)
            for model_name in MODEL_MAPPING.keys():
                model_dir = os.path.join(base_dir, model_name)
                model_images = get_image_paths(model_dir, limit=opt.max_images)

                for p in tqdm(model_images, desc=f"Processing {model_name}/{real_fake}"):
                    scene_name = classify_scene_with_qwen(p)
                    if scene_name is None:
                        continue

                    scene_id = get_or_add_class_id(scene_name)

                    move_dir = opt.sorted_dir if opt.sorted_dir else model_dir
                    new_path = move_image_to_category(p, scene_name, base_output_dir=move_dir)

                    image_list.append((new_path, scene_id, MODEL_MAPPING[model_name], real_fake_flag))

        print(f"\nCollected {len(image_list)} valid images")

    else:
        print("⚡ Not using API, only reading images from existing folders and trying to parse categories")
        for real_fake in ['real', 'fake']:
            real_fake_flag = 0 if real_fake == 'real' else 1
            base_dir = os.path.join(opt.images_dir, real_fake)
            for model_name in MODEL_MAPPING.keys():
                model_dir = os.path.join(base_dir, model_name)
                model_images = get_image_paths(model_dir, limit=opt.max_images)

                for p in tqdm(model_images, desc=f"Collecting {model_name}/{real_fake}"):
                    scene_id = get_content_category_from_path(p)
                    if scene_id == -1:
                        continue 
                    image_list.append((p, scene_id, MODEL_MAPPING[model_name], real_fake_flag))

        print(f"\nCollected {len(image_list)} images (based on existing folder classification)")

    filtered_txt = os.path.join(opt.save_path, "filtered_out.txt")
    train_set, val_set, test_set, category_stats, stats_by_model_cat_sample = split_data_by_category(
        image_list, train_ratio=opt.train_ratio,val_ratio=opt.val_ratio, min_pair=opt.min_pair, save_filtered=filtered_txt
    )
    stats_by_model_cat = {mid: {} for mid in ALL_MODEL_IDS}
    for mid in ALL_MODEL_IDS:
        for cat_id in ALL_CLASS_IDS:
            counts = stats_by_model_cat_sample[mid].get(cat_id, {'real': 0, 'fake': 0})
            pairs_total = min(counts['real'], counts['fake'])
            if pairs_total >= opt.min_pair:
                train_pairs = int(pairs_total * opt.train_ratio)
                val_pairs = int(pairs_total * 0.1)  
                test_pairs = pairs_total - train_pairs - val_pairs
            else:
                train_pairs = val_pairs = test_pairs = 0
            stats_by_model_cat[mid][cat_id] = {
                'train_pairs': train_pairs,
                'val_pairs': val_pairs,
                'test_pairs': test_pairs
            }

    if not opt.no_visualization:
        visualize_model_category_distribution(stats_by_model_cat, opt.save_path, opt.show_plots)

    for mid, cat_stats in stats_by_model_cat_sample.items():
        model_name = ID_TO_MODEL[mid]
        print(f"\n=== model: {model_name} ===")
        for cat_id in ALL_CLASS_IDS:
            cat_name = ID_TO_CLASS.get(cat_id, f"Unknown {cat_id}")
            counts = cat_stats.get(cat_id, {'real': 0, 'fake': 0})
            total_pairs = min(counts.get('real', 0), counts.get('fake', 0))
            if total_pairs >= opt.min_pair:
                train_pairs = int(total_pairs * opt.train_ratio)
                val_pairs = int(total_pairs * 0.1)
                test_pairs = total_pairs - train_pairs - val_pairs
            else:
                train_pairs = val_pairs = test_pairs = 0
            print(f"Category {cat_name}: real/fake = {counts.get('real', 0)}/{counts.get('fake', 0)}, "
                  f"[train={train_pairs}, val={val_pairs}, test={test_pairs}]")
    clip_img_model, clip_processor = get_clip_model(clip_name='openai/clip-vit-large-patch14', device=device)
    clipcap_model, tokenizer = get_clipcap_model(opt.model_path, device=device)
    process_and_save(
        train_set, os.path.join(opt.save_path, "train_FLUX.txt"), "Training", opt,
        clip_img_model, clip_processor, clipcap_model, tokenizer,
    )
    process_and_save(
        test_set, os.path.join(opt.save_path, "test_FLUX.txt"), "Test", opt,
        clip_img_model, clip_processor, clipcap_model, tokenizer,
    )
    process_and_save(
        val_set, os.path.join(opt.save_path, "val_FLUX.txt"), "Validation", opt,
        clip_img_model, clip_processor, clipcap_model, tokenizer,
    )

