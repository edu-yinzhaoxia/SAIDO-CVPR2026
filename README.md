# SAIDO

[Paper](https://arxiv.org/abs/2512.00539)


CVPR 2026

This project proposes SAIDO, a continual learning framework for AI-generated image detection. It constructs a scene-aware adaptive expert architecture and designs an importance-constrained optimization mechanism to effectively balance stability and plasticity, thereby mitigating cross-scene distribution shifts and catastrophic forgetting. The framework significantly enhances detection generalization across multiple generative models and open-world environments.


## 🗂️ Project Structure

```
SAIDO/
├── make_annotations.py               # Scene classification + label generation (train/val/test .txt)
├── training.py                       # Main training entry
├── get_config.py                     # YAML config loader
├── load_dataset.py                   # Dataset + continual scenario construction
├── get_images_features.py            # CLIP & caption feature utilities
├── parse_log_to_result.py            # Log parsing helper
├── avalanche/                        # (Customized) Avalanche-style CL framework
│   ├── models/              
│   │   └── SAIDO_CLIP.py             # SAIDO multi-scene model (HF online loading supported)
│   ├── training/
│   │   ├── strategies/               # Strategies (BaseStrategy, wrappers, validation, etc.)
│   │   └── plugins/                  # Plugins (LoRA/RegO/...)
│   ├── benchmarks/                   # Scenarios, dataset wrappers, loaders
│   └── evaluation/                   # Metrics
└── dataset/                           # Data directory (user-provided)
    ├── real/
    ├── fake/
    └── captions/                      # Generated txt labels
```

## 🗂️ Environments

Recommended:

- Python 3.10
- PyTorch 2.7.1+cu126
- `transformers`, `peft`, `torchvision`, `avalanche-lib`

If you use Qwen-VL for classification in `make_annotations.py`, prepare:

- api_key
- Qwen-compatible base URL

### 1. Create environment

conda create -n saido python=3.10
conda activate saido

### 2. Install dependencies

pip install -r requirements.txt

## 🗂️ Framework

![Framework](framework.png)

The SAIDO framework consists of two key components: the Scene-Aware Expert Module (SAEM) and the Importance-Guided Dynamic Optimization mechanism (IDOM).

First, a Vision-Language Model identifies the scene of the input image and routes it to the corresponding scene-specific expert. Each scene is associated with a lightweight LoRA module that works alongside a shared CLIP backbone, forming a “shared representation + scene-specialized adaptation” architecture. This design reduces cross-scene distribution shifts and enables controlled scene expansion in open-world environments.

During continual learning, IDOM estimates neuron importance and performs constrained gradient updates. Neurons with high importance primarily preserve previously learned knowledge, while less important neurons adapt to new tasks. This dynamic regulation achieves an effective balance between stability and plasticity, mitigating catastrophic forgetting.

By integrating scene-aware modeling with importance-guided optimization, SAIDO significantly improves generalization and long-term robustness under multi-generator continual learning settings.

## 🔁 Continual Learning Protocol

- Each generator is treated as one task.
- Tasks are trained sequentially.
- We report:
  - Average Accuracy
  - Average Forgetting

## 🚀 Quick Start

### Step 1: Generate labels

```bash
python make_annotations.py --use_qwen --images_dir ./dataset --save_path ./dataset/captions --train_ratio 0.8 --val_ratio 0.05 --min_pair 50
```

### Step 2: Train

```bash
python training.py --yaml mytrain.yaml
```

## 📚 Citation

If you find this project useful, please cite:

```bibtex
@article{hu2025saido,
  title={SAIDO: Generalizable Detection of AI-Generated Images via Scene-Aware and Importance-Guided Dynamic Optimization in Continual Learning},
  author={Hu, Yongkang and Cheng, Yu and Zhang, Yushuo and Xie, Yuan and Yin, Zhaoxia},
  journal={arXiv preprint arXiv:2512.00539},
  year={2025}
}
