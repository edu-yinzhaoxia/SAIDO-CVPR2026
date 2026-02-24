import copy
from collections import defaultdict
import torch
import math
import random
from avalanche.models import avalanche_forward, MultiTaskModule
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from torch.utils.data import DataLoader
import collections
import numpy as np
from torch.nn.functional import cosine_similarity
from scipy.optimize import fsolve
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader, SceneGroupedTaskBalancedDataLoader


def _split_real_fake(y, target2ti, device):
    # y: LongTensor
    # target2ti: dict[str]->{-1,1}
    max_c = int(y.max().item())
    mapping = torch.full((max_c + 1,), 0, dtype=torch.int64, device=device)
    for k, v in target2ti.items():
        ki = int(k)
        if ki <= max_c:
            mapping[ki] = v
    mapped = mapping[y]
    real_idx = (mapped == -1).nonzero(as_tuple=False).view(-1)
    fake_idx = (mapped == 1).nonzero(as_tuple=False).view(-1)
    return real_idx, fake_idx


def _get_scene_from_batch_or_model(batch, model):
    """
    取当前 batch 的 scene_id（要求 batch 同一场景），
    若 batch 无 scene，退回到 model._active_scene。
    """
    scene_id = None
    if isinstance(batch, (list, tuple)) and len(batch) >= 4:
        scene = batch[3]
        if isinstance(scene, torch.Tensor):
            uniq = torch.unique(scene)
            if len(uniq) == 1:
                scene_id = str(int(uniq.item()))
    if scene_id is None:
        # 回退：使用模型当前激活场景
        scene_id = model._active_scene if getattr(model, "_active_scene", None) is not None else "0"
    return scene_id


def _is_lora_param_name(n: str):
    return "lora_" in n


def _is_shared_head_name(n: str):
    # 你的共享层命名：visual_projection & fc
    return ("visual_projection.weight" in n) or ("fc." in n and ".weight" in n)


class LoRAPlugin(StrategyPlugin):
    """
    修改说明（消融）：该版本的 LoRAPlugin 不对梯度或参数施加任何约束。
    - before_update: 不修改 p.grad，直接返回（允许所有参数自由更新）。
    - after_training_exp: 不计算重要性与掩码（空操作）。
    其它方法保留（但在当前消融设置下不会被使用），以保证接口兼容性。
    """

    def __init__(self, ef_thresh=0.1, model=None, importnat_thresh=0.75,
                 target2ti=None, criterion=None, use_ebbinghaus=True,
                 use_grad_scale=True, grad_scale_k=0.5, grad_scale_beta=1.0):
        super().__init__()
        # 参数保留以兼容外部使用，但在本消融版本中不会生效
        self.ef_thresh = float(ef_thresh)
        self.importnat_thresh = float(importnat_thresh)
        self.target2ti = target2ti or {"0": -1, "1": 1}
        self.criterion = criterion
        self.use_ebbinghaus = use_ebbinghaus
        self.use_grad_scale = bool(use_grad_scale)
        self.grad_scale_k = float(grad_scale_k)
        self.grad_scale_beta = float(grad_scale_beta)

        # 以下成员保持以兼容外部接口 / 防止 AttributeError
        self.old_grad_shared = {}
        self.old_grad_scene = defaultdict(dict)

        self.scene_real_imp = defaultdict(lambda: defaultdict(dict))
        self.scene_fake_imp = defaultdict(lambda: defaultdict(dict))
        self.scene_real_mask = defaultdict(lambda: defaultdict(dict))
        self.scene_fake_mask = defaultdict(lambda: defaultdict(dict))

        self.shared_real_imp = defaultdict(dict)
        self.shared_fake_imp = defaultdict(dict)
        self.shared_real_mask = defaultdict(dict)
        self.shared_fake_mask = defaultdict(dict)

        self._targets_ready = False
        self._target_param_names = set()
        self._lora_param_names = set()
        self._shared_param_names = set()

    def _init_targets(self, model):
        # 枚举参数名（仅做记录，未用于约束）
        if self._targets_ready:
            return
        for n, p in model.named_parameters():
            if _is_lora_param_name(n):
                self._lora_param_names.add(n)
                self._target_param_names.add(n)
            elif _is_shared_head_name(n):
                self._shared_param_names.add(n)
                self._target_param_names.add(n)
        self._targets_ready = True

    @torch.no_grad()
    def _accumulate_imp(self, bucket: dict, name: str, grad_sq: torch.Tensor):
        if name not in bucket:
            bucket[name] = grad_sq.clone()
        else:
            bucket[name] += grad_sq

    def _normalize_imp(self, bucket: dict, count: int):
        if count <= 0:
            return
        for k in bucket:
            bucket[k] /= float(count)

    def _grad_scale_from_importance(self, imp_tensor: torch.Tensor):
        # 此函数保留但在消融设置中不会被调用
        if (imp_tensor is None) or (not self.use_grad_scale):
            return None
        flat = imp_tensor.flatten()
        if flat.numel() == 0:
            return None
        q = torch.quantile(flat, 0.9) if flat.numel() > 1 else (flat.abs().max() + 1e-8)
        denom = (q + 1e-8)
        imp_norm = imp_tensor / denom
        scale = 1.0 / (1.0 + self.grad_scale_k * (imp_norm.clamp_min(0.0) ** self.grad_scale_beta))
        return scale

    def _compute_importances_sceneaware(self, model, dataset, device, batch_size, strategy):
        """
        原始实现用于计算重要性；在本消融版本中保留实现以便未来对比试验使用，
        但如果你希望彻底删去该计算，也可以把这个函数实现为空。
        """
        model.eval()
        scene_real = defaultdict(dict)  # scene -> name->tensor
        scene_fake = defaultdict(dict)
        scene_real_cnt = defaultdict(int)
        scene_fake_cnt = defaultdict(int)
        shared_real = {}
        shared_fake = {}
        shared_real_cnt = 0
        shared_fake_cnt = 0

        tmp_loader = SceneGroupedTaskBalancedDataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in tmp_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                # 兼容你的 batch 格式： (x, y, task_label, scene_id, batch_prompts)
                try:
                    x, y, task_label, scene_id, batch_prompts = batch
                except Exception:
                    # 兜底：若没有 prompts 或 scene 位置不匹配
                    if len(batch) == 4:
                        x, y, task_label, scene_id = batch
                        batch_prompts = {}
                    else:
                        x, y = batch
                        task_label = None
                        scene_id = torch.tensor([0] * x.size(0))
                        batch_prompts = {}
            else:
                x, y = batch
                task_label = None
                scene_id = torch.tensor([0] * x.size(0))
                batch_prompts = {}

            x = x.to(device)
            y = y.to(device)
            # 安全处理 prompts
            if isinstance(batch_prompts, dict):
                batch_prompts = {k: v.to(device) for k, v in batch_prompts.items()}
            else:
                batch_prompts = {}

            unique_scenes = torch.unique(scene_id)
            # 如果 batch 中含多个 scene，我们这里只取第一个作为代表（与原实现假定一致）
            unique_scenes = int(unique_scenes[0].item()) if len(unique_scenes) > 0 else 0

            real_idx, fake_idx = _split_real_fake(y, self.target2ti, y.device)

            # ---- real ----
            if real_idx.numel() > 0:
                xi = x.index_select(0, real_idx)
                yi = y.index_select(0, real_idx)
                pi = {k: v.index_select(0, real_idx) for k, v in batch_prompts.items()} if batch_prompts else {}
                out = avalanche_forward(model, xi, task_label, scene_id, pi)
                loss = strategy.criterion(out, yi.long()) if strategy else self.criterion(out, yi.long())
                model.zero_grad(set_to_none=True)
                loss.backward()

                for n, p in model.named_parameters():
                    if n not in self._target_param_names or p.grad is None:
                        continue
                    g2 = p.grad.detach().pow(2)
                    if n in self._lora_param_names and p.requires_grad:
                        self._accumulate_imp(scene_real[unique_scenes], n, g2)
                    elif n in self._shared_param_names and p.requires_grad:
                        self._accumulate_imp(shared_real, n, g2)
                scene_real_cnt[unique_scenes] += int(real_idx.numel())
                shared_real_cnt += int(real_idx.numel())

            # ---- fake ----
            if fake_idx.numel() > 0:
                xi = x.index_select(0, fake_idx)
                yi = y.index_select(0, fake_idx)
                pi = {k: v.index_select(0, fake_idx) for k, v in batch_prompts.items()} if batch_prompts else {}
                out = avalanche_forward(model, xi, task_label, scene_id, pi)
                loss = strategy.criterion(out, yi.long()) if strategy else self.criterion(out, yi.long())
                model.zero_grad(set_to_none=True)
                loss.backward()

                for n, p in model.named_parameters():
                    if n not in self._target_param_names or p.grad is None:
                        continue
                    g2 = p.grad.detach().pow(2)
                    if n in self._lora_param_names and p.requires_grad:
                        self._accumulate_imp(scene_fake[unique_scenes], n, g2)
                    elif n in self._shared_param_names and p.requires_grad:
                        self._accumulate_imp(shared_fake, n, g2)
                scene_fake_cnt[unique_scenes] += int(fake_idx.numel())
                shared_fake_cnt += int(fake_idx.numel())

        # 归一化
        for s in scene_real:
            self._normalize_imp(scene_real[s], scene_real_cnt[s])
        for s in scene_fake:
            self._normalize_imp(scene_fake[s], scene_fake_cnt[s])
        self._normalize_imp(shared_real, shared_real_cnt)
        self._normalize_imp(shared_fake, shared_fake_cnt)

        return scene_real, scene_fake, shared_real, shared_fake

    # ---------- 量化为掩码 ----------
    def _quantize_masks(self, imp_dict: dict, q: float, scale=1):
        masks = {}
        for n, t in imp_dict.items():
            if t is None:
                continue
            thr = torch.quantile(t.flatten(), q)
            masks[n] = (t > thr).to(torch.int) * int(scale)
        return masks

    def set_ebbinghaus_forgetting_weight(self, task_num):
        # 保留以兼容接口；本消融版本可不使用
        def equation(x, task_num):
            formula = 0
            for i in range(1, task_num + 1):
                formula += np.exp(-i / x)
            return formula - 1

        initial_guess = 1.0
        solution = fsolve(equation, initial_guess, args=(task_num))
        return list(reversed([np.exp(-i / solution[0]) for i in range(1, task_num + 1)]))

    # ---------- 训练期钩子 ----------
    def before_update(self, strategy, **kwargs):
        """
        消融版本：不对梯度做任何过滤或修改，允许所有参数自由更新。
        直接返回，保持训练流程不受该插件约束。
        """
        # 仍然初始化目标集合以防外部代码依赖这些属性
        model = strategy.model
        self._init_targets(model)
        # 不改变任何 p.grad
        return

    def after_training_exp(self, strategy, **kwargs):
        """
        消融版本：不计算/保存重要性或掩码（空操作）。
        保留接口以便外部调用。
        """
        return
