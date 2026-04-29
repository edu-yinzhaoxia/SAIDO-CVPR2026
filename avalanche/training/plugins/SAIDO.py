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
    scene_id = None
    if isinstance(batch, (list, tuple)) and len(batch) >= 4:
        scene = batch[3]
        if isinstance(scene, torch.Tensor):
            uniq = torch.unique(scene)
            if len(uniq) == 1:
                scene_id = str(int(uniq.item()))
    if scene_id is None:
        print("can't find scene")
        scene_id = model._active_scene if model._active_scene is not None else "0"
    return scene_id


def _is_lora_param_name(n: str):
    return "lora_" in n


def _is_shared_head_name(n: str):
    return ("visual_projection.weight" in n) or ("fc." in n and ".weight" in n)


class SAIDOPlugin(StrategyPlugin):
    def __init__(self, ef_thresh=0.1, model=None, importnat_thresh=0.75,
                 target2ti=None, criterion=None, use_ebbinghaus=True,
                 use_grad_scale=True, grad_scale_k=0.5, grad_scale_beta=1.0):
        super().__init__()
        self.ef_thresh = float(ef_thresh)
        self.importnat_thresh = float(importnat_thresh)
        self.target2ti = target2ti or {"0": -1, "1": 1}
        self.criterion = criterion
        self.use_ebbinghaus = use_ebbinghaus
        self.use_grad_scale = bool(use_grad_scale)
        self.grad_scale_k = float(grad_scale_k)
        self.grad_scale_beta = float(grad_scale_beta)

        self.old_grad_shared = {}

        self.old_grad_scene = defaultdict(dict)

        self.scene_real_imp = defaultdict(lambda: defaultdict(dict))  # exp -> scene -> {name:tensor}
        self.scene_fake_imp = defaultdict(lambda: defaultdict(dict))
        self.scene_real_mask = defaultdict(lambda: defaultdict(dict))  # exp -> scene -> {name:int mask}
        self.scene_fake_mask = defaultdict(lambda: defaultdict(dict))

        self.shared_real_imp = defaultdict(dict)  # exp -> {name:tensor}
        self.shared_fake_imp = defaultdict(dict)
        self.shared_real_mask = defaultdict(dict)  # exp -> {name:int mask}
        self.shared_fake_mask = defaultdict(dict)

        self._targets_ready = False
        self._target_param_names = set()
        self._lora_param_names = set()
        self._shared_param_names = set()

    def _init_targets(self, model):
        for n, p in model.named_parameters():
            if _is_lora_param_name(n):
                self._lora_param_names.add(n)
                self._target_param_names.add(n)
            elif _is_shared_head_name(n):
                self._shared_param_names.add(n)
                self._target_param_names.add(n)

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
                x, y, task_label,scene_id,batch_prompts = batch
            else:
                print("data length error")
                x, y = batch
                task_label = None

            x = x.to(device)
            y = y.to(device)
            batch_prompts = {k: v.to(device) for k, v in batch_prompts.items()}
            unique_scenes = torch.unique(scene_id)
            unique_scenes = int(unique_scenes.item())
            real_idx, fake_idx = _split_real_fake(y, self.target2ti, y.device)

            if real_idx.numel() > 0:
                xi = x.index_select(0, real_idx)
                yi = y.index_select(0, real_idx)
                pi = {k: v.index_select(0, real_idx) for k, v in batch_prompts.items()}
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

            if fake_idx.numel() > 0:
                xi = x.index_select(0, fake_idx)
                yi = y.index_select(0, fake_idx)
                pi = {k: v.index_select(0, fake_idx) for k, v in batch_prompts.items()}
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

        for s in scene_real:
            self._normalize_imp(scene_real[s], scene_real_cnt[s])
        for s in scene_fake:
            self._normalize_imp(scene_fake[s], scene_fake_cnt[s])
        self._normalize_imp(shared_real, shared_real_cnt)
        self._normalize_imp(shared_fake, shared_fake_cnt)

        return scene_real, scene_fake, shared_real, shared_fake

    def _quantize_masks(self, imp_dict: dict, q: float, scale=1):
        masks = {}
        for n, t in imp_dict.items():
            if t is None:
                continue
            thr = torch.quantile(t.flatten(), q)
            masks[n] = (t > thr).to(torch.int) * int(scale)
        return masks

    def set_ebbinghaus_forgetting_weight(self, task_num):
        def equation(x, task_num):
            formula = 0
            for i in range(1, task_num + 1):
                formula += np.exp(-i / x)
            return formula - 1

        initial_guess = 1.0
        solution = fsolve(equation, initial_guess, args=(task_num))
        return list(reversed([np.exp(-i / solution[0]) for i in range(1, task_num + 1)]))

    def before_update(self, strategy, **kwargs):
        model = strategy.model  # 你的 CLIPLoRA_MultiScene
        device = strategy.device
        self._init_targets(model)
        fake_batch_idx = []
        real_batch_idx = []
        for idx, s_output in enumerate(strategy.mb_output):
            if self.target2ti[str(int(strategy.mb_y[idx]))] == 1:
                fake_batch_idx.append(idx)
            else:
                real_batch_idx.append(idx)
        real_num = len(real_batch_idx)
        fake_num = len(fake_batch_idx)
        try:
            current_scene = _get_scene_from_batch_or_model(
                (strategy.mb_x, strategy.mb_y, strategy.mb_task_id, strategy.mb_scene_id),
                model
            )
        except Exception:
            print("current_scene error")
            current_scene = model._active_scene or "0"

        cur_exp = strategy.experience.current_experience
        if cur_exp == 0:
            for n, p in model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    if n in self._lora_param_names:
                        self.old_grad_scene[current_scene][n] = p.grad.detach().clone()
                    elif n in self._shared_param_names:
                        self.old_grad_shared[n] = p.grad.detach().clone()
            return

        if self.use_ebbinghaus:
            ebb_weights = self.set_ebbinghaus_forgetting_weight(cur_exp)
        else:
            ebb_weights = [1.0 for _ in range(cur_exp)]

        agg_lora_mask = {}
        agg_shared_mask = {}
        pre_real = {}
        pre_fake = {}
        agg_lora_real_imp = {}
        agg_lora_fake_imp = {}
        agg_shared_real_imp = {}
        agg_shared_fake_imp = {}

        for exp in range(cur_exp):
            w = ebb_weights[exp]
            cr = self.scene_real_mask.get(exp, {}).get(int(current_scene), {})
            cf = self.scene_fake_mask.get(exp, {}).get(int(current_scene), {})
            for n, m in cr.items():
                agg_lora_mask[n] = agg_lora_mask.get(n, torch.zeros_like(m)) + m * w
                pre_real[n] = pre_real.get(n, torch.zeros_like(m)) | m
            for n, m in cf.items():
                agg_lora_mask[n] = agg_lora_mask.get(n, torch.zeros_like(m)) + (m/2) * w
                pre_fake[n] = pre_fake.get(n, torch.zeros_like(m)) | m
            l_real_imp_scene = self.scene_real_imp.get(exp, {}).get(int(current_scene), {})
            l_fake_imp_scene = self.scene_fake_imp.get(exp, {}).get(int(current_scene), {})
            for n, t in l_real_imp_scene.items():
                agg_lora_real_imp[n] = agg_lora_real_imp.get(n, torch.zeros_like(t)) + t * w
            for n, t in l_fake_imp_scene.items():
                agg_lora_fake_imp[n] = agg_lora_fake_imp.get(n, torch.zeros_like(t)) + t * w
            sr = self.shared_real_mask.get(exp, {})
            sf = self.shared_fake_mask.get(exp, {})
            for n, m in sr.items():
                agg_shared_mask[n] = agg_shared_mask.get(n, torch.zeros_like(m)) + m * w
                pre_real[n] = pre_real.get(n, torch.zeros_like(m)) | m
            for n, m in sf.items():
                agg_shared_mask[n] = agg_shared_mask.get(n, torch.zeros_like(m)) + (m/2) * w
                pre_fake[n] = pre_fake.get(n, torch.zeros_like(m)) | m
            s_real_imp = self.shared_real_imp.get(exp, {})
            s_fake_imp = self.shared_fake_imp.get(exp, {})
            for n, t in s_real_imp.items():
                agg_shared_real_imp[n] = agg_shared_real_imp.get(n, torch.zeros_like(t)) + t * w
            for n, t in s_fake_imp.items():
                agg_shared_fake_imp[n] = agg_shared_fake_imp.get(n, torch.zeros_like(t)) + t * w
        for n in list(agg_lora_mask.keys()):
            agg_lora_mask[n] = (agg_lora_mask[n] > self.ef_thresh).to(torch.int)
        for n in list(agg_shared_mask.keys()):
            agg_shared_mask[n] = (agg_shared_mask[n] > self.ef_thresh).to(torch.int)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    if n in self._lora_param_names:
                        forget_gate = agg_lora_mask.get(n, None)
                        real_imp_hist = agg_lora_real_imp.get(n, None)
                        fake_imp_hist = agg_lora_fake_imp.get(n, None)
                    else:
                        forget_gate = agg_shared_mask.get(n, None)
                        real_imp_hist = agg_shared_real_imp.get(n, None)
                        fake_imp_hist = agg_shared_fake_imp.get(n, None)

                    if forget_gate is None:
                        continue

                    pre_r = pre_real.get(n, torch.zeros_like(forget_gate))
                    pre_f = pre_fake.get(n, torch.zeros_like(forget_gate))
                    grad_filter = (pre_r + pre_f) * forget_gate
                    current_grad = p.grad.clone()
                    if n in self._lora_param_names:
                        old_grad = self.old_grad_scene.get(current_scene, {}).get(n, torch.zeros_like(p.grad))
                    else:
                        old_grad = self.old_grad_shared.get(n, torch.zeros_like(p.grad))

                    a_zone = (grad_filter == 0).to(p.grad.dtype)
                    b_zone = (grad_filter != 0).to(p.grad.dtype)

                    new_grad = current_grad * a_zone

                    if (b_zone.sum() > 0) and (old_grad.norm() > 0):
                        dot = torch.dot(current_grad.view(-1), old_grad.view(-1))
                        proj = dot * old_grad / (old_grad.norm() ** 2)
                        proj = proj.view_as(p.grad)
                        ortho = current_grad - proj
                        if (real_imp_hist is not None) or (fake_imp_hist is not None):
                            r_imp = real_imp_hist if real_imp_hist is not None else torch.zeros_like(p.grad)
                            f_imp = fake_imp_hist if fake_imp_hist is not None else torch.zeros_like(p.grad)
                            total_imp = (r_imp + f_imp).clamp_min(1e-12)
                            w_real = (r_imp / total_imp).to(p.grad.dtype)
                            w_fake = (f_imp / total_imp).to(p.grad.dtype)
                        else:
                            w_real = torch.full_like(p.grad, 0.5)
                            w_fake = torch.full_like(p.grad, 0.5)
                        mixed = w_real * proj + w_fake * ortho
                        new_grad = new_grad + mixed * b_zone
                    else:
                        new_grad = new_grad + current_grad * b_zone

                    if self.use_grad_scale:
                        total_imp_hist = None
                        if (real_imp_hist is not None) or (fake_imp_hist is not None):
                            r_imp = real_imp_hist if real_imp_hist is not None else torch.zeros_like(p.grad)
                            f_imp = fake_imp_hist if fake_imp_hist is not None else torch.zeros_like(p.grad)
                            total_imp_hist = (r_imp + f_imp)
                        scale = self._grad_scale_from_importance(total_imp_hist)
                        if scale is not None:
                            new_grad = new_grad * scale.to(new_grad.dtype)

                    p.grad.copy_(new_grad)

                    if n in self._lora_param_names:
                        self.old_grad_scene[current_scene][n] = p.grad.detach().clone()
                    else:
                        self.old_grad_shared[n] = p.grad.detach().clone()

    def after_training_exp(self, strategy, **kwargs):
        model = strategy.model
        device = strategy.device
        bs = strategy.train_mb_size
        dataset = strategy.experience.dataset

        self._init_targets(model)

        scene_real, scene_fake, shared_real, shared_fake = \
            self._compute_importances_sceneaware(model, dataset, device, bs, strategy)

        exp_id = strategy.experience.current_experience

        self.scene_real_imp[exp_id] = scene_real
        self.scene_fake_imp[exp_id] = scene_fake
        self.shared_real_imp[exp_id] = shared_real
        self.shared_fake_imp[exp_id] = shared_fake
        self.scene_real_mask[exp_id] = {}
        self.scene_fake_mask[exp_id] = {}
        for scene_id, imp_dict in scene_real.items():
            self.scene_real_mask[exp_id][scene_id] = self._quantize_masks(imp_dict, self.importnat_thresh, scale=1)
        for scene_id, imp_dict in scene_fake.items():
            self.scene_fake_mask[exp_id][scene_id] = self._quantize_masks(imp_dict, self.importnat_thresh, scale=2)

        self.shared_real_mask[exp_id] = self._quantize_masks(shared_real, self.importnat_thresh, scale=1)
        self.shared_fake_mask[exp_id] = self._quantize_masks(shared_fake, self.importnat_thresh, scale=2)