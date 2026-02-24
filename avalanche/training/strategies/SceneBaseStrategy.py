# scene_base_strategy.py
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.models.utils import avalanche_forward
import torch.nn.functional as F
import torch

class SceneBaseStrategy(BaseStrategy):
    @property
    def mb_scene_id(self):
        return self.mbatch[-2]

    def mb_prompts(self):
        return self.mbatch[-1]

    def _unpack_minibatch(self):
        assert len(self.mbatch) >= 5, "Batch should be (x, y, task_id, scene_id, prompts)"
        new_batch = []
        for item in self.mbatch:
            if isinstance(item, torch.Tensor):
                new_batch.append(item.to(self.device))
            else:
                new_batch.append(item)  
        self.mbatch = tuple(new_batch)

    def criterion(self, outputs=None, targets=None):
        if outputs is None:
            outputs = self.output
            targets = self.mb_y.long()
        else:
            targets = targets.to(dtype=torch.long,
                                 device=outputs[0].device if not isinstance(outputs, torch.Tensor) else outputs.device)

        if isinstance(outputs, torch.Tensor):
            logits = outputs
            return self._criterion(logits, targets)
        else:
            logits, img_embeds, text_embeds = outputs
        ce_loss = self._criterion(logits, targets)

        clip_loss = 0.0
        if text_embeds is not None:
            B = img_embeds.size(0)
            logit_scale = getattr(self.model, "logit_scale", torch.tensor(1.0, device=img_embeds.device))
            logit_scale = logit_scale.exp() if logit_scale.ndim == 0 else logit_scale

            logits_per_image = logit_scale * img_embeds @ text_embeds.t()
            logits_per_text = logits_per_image.t()
            ground_truth = torch.arange(B, device=img_embeds.device)

            clip_loss = (
                                F.cross_entropy(logits_per_image, ground_truth) +
                                F.cross_entropy(logits_per_text, ground_truth)
                        ) / 2

        clip_loss_weight = 1.0
        loss = ce_loss + clip_loss_weight * clip_loss
        return loss

    def forward(self):
        scene = self.mb_scene_id
        if isinstance(scene, torch.Tensor):
            uniq = torch.unique(scene)
            if len(uniq) != 1:
                raise ValueError(f"Batch contains multiple scenes: {uniq}")
            scene_val = int(uniq.item())
        else:
            scene_val = int(scene)
        if isinstance(self.model, torch.nn.DataParallel):
            root_model = self.model.module
        else:
            root_model = self.model
        root_model.set_active_scene(scene_val)
        outputs = self.model(self.mb_x, task_labels=self.mb_task_id,
                          scene_id=self.mb_scene_id,batch_prompts=self.mb_prompts())
        if self.model.training:
            return outputs  # (logits, img_embeds, text_embeds)
        else:
            return outputs[0]
