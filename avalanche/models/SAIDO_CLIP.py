import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel  


class SAIDO_MultiScene(nn.Module):

    def init_all_scenes(self, scene_ids: list):
        for sid in scene_ids:
            self._ensure_scene(str(sid))
        print(f"[INIT] all scenes LoRA initialized: {self._scenes}")

    def __init__(self,
                 model_name: str = "./CLIP14",
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 last_k: int = 0,  # 0 = all layers, >0 = only attach to last k layers
                 target_modules: list = ["q_proj", "k_proj", "v_proj", "mlp"],
                 # ["q_proj","k_proj","v_proj","out_proj"]
                 select_feature: str = "cls",
                 num_classes: int = 2,
                 with_text=True):
        super().__init__()
        self.select_feature = select_feature
        self.with_text = with_text

        base_model = CLIPVisionModel.from_pretrained(model_name, local_files_only=True)
        base_model.requires_grad_(False)
        base_model.eval()
        self.hidden_size = base_model.config.hidden_size
        self.proj_dim = base_model.config.projection_dim

        n_layers = base_model.config.num_hidden_layers  
        selected_modules = []
        for i in range(n_layers):
            if last_k > 0 and i < n_layers - last_k:
                continue
            prefix = f"vision_model.encoder.layers.{i}.self_attn"
            for proj in target_modules:
                selected_modules.append(f"{prefix}.{proj}")
            if "mlp" in target_modules:
                selected_modules.append(f"vision_model.encoder.layers.{i}.mlp.fc1")
                selected_modules.append(f"vision_model.encoder.layers.{i}.mlp.fc2")

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=selected_modules,
            bias="none"
        )

        self.clip_peft: PeftModel = get_peft_model(base_model, self.lora_config)
        for n, p in self.clip_peft.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False

        self.visual_projection = nn.Linear(self.hidden_size, self.proj_dim)
        nn.init.xavier_uniform_(self.visual_projection.weight)
        if self.visual_projection.bias is not None:
            nn.init.zeros_(self.visual_projection.bias)

        self.fc = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.proj_dim, num_classes)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0.0, 0.02)  
                if layer.bias is not None:  
                    nn.init.zeros_(layer.bias)

        for p in self.visual_projection.parameters():
            p.requires_grad = True
        for p in self.fc.parameters():
            p.requires_grad = True

        self._scenes = set()
        self._active_scene = None

        if self.with_text:
            self.tokenizer = CLIPTokenizer.from_pretrained("./CLIP14")
            self.base_text = CLIPTextModel.from_pretrained(model_name, local_files_only=True)
            self.base_text.requires_grad_(False)
            self.base_text.eval()
            self.text_hidden_size = self.base_text.config.hidden_size

            self.text_projection = CLIPModel.from_pretrained(model_name, local_files_only=True).text_projection
            for param in self.text_projection.parameters():
                param.requires_grad = False

    def _ensure_scene(self, scene_id: str):
        if scene_id not in self._scenes:
            self.add_scene(scene_id)

    def add_scene(self, scene_id: str):

        self.clip_peft.add_adapter(scene_id, self.lora_config)
        self._scenes.add(scene_id)
        print(f"[ADD] new scene '{scene_id}', current scene list={list(self._scenes)}")

    def set_active_scene(self, scene_id: str):
        if isinstance(scene_id, torch.Tensor):
            scene_id = scene_id.item()
        scene_id = str(scene_id)
        self._ensure_scene(scene_id)
        self.clip_peft.set_adapter(scene_id)

        self._active_scene = scene_id
        self._set_requires_grad_for_active()

    def _set_requires_grad_for_active(self):
        active = self._active_scene
        for n, p in self.clip_peft.named_parameters():
            if "lora_" in n:
                p.requires_grad = (f"lora_A.{active}.weight" in n or f"lora_B.{active}.weight" in n)
            else:
                p.requires_grad = False
        for p in self.visual_projection.parameters():
            p.requires_grad = True
        for p in self.fc.parameters():
            p.requires_grad = True

    def forward(self, images, task_labels=None, scene_id=None,batch_prompts=None):
        device = images.device
        B = images.size(0)

        if scene_id is None:
            raise ValueError(f"scene_id is None")
        unique_scenes = torch.unique(scene_id)
        if len(unique_scenes) != 1:
            raise ValueError(f"scene_id is not consistent: {unique_scenes}")
        scene_id = str(unique_scenes[0].item())
        if self._active_scene != scene_id:
            self.set_active_scene(scene_id)
        outputs = self.clip_peft(images, output_hidden_states=True)
        feats = outputs.hidden_states[-1]  # (B, 1+N, D)

        if self.select_feature == "cls":
            feats = feats[:, 0:1]
        elif self.select_feature == "patch":
            feats = feats[:, 1:]
            feats = feats.mean(dim=1)
        elif self.select_feature == "cls+patch":
            feats = feats.mean(dim=1)
        else:
            raise ValueError(f"Unknown select_feature: {self.select_feature}")

        proj = self.visual_projection(feats)
        proj = proj / (proj.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        img_embeds = proj.squeeze(1)
        logits = self.fc(img_embeds)

        if batch_prompts is None:
            return logits
        else:
            texts = {k: v.to(device) for k, v in batch_prompts.items()}
            input_ids = texts["input_ids"]
            attention_mask = texts["attention_mask"]
            text_outputs = self.base_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            text_feats = text_outputs.last_hidden_state

            eot_token_id = self.tokenizer.eos_token_id
            eot_indices = (input_ids == eot_token_id).int().argmax(dim=-1)

            B_text = input_ids.size(0)
            batch_idx = torch.arange(B_text, device=text_feats.device)
            text_embeds = text_feats[batch_idx, eot_indices.to(text_feats.device)]
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds / (text_embeds.norm(p=2, dim=-1, keepdim=True) + 1e-12)

            return logits, img_embeds, text_embeds

    def save_scene_lora(self, dir_path: str, scene_id: str):
        if scene_id not in self._scenes:
            raise ValueError(f"Scene '{scene_id}' not found.")
        self.clip_peft.save_pretrained(dir_path, selected_adapters=[scene_id])

    def load_scene_lora(self, dir_path: str, scene_id: str):
        if scene_id not in self._scenes:
            self.add_scene(scene_id)
        self.clip_peft.load_adapter(dir_path, adapter_name=scene_id)
