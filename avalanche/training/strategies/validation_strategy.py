# validation_strategy.py
import torch
import torch.nn.functional as F
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.models.utils import avalanche_forward
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.evaluation import PluginMetric
from collections import defaultdict
import os
import copy


class ValidationStrategy(BaseStrategy):
    
    def __init__(self, model, optimizer, criterion, val_datasets=None, 
                 save_dir="./best_weights", eval_every=1, **kwargs):
        super().__init__(model, optimizer, criterion, **kwargs)
        self.val_datasets = val_datasets or {}  
        self.save_dir = save_dir
        self.eval_every = eval_every  
        self.best_avg_acc = 0.0
        self.best_weights = None
        self.current_task = 0
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.eval_plugin = EvaluationPlugin(
            accuracy_metrics(epoch=True, experience=True),
            loggers=[]
        )
    
    def training_epoch(self, **kwargs):
        super().training_epoch(**kwargs)
        
        if self.clock.train_exp_epochs % self.eval_every == 0:
            self._evaluate_on_validation()
    
    def _evaluate_on_validation(self):
        if not self.val_datasets:
            return
            
        self.model.eval()
        task_accuracies = {}
        
        for task_id in range(self.current_task + 1):
            if task_id not in self.val_datasets:
                continue
                
            val_dataset = self.val_datasets[task_id]
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=32, shuffle=False
            )
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) >= 5:
                        x, y, task_label, scene_id, prompts = batch
                    else:
                        x, y = batch
                        task_label = None
                        scene_id = None
                        prompts = None
                    
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    if hasattr(self.model, 'module'):
                        root_model = self.model.module
                        if scene_id is not None:
                            scene_val = int(torch.unique(scene_id).item())
                            root_model.set_active_scene(scene_val)
                    else:
                        root_model = self.model
                        if scene_id is not None:
                            scene_val = int(torch.unique(scene_id).item())
                            root_model.set_active_scene(scene_val)
                    
                    outputs = self.model(x, task_labels=task_label, 
                                      scene_id=scene_id, batch_prompts=prompts)
                    
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    _, predicted = torch.max(logits.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            
            task_acc = correct / total if total > 0 else 0.0
            task_accuracies[task_id] = task_acc
            print(f"Task {task_id} validation accuracy: {task_acc:.4f}")
        
        if task_accuracies:
            avg_acc = sum(task_accuracies.values()) / len(task_accuracies)
            print(f"Average validation accuracy: {avg_acc:.4f}")
            
            if avg_acc > self.best_avg_acc:
                self.best_avg_acc = avg_acc
                self._save_best_weights()
                print(f"New best average accuracy: {avg_acc:.4f}, weights saved")
        
        self.model.train()
    
    def _save_best_weights(self):
        if hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        self.best_weights = copy.deepcopy(model_state)
        
        save_path = os.path.join(self.save_dir, f"best_weights_task_{self.current_task}.pth")
        torch.save(model_state, save_path)
        print(f"Best weights saved to {save_path}")
    
    def _load_best_weights(self):
        if self.best_weights is not None:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(self.best_weights)
            else:
                self.model.load_state_dict(self.best_weights)
            print("Best weights loaded")
    
    def train_exp(self, experience, eval_streams=None, **kwargs):
        if self.current_task > 0:
            self._load_best_weights()
        
        result = super().train_exp(experience, eval_streams, **kwargs)
        
        self._evaluate_on_validation()
        
        self.current_task = experience.current_experience
        
        return result
    
    def add_validation_dataset(self, task_id, val_dataset):
        self.val_datasets[task_id] = val_dataset
        print(f"Added validation dataset for task {task_id}")


class ValidationPlugin:
    
    def __init__(self, val_datasets, save_dir="./best_weights", eval_every=1):
        self.val_datasets = val_datasets
        self.save_dir = save_dir
        self.eval_every = eval_every
        self.best_avg_acc = 0.0
        self.best_weights = None
        self.current_task = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def before_training_epoch(self, strategy, **kwargs):
        if strategy.clock.train_exp_epochs % self.eval_every == 0:
            self._evaluate_on_validation(strategy)
    
    def _evaluate_on_validation(self, strategy):
        if not self.val_datasets:
            return
            
        model = strategy.model
        model.eval()
        task_accuracies = {}
        
        for task_id in range(self.current_task + 1):
            if task_id not in self.val_datasets:
                continue
                
            val_dataset = self.val_datasets[task_id]
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=32, shuffle=False
            )
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) >= 5:
                        x, y, task_label, scene_id, prompts = batch
                    else:
                        x, y = batch
                        task_label = None
                        scene_id = None
                        prompts = None
                    
                    x = x.to(strategy.device)
                    y = y.to(strategy.device)
                    
                    if hasattr(model, 'module'):
                        root_model = model.module
                        if scene_id is not None:
                            scene_val = int(torch.unique(scene_id).item())
                            root_model.set_active_scene(scene_val)
                    else:
                        root_model = model
                        if scene_id is not None:
                            scene_val = int(torch.unique(scene_id).item())
                            root_model.set_active_scene(scene_val)
                    
                    outputs = model(x, task_labels=task_label, 
                                  scene_id=scene_id, batch_prompts=prompts)
                    
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    _, predicted = torch.max(logits.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            
            task_acc = correct / total if total > 0 else 0.0
            task_accuracies[task_id] = task_acc
            print(f"Task {task_id} validation accuracy: {task_acc:.4f}")
        
        if task_accuracies:
            avg_acc = sum(task_accuracies.values()) / len(task_accuracies)
            print(f"Average validation accuracy: {avg_acc:.4f}")
            
            if avg_acc > self.best_avg_acc:
                self.best_avg_acc = avg_acc
                self._save_best_weights(model, strategy)
                print(f"New best average accuracy: {avg_acc:.4f}, weights saved")
        
        model.train()
    
    def _save_best_weights(self, model, strategy):
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        self.best_weights = copy.deepcopy(model_state)
        
        save_path = os.path.join(self.save_dir, f"best_weights_task_{self.current_task}.pth")
        torch.save(model_state, save_path)
        print(f"Best weights saved to {save_path}")
    
    def _load_best_weights(self, model):
        if self.best_weights is not None:
            if hasattr(model, 'module'):
                model.module.load_state_dict(self.best_weights)
            else:
                model.load_state_dict(self.best_weights)
            print("Best weights loaded")
    
    def before_training_exp(self, strategy, **kwargs):
        if self.current_task > 0:
            self._load_best_weights(strategy.model)
    
    def after_training_exp(self, strategy, **kwargs):
        self.current_task = strategy.experience.current_experience