################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
import logging
import warnings

import torch
from torch.utils.data import DataLoader
from typing import Optional, Sequence, Union, List

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader,SceneGroupedTaskBalancedDataLoader
from avalanche.models import DynamicModule
from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.clock import Clock
from avalanche.training.plugins.evaluation import default_logger
from typing import TYPE_CHECKING
from typing import List
from avalanche.training.plugins import EvaluationPlugin

if TYPE_CHECKING:
    from avalanche.core import StrategyCallbacks
    from avalanche.training.plugins import StrategyPlugin


logger = logging.getLogger(__name__)

def _get_root_model(model):
    """Return underlying model if wrapped by DataParallel / DistributedDataParallel"""
    return model.module if hasattr(model, "module") else model

def collect_lora_param_pairs(model):
    """
    Collect LoRA A/B weight parameter pairs from model.named_parameters().
    Returns dict: {(prefix, adapter_name): (A_param, B_param)}
    prefix: string prefix of parameter name up to the '.lora_A' / '.lora_B' part
    adapter_name: the token after 'lora_A.' (e.g. 'default' or '6')
    """
    m = _get_root_model(model)
    a_map = {}
    b_map = {}
    for name, p in m.named_parameters():
        # we only consider weight tensors for pairing
        if ".lora_A." in name and name.endswith(".weight"):
            prefix, rest = name.split(".lora_A.", 1)
            adapter = rest.rsplit(".weight", 1)[0]
            key = (prefix, adapter)
            a_map[key] = p
        elif ".lora_B." in name and name.endswith(".weight"):
            prefix, rest = name.split(".lora_B.", 1)
            adapter = rest.rsplit(".weight", 1)[0]
            key = (prefix, adapter)
            b_map[key] = p
    pairs = {}
    for k in a_map.keys() & b_map.keys():
        pairs[k] = (a_map[k], b_map[k])
    return pairs, a_map, b_map

def print_lora_delta_stats(model, step=None, topk=0):
    """
    Print statistics of ΔW = B @ A for all LoRA A/B pairs found.
    If no pairs are found, prints what was matched to help debug.
    """
    pairs, a_map, b_map = collect_lora_param_pairs(model)
    if len(pairs) == 0:
        print("[LoRA DEBUG] No matched (A,B) pairs found.")
        if len(a_map) > 0:
            print("[LoRA DEBUG] Found A params (no B match):")
            for (pref, adapter), p in a_map.items():
                print(f"  A: prefix='{pref}', adapter='{adapter}', name_shape={tuple(p.shape)}")
        if len(b_map) > 0:
            print("[LoRA DEBUG] Found B params (no A match):")
            for (pref, adapter), p in b_map.items():
                print(f"  B: prefix='{pref}', adapter='{adapter}', name_shape={tuple(p.shape)}")
        return

    # print header
    header = f"[LoRA ΔW] Step={step}" if step is not None else "[LoRA ΔW]"
    print(header)
    # optionally limit number printed (topk by norm)
    stats = []
    for (prefix, adapter), (A_param, B_param) in pairs.items():
        # ensure they are on same device
        try:
            with torch.no_grad():
                A = A_param.detach()
                B = B_param.detach()
                # compute delta = B @ A
                # A: [r, in], B: [out, r]  => delta: [out, in]
                try:
                    delta = B @ A
                except RuntimeError:
                    # if shapes mismatch, try transpose A (safety)
                    delta = B @ A.t() if A.dim() == 2 else B.matmul(A)
                norm = float(delta.norm().cpu().item())
                mean = float(delta.mean().cpu().item())
                mn = float(delta.min().cpu().item())
                mx = float(delta.max().cpu().item())
                shape = tuple(delta.shape)
                stats.append(((prefix, adapter), norm, mean, mn, mx, shape))
        except Exception as e:
            print(f"  [ERROR] computing ΔW for prefix={prefix}, adapter={adapter}: {e}")

    # sort by norm desc
    stats.sort(key=lambda x: x[1], reverse=True)
    if topk > 0:
        stats = stats[:topk]

    for (prefix, adapter), norm, mean, mn, mx, shape in stats:
        print(f"  prefix='{prefix}', adapter='{adapter}': ΔW shape={shape}, norm={norm:.6e}, mean={mean:.6e}, min={mn:.6e}, max={mx:.6e}")

class BaseStrategy:

    DISABLED_CALLBACKS: Sequence[str] = ()
    """Internal class attribute used to disable some callbacks if a strategy
    does not support them."""

    def __init__(self, model: Module, optimizer: Optimizer,
                 criterion=CrossEntropyLoss(),
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger, eval_every=-1):

        self._criterion = criterion

        self.model: Module = model
        """ PyTorch model. """

        self.optimizer: Optimizer = optimizer
        """ PyTorch optimizer. """

        self.train_epochs: int = train_epochs
        """ Number of training epochs. """

        self.train_mb_size: int = train_mb_size
        """ Training mini-batch size. """

        self.eval_mb_size: int = train_mb_size if eval_mb_size is None \
            else eval_mb_size
        """ Eval mini-batch size. """

        self.device = device
        """ PyTorch device where the model will be allocated. """

        self.plugins = [] if plugins is None else plugins
        """ List of `StrategyPlugin`s. """

        if evaluator is None:
            evaluator = EvaluationPlugin()
        self.plugins.append(evaluator)
        self.evaluator = evaluator
        """ EvaluationPlugin used for logging and metric computations. """

        self.clock = Clock()
        """ Incremental counters for strategy events. """
        # WARNING: Clock needs to be the last plugin, otherwise
        # counters will be wrong for plugins called after it.
        self.plugins.append(self.clock)

        self.eval_every = eval_every
        """ Frequency of the evaluation during training. """

        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################
        self.experience = None
        """ Current experience. """

        self.adapted_dataset = None
        """ Data used to train. It may be modified by plugins. Plugins can 
        append data to it (e.g. for replay). 

        .. note::

            This dataset may contain samples from different experiences. If you 
            want the original data for the current experience  
            use :attr:`.BaseStrategy.experience`.
        """

        self.dataloader = None
        """ Dataloader. """

        self.mbatch = None
        """ Current mini-batch. """

        self.mb_output = None
        """ Model's output computed on the current mini-batch. """

        self.loss = None
        """ Loss of the current mini-batch. """

        self.is_training: bool = False
        """ True if the strategy is in training mode. """

        self.current_eval_stream = None
        """ Current evaluation stream. """

        self._stop_training = False

        self._warn_for_disabled_plugins_callbacks()
        self._warn_for_disabled_metrics_callbacks()

    @property
    def training_exp_counter(self):
        """ Counts the number of training steps. +1 at the end of each
        experience. """
        warnings.warn(
            "Deprecated attribute. You should use self.clock.train_exp_counter"
            " instead.", DeprecationWarning)
        return self.clock.train_exp_counter

    @property
    def epoch(self):
        """ Epoch counter. """
        warnings.warn(
            "Deprecated attribute. You should use self.clock.train_exp_epochs"
            " instead.", DeprecationWarning)
        return self.clock.train_exp_epochs

    @property
    def mb_it(self):
        """ Iteration counter. Reset at the start of a new epoch. """
        warnings.warn(
            "Deprecated attribute. You should use "
            "self.clock.train_epoch_iterations"
            " instead.", DeprecationWarning)
        return self.clock.train_epoch_iterations

    @property
    def is_eval(self):
        """ True if the strategy is in evaluation mode. """
        return not self.is_training

    @property
    def mb_x(self):
        """ Current mini-batch input. """
        return self.mbatch[0]

    @property
    def mb_y(self):
        """ Current mini-batch target. """
        return self.mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        assert len(self.mbatch) >= 5
        return self.mbatch[-3]

    def criterion(self):
        """ Loss function. """
        # import pdb;pdb.set_trace()
        mb_y_long = self.mb_y.to(dtype=torch.long)
        # print("mb_output.shape:", self.mb_output.shape)
        # print("mb_y_long.shape:", mb_y_long.shape)
        return self._criterion(self.mb_output, mb_y_long)
        # return self._criterion(self.mb_output, self.mb_y)

    def train(self, experiences: Union[Experience, Sequence[Experience]],
              eval_streams: Optional[Sequence[Union[Experience,
              Sequence[
                  Experience]]]] = None,
              val_streams: Optional[Sequence[Union[Experience, Sequence[Experience]]]] = None,
              **kwargs):
        """ Training loop. if experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.
        It returns a dictionary with last recorded value for each metric.

        :param experiences: single Experience or sequence.
        :param eval_streams: list of streams for evaluation.
            If None: use training experiences for evaluation.
            Use [] if you do not want to evaluate during training.

        :return: dictionary containing last recorded value for
            each metric name.
        """
        self.is_training = True
        self._stop_training = False

        self.model.train()
        self.model.to(self.device)
        if not isinstance(experiences, Sequence):
            experiences = [experiences]
        if eval_streams is None:
            #print('---------------------ok----------------------')
            eval_streams = [experiences]

        # Save validation stream (optional), for validation set evaluation after each epoch
        self._val_streams_global = val_streams
        self._before_training(**kwargs)
        self._periodic_eval(eval_streams, do_final=False, do_initial=True)

        for self.experience_id, self.experience in enumerate(experiences):
            # Select corresponding validation stream for current experience (assume val_streams organized by experience)
            current_val_stream = None
            if self._val_streams_global is not None:
                if isinstance(self._val_streams_global, Sequence) and len(self._val_streams_global) > self.experience_id:
                    current_val_stream = self._val_streams_global[self.experience_id]
                elif not isinstance(self._val_streams_global, Sequence):
                    # If val_streams is not a sequence, use directly
                    current_val_stream = self._val_streams_global
            self.train_exp(self.experience, val_streams=current_val_stream, **kwargs)
        self._after_training(**kwargs)
        res = self.evaluator.get_last_metrics()
        return res

    def train_exp(self, experience: Experience, eval_streams=None, val_streams=None, **kwargs):
        """ Training loop over a single Experience object.

        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param kwargs: custom arguments.
        """
        self.experience = experience
        self.model.train()

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Sequence):
                eval_streams[i] = [exp]

        if val_streams is None:
            val_streams = getattr(self, "_val_streams_global", None)

        self._best_val_acc_epoch = 0
        self._best_state_epoch = None
        self._best_epoch_idx = -1

        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)
        self.make_train_dataloader(**kwargs)

        self.model = self.model_adaptation()
        self.make_optimizer()

        self._before_training_exp(**kwargs)

        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            import pdb
            pdb.set_trace()
            self._after_training_epoch(**kwargs)
            import pdb
            pdb.set_trace()

            if val_streams is not None:
                print(f"[VAL] Epoch { _ + 1}/{self.train_epochs}")
                try:
                    avg_acc = self._validate_on_streams(val_streams)
                    if avg_acc is not None:
                        print(f"[VAL] avg_acc={avg_acc:.4f}, best={self._best_val_acc_epoch:.4f}")
                        if avg_acc > self._best_val_acc_epoch:
                            self._best_val_acc_epoch = avg_acc
                            self._best_epoch_idx = _
                            # Save current model weights
                            root = self.model.module if hasattr(self.model, 'module') else self.model
                            self._best_state_epoch = {k: v.detach().cpu().clone() for k, v in root.state_dict().items()}
                except Exception as e:
                    print(f"[VAL ERROR] {e}")
                    pass

        if self._best_state_epoch is not None:
            try:
                root = self.model.module if hasattr(self.model, 'module') else self.model
                root.load_state_dict(self._best_state_epoch)
            except Exception:
                pass
        self._after_training_exp(**kwargs)

    def _validate_on_streams(self, val_streams):
        """Evaluate on given validation stream, return average Top1 accuracy (if available)."""
        prev_training = self.is_training
        self.is_training = False
        self.model.eval()

        streams = val_streams
        if streams is None:
            return None
        if not isinstance(streams, Sequence):
            streams = [streams]
        for i, exp in enumerate(streams):
            if not isinstance(exp, Sequence):
                streams[i] = [exp]

        acc_list = []
        for exp_group in streams:
            for exp in exp_group:
                try:
                    res = self.eval(exp)
                    key_candidates = [k for k in res.keys() if 'Top1_Acc' in k and 'eval_phase' in k]
                    val = None
                    for k in key_candidates:
                        v = res.get(k, None)
                        try:
                            v = float(v.cpu().item()) if isinstance(v, torch.Tensor) else float(v)
                        except Exception:
                            v = None
                        if v is not None:
                            val = v
                            break
                    if val is not None:
                        acc_list.append(val)
                except Exception:
                    pass

        self.is_training = prev_training
        if prev_training:
            self.model.train()

        if len(acc_list) == 0:
            return None
        return sum(acc_list) / len(acc_list)

    def _periodic_eval(self, eval_streams, do_final, do_initial=False):
        """ Periodic eval controlled by `self.eval_every`. """
        # Since we are switching from train to eval model inside the training
        # loop, we need to save the training state, and restore it after the
        # eval is done.
        _prev_state = (
            self.experience,
            self.adapted_dataset,
            self.dataloader,
            self.is_training)

        # save each layer's training mode, to restore it later
        _prev_model_training_modes = {}
        for name, layer in self.model.named_modules():
            _prev_model_training_modes[name] = layer.training

        curr_epoch = self.clock.train_exp_epochs
        if (self.eval_every == 0 and (do_final or do_initial)) or \
                (self.eval_every > 0 and do_initial) or \
                (self.eval_every > 0 and curr_epoch % self.eval_every == 0):
            # in the first case we are outside epoch loop
            # in the second case we are within epoch loop
            for exp in eval_streams:
                #print('exp:',exp.current_experience)
                self.eval(exp)

        # restore train-state variables and training mode.
        self.experience, self.adapted_dataset = _prev_state[:2]
        self.dataloader = _prev_state[2]
        self.is_training = _prev_state[3]

        # restore each layer's training mode to original
        for name, layer in self.model.named_modules():
            prev_mode = _prev_model_training_modes[name]
            layer.train(mode=prev_mode)

    def stop_training(self):
        """ Signals to stop training at the next iteration. """
        self._stop_training = True

    def train_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """
        self.adapted_dataset = self.experience.dataset
        self.adapted_dataset = self.adapted_dataset.train()

    @torch.no_grad()
    def eval(self,
             exp_list: Union[Experience, Sequence[Experience]],
             **kwargs):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        self.is_training = False
        self.model.eval()

        if not isinstance(exp_list, Sequence):
            exp_list = [exp_list]
        self.current_eval_stream = exp_list

        self._before_eval(**kwargs)
        for self.experience in exp_list:
            # Data Adaptation
            self._before_eval_dataset_adaptation(**kwargs)
            self.eval_dataset_adaptation(**kwargs)
            self._after_eval_dataset_adaptation(**kwargs)
            self.make_eval_dataloader(**kwargs)

            # Model Adaptation (e.g. freeze/add new units)
            self.model = self.model_adaptation()

            self._before_eval_exp(**kwargs)
            self.eval_epoch(**kwargs)
            self._after_eval_exp(**kwargs)

        self._after_eval(**kwargs)
        res = self.evaluator.get_last_metrics()
        return res

    def _before_training_exp(self, **kwargs):
        """
        Called  after the dataset and data loader creation and
        before the training loop.
        """
        for p in self.plugins:
            p.before_training_exp(self, **kwargs)

    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
        """ Data loader initialization.

        Called at the start of each learning experience after the dataset
        adaptation.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """
        self.dataloader = SceneGroupedTaskBalancedDataLoader(
            avalanche_dataset=self.adapted_dataset,  # Changed here
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            oversample_small_tasks=True  # Can be set according to needs
        )
        '''
        for i, batch in enumerate(self.dataloader):
            imgs, ys, tasks, scenes, prompts = batch
            print(prompts)
            print(type(prompts[0]))
        '''

    def make_eval_dataloader(self, num_workers=0, pin_memory=True,
                             **kwargs):
        """
        Initializes the eval data loader.
        :param num_workers: How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
            (default: 0).
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param kwargs:
        :return:
        """
        self.dataloader = SceneGroupedTaskBalancedDataLoader(
            avalanche_dataset=self.adapted_dataset,  # Changed here
            batch_size=self.train_mb_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            oversample_small_tasks=True  # Can be set according to needs
        )

    def _after_train_dataset_adaptation(self, **kwargs):
        """
        Called after the dataset adaptation and before the
        dataloader initialization. Allows to customize the dataset.
        :param kwargs:
        :return:
        """
        for p in self.plugins:
            p.after_train_dataset_adaptation(self, **kwargs)

    def _before_training_epoch(self, **kwargs):
        """
        Called at the beginning of a new training epoch.
        :param kwargs:
        :return:
        """
        for p in self.plugins:
            p.before_training_epoch(self, **kwargs)

    def training_epoch(self, **kwargs):
        """ Training epoch.

        :param kwargs:
        :return:
        """

        for self.i_batch, self.mbatch in enumerate(self.dataloader):
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            self._before_forward(**kwargs)
            self.output=self.forward()
            self.mb_output = self.output[0]
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _unpack_minibatch(self):
        """ We assume mini-batches have the form <x, y, ..., t>.
        This allows for arbitrary tensors between y and t.
        Keep in mind that in the most general case mb_task_id is a tensor
        which may contain different labels for each sample.
        """
        assert len(self.mbatch) >= 5
        for i in range(len(self.mbatch)):
            self.mbatch[i] = self.mbatch[i].to(self.device)

    def _before_training(self, **kwargs):
        for p in self.plugins:
            p.before_training(self, **kwargs)

    def _after_training(self, **kwargs):
        for p in self.plugins:
            p.after_training(self, **kwargs)

    def _before_training_iteration(self, **kwargs):
        for p in self.plugins:
            p.before_training_iteration(self, **kwargs)

    def _before_forward(self, **kwargs):
        for p in self.plugins:
            p.before_forward(self, **kwargs)

    def _after_forward(self, **kwargs):
        for p in self.plugins:
            p.after_forward(self, **kwargs)

    def _before_backward(self, **kwargs):
        for p in self.plugins:
            p.before_backward(self, **kwargs)

    def _after_backward(self, **kwargs):
        for p in self.plugins:
            p.after_backward(self, **kwargs)

    def _after_training_iteration(self, **kwargs):
        for p in self.plugins:
            p.after_training_iteration(self, **kwargs)

    def _before_update(self, **kwargs):
        for p in self.plugins:
            p.before_update(self, **kwargs)

    def _after_update(self, **kwargs):
        for p in self.plugins:
            p.after_update(self, **kwargs)

    def _after_training_epoch(self, **kwargs):
        print(self.plugins)
        for p in self.plugins:
            print('plugin:',p)
            p.after_training_epoch(self, **kwargs)

    def _after_training_exp(self, **kwargs):
        for p in self.plugins:
            p.after_training_exp(self, **kwargs)

    def _before_eval(self, **kwargs):
        for p in self.plugins:
            p.before_eval(self, **kwargs)

    def _before_eval_exp(self, **kwargs):
        for p in self.plugins:
            p.before_eval_exp(self, **kwargs)

    def eval_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """
        self.adapted_dataset = self.experience.dataset
        self.adapted_dataset = self.adapted_dataset.eval()

    def _before_eval_dataset_adaptation(self, **kwargs):
        for p in self.plugins:
            p.before_eval_dataset_adaptation(self, **kwargs)

    def _after_eval_dataset_adaptation(self, **kwargs):
        for p in self.plugins:
            p.after_eval_dataset_adaptation(self, **kwargs)

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)
            self._before_eval_forward(**kwargs)
            self.output = self.forward()
            self.mb_output = self.output
            #self.mb_output = self.forward()
            self._after_eval_forward(**kwargs)
            self.loss = self.criterion()

            self._after_eval_iteration(**kwargs)

    def _after_eval_exp(self, **kwargs):
        for p in self.plugins:
            p.after_eval_exp(self, **kwargs)

    def _after_eval(self, **kwargs):
        for p in self.plugins:
            p.after_eval(self, **kwargs)

    def _before_eval_iteration(self, **kwargs):
        for p in self.plugins:
            p.before_eval_iteration(self, **kwargs)

    def _before_eval_forward(self, **kwargs):
        for p in self.plugins:
            p.before_eval_forward(self, **kwargs)

    def _after_eval_forward(self, **kwargs):
        for p in self.plugins:
            p.after_eval_forward(self, **kwargs)

    def _after_eval_iteration(self, **kwargs):
        for p in self.plugins:
            p.after_eval_iteration(self, **kwargs)

    def _before_train_dataset_adaptation(self, **kwargs):
        for p in self.plugins:
            p.before_train_dataset_adaptation(self, **kwargs)

    def model_adaptation(self, model=None):
        """Adapts the model to the current data.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        if model is None:
            model = self.model

        for module in model.modules():
            if isinstance(module, DynamicModule):
                module.adaptation(self.experience.dataset)
        return model.to(self.device)

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)

    def make_optimizer(self):
        """Optimizer initialization.

        Called before each training experiene to configure the optimizer.
        """
        # we reset the optimizer's state after each experience.
        # This allows to add new parameters (new heads) and
        # freezing old units during the model's adaptation phase.
        reset_optimizer(self.optimizer, self.model)

    def _warn_for_disabled_plugins_callbacks(self):
        self._warn_for_disabled_callbacks(self.plugins)

    def _warn_for_disabled_metrics_callbacks(self):
        self._warn_for_disabled_callbacks(self.evaluator.metrics)

    def _warn_for_disabled_callbacks(
            self,
            plugins: List["StrategyCallbacks"]
    ):
        """
        Will log some warnings in case some plugins appear to be using callbacks
        that have been de-activated by the strategy class.
        """
        for disabled_callback_name in self.DISABLED_CALLBACKS:
            for plugin in plugins:
                callback = getattr(plugin, disabled_callback_name)
                callback_class = callback.__qualname__.split('.')[0]
                if callback_class not in (
                        "StrategyPlugin",
                        "PluginMetric",
                        "EvaluationPlugin",
                        "GenericPluginMetric",
                ):
                    logger.warning(
                        f"{plugin.__class__.__name__} seems to use "
                        f"the callback {disabled_callback_name} "
                        f"which is disabled by {self.__class__.__name__}"
                    )


__all__ = ['BaseStrategy']
