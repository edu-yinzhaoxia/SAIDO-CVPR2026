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
"""
    Avalanche supports data loading using pytorch's dataloaders.
    This module provides custom dataloaders for continual learning such as
    support for balanced dataloading between different tasks or balancing
    between the current data and the replay memory.
"""

from itertools import chain
from typing import Dict, Sequence
from torch.utils.data import Dataset, Subset
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader, Sampler, BatchSampler
from avalanche.benchmarks.utils import AvalancheDataset
from torch.utils.data import DataLoader, Sampler
import random
from collections import defaultdict
from typing import Dict, Sequence, Callable, Any
import itertools
from transformers import CLIPVisionModel, CLIPTextModel, CLIPModel, CLIPTokenizer
def _default_collate_mbatches_fn(mbatches):
    """ Combines multiple mini-batches together.

    Concatenates each tensor in the mini-batches along dimension 0 (usually this
    is the batch size).

    :param mbatches: sequence of mini-batches.
    :return: a single mini-batch
    """
    batch = []
    for i in range(len(mbatches[0])):
        t = torch.cat([el[i] for el in mbatches], dim=0)
        batch.append(t)
    return batch

class SceneGroupedSampler(Sampler):
    """
    Sampler that ensures each batch only contains samples from the same scene_id.
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.scene_to_indices = {}
        for idx in range(len(dataset)):
            sample = dataset[idx]

            if isinstance(sample, tuple) and len(sample) >= 4:
                scene_id = sample[3]
            elif isinstance(sample, dict) and "scene_id" in sample:
                scene_id = sample["scene_id"]
            else:
                raise ValueError(f"无法从样本结构中提取 scene_id: {type(sample)}, {sample}")

            self.scene_to_indices.setdefault(scene_id, []).append(idx)

    def __iter__(self):
        all_batches = []
        import random
        for scene_id, idxs in self.scene_to_indices.items():
            if self.shuffle:
                random.shuffle(idxs)

            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i + self.batch_size]
                all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        return sum((len(idxs) + self.batch_size - 1) // self.batch_size
                   for idxs in self.scene_to_indices.values())

class TaskBalancedDataLoader:
    """ Task-balanced data loader for Avalanche's datasets."""

    def __init__(self, data: AvalancheDataset,
                 oversample_small_tasks: bool = False,
                 collate_mbatches=_default_collate_mbatches_fn,
                 **kwargs):
        """ Task-balanced data loader for Avalanche's datasets.

        The iterator returns a mini-batch balanced across each task, which
        makes it useful when training in multi-task scenarios whenever data is
        highly unbalanced.

        If `oversample_small_tasks == True` smaller tasks are
        oversampled to match the largest task. Otherwise, once the data for a
        specific task is terminated, that task will not be present in the
        subsequent mini-batches.

        :param data: an instance of `AvalancheDataset`.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        self.data = data
        self.dataloaders: Dict[int, DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks
        self.collate_mbatches = collate_mbatches

        # split data by task.
        task_datasets = []
        for task_label in self.data.task_set:
            tdata = self.data.task_set[task_label]
            task_datasets.append(tdata)

        # the iteration logic is implemented by GroupBalancedDataLoader.
        # we use kwargs to pass the arguments to avoid passing the same
        # arguments multiple times.
        if 'data' in kwargs:
            del kwargs['data']
        # needed if they are passed as positional arguments
        kwargs['oversample_small_groups'] = oversample_small_tasks
        kwargs['collate_mbatches'] = collate_mbatches
        self._dl = GroupBalancedDataLoader(datasets=task_datasets, **kwargs)

    def __iter__(self):
        for el in self._dl.__iter__():
            yield el

    def __len__(self):
        return self._dl.__len__()

class GroupBalancedDataLoader:
    """Data loader that balances data from multiple datasets."""

    def __init__(self,
                 datasets: Sequence[Any],
                 oversample_small_groups: bool = False,
                 collate_mbatches: Callable = _default_collate_mbatches_fn,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        self.datasets = datasets
        self.oversample_small_groups = oversample_small_groups
        self.collate_mbatches = collate_mbatches

        self.dataloaders = []
        for i, data in enumerate(self.datasets):
            dl = DataLoader(
                data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

            print(f"[DEBUG] dataset {i} type={type(data)} len={len(data)}")
            self.dataloaders.append(dl)

        self.max_len = max(len(dl) for dl in self.dataloaders)

    def __iter__(self):
        iter_dataloaders = [iter(dl) for dl in self.dataloaders]

        for it in range(self.max_len):
            mb_curr = []
            for tid, t_loader in enumerate(iter_dataloaders):
                if t_loader is None:
                    continue
                try:
                    batch = next(t_loader)
                except StopIteration:
                    if self.oversample_small_groups:
                        iter_dataloaders[tid] = iter(self.dataloaders[tid])
                        batch = next(iter_dataloaders[tid])
                    else:
                        iter_dataloaders[tid] = None
                        continue

                if batch is None:
                    print(f"[WARNING] None batch from dataset {tid} at iter {it}")
                    continue
                mb_curr.append(batch)

            if len(mb_curr) == 0:
                print(f"[WARNING] mb_curr empty at iter {it}")
                continue

            for idx, b in enumerate(mb_curr):
                if isinstance(b, (list, tuple)):
                    print(f"[DEBUG] iter {it}, dataset {idx} batch len={len(b)}, type={type(b[0])}")
                else:
                    print(f"[DEBUG] iter {it}, dataset {idx} batch type={type(b)}")

            yield self.collate_mbatches(mb_curr)

    def __len__(self):
        return self.max_len

class SceneGroupedTaskBalancedDataLoader:

    def __init__(self,
                 avalanche_dataset: AvalancheDataset,
                 batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 oversample_small_tasks: bool = True):
        self.avalanche_dataset = avalanche_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.oversample_small_tasks = oversample_small_tasks
        self.tokenizer = CLIPTokenizer.from_pretrained("./CLIP14")

        sampler = SceneGroupedSampler(
            self.avalanche_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        self._dl = DataLoader(
            self.avalanche_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_and_unpack
        )

    def _collate_and_unpack(self, batch):
        """
        解包 AvalancheDataset 的样本，返回 (x, y, task_label, scene_id, prompt)。
        """
        imgs, ys, tasks, scenes, prompts = [], [], [], [], []
        for sample in batch:
            if len(sample) >= 5:
                x, y, task, scene, prompt = sample[:5]
            else:
                raise ValueError(
                    f"样本长度不足以提取 prompt: {len(sample)} {sample}"
                )
            #print(prompt)
            #import pdb
            #pdb.set_trace()
            imgs.append(x)
            ys.append(y)
            tasks.append(task)
            scenes.append(scene)
            prompts.append(prompt)

        
        if isinstance(imgs[0], torch.Tensor):
            imgs = torch.stack(imgs)
            ys = torch.tensor(ys)
            tasks = torch.tensor(tasks)
            scenes = torch.tensor(scenes)
            
        batch_prompts = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        batch_prompts = dict(batch_prompts)
        
        return imgs, ys, tasks, scenes, batch_prompts  

    def __iter__(self):
        return iter(self._dl)

    def __len__(self):
        return len(self._dl)

class GroupBalancedInfiniteDataLoader:
    """ Data loader that balances data from multiple datasets emitting an
        infinite stream."""

    def __init__(self, datasets: Sequence[AvalancheDataset],
                 collate_mbatches=_default_collate_mbatches_fn,
                 **kwargs):
        """ Data loader that balances data from multiple datasets emitting an
        infinite stream.

        Mini-batches emitted by this dataloader are created by collating
        together mini-batches from each group. It may be used to balance data
        among classes, experiences, tasks, and so on.

        :param datasets: an instance of `AvalancheDataset`.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param kwargs: data loader arguments used to instantiate the loader for
            each group separately. See pytorch :class:`DataLoader`.
        """
        self.datasets = datasets
        self.dataloaders = []
        self.collate_mbatches = collate_mbatches

        for data in self.datasets:
            infinite_sampler = RandomSampler(data, replacement=True,
                                             num_samples=10 ** 10)
            dl = DataLoader(
                data,
                sampler=infinite_sampler,
                **kwargs)
            self.dataloaders.append(dl)
        self.max_len = 10 ** 10

    def __iter__(self):
        iter_dataloaders = []
        for dl in self.dataloaders:
            iter_dataloaders.append(iter(dl))

        while True:
            mb_curr = []
            for tid, t_loader in enumerate(iter_dataloaders):
                batch = next(t_loader)
                mb_curr.append(batch)
            yield self.collate_mbatches(mb_curr)

    def __len__(self):
        return self.max_len


class ReplayDataLoader:
    """ Custom data loader for rehearsal/replay strategies."""

    def __init__(self, data: AvalancheDataset, memory: AvalancheDataset = None,
                 oversample_small_tasks: bool = False,
                 collate_mbatches=_default_collate_mbatches_fn,
                 batch_size: int = 32,
                 force_data_batch_size: int = None,
                 **kwargs):
        """ Custom data loader for rehearsal strategies.

        The iterates in parallel two datasets, the current `data` and the
        rehearsal `memory`, which are used to create mini-batches by
        concatenating their data together. Mini-batches from both of them are
        balanced using the task label (i.e. each mini-batch contains a balanced
        number of examples from all the tasks in the `data` and `memory`).
        
        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: AvalancheDataset.
        :param memory: AvalancheDataset.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param batch_size: the size of the batch. It must be greater than or
            equal to the number of tasks.
        :param ratio_data_mem: How many of the samples should be from
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """

        self.data = data
        self.memory = memory
        self.loader_data: Sequence[DataLoader] = {}
        self.loader_memory: Sequence[DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks
        self.collate_mbatches = collate_mbatches

        if force_data_batch_size is not None:
            assert force_data_batch_size <= batch_size, \
                "Forced batch size of data must be <= entire batch size"

            mem_batch_size = batch_size - force_data_batch_size
            remaining_example = 0
            mem_keys = len(self.memory.task_set)
            assert mem_batch_size >= mem_keys, \
                "Batch size must be greator or equal " \
                "to the number of tasks in the memory."

            self.loader_data, _ = self._create_dataloaders(
                data, force_data_batch_size,
                remaining_example, **kwargs)
            self.loader_memory, _ = self._create_dataloaders(
                memory, mem_batch_size,
                remaining_example, **kwargs)
        else:
            num_keys = len(self.data.task_set) + len(self.memory.task_set)
            assert batch_size >= num_keys, \
                "Batch size must be greator or equal " \
                "to the number of tasks in the memory " \
                "and current data."

            single_group_batch_size = batch_size // num_keys
            remaining_example = batch_size % num_keys

            self.loader_data, remaining_example = self._create_dataloaders(
                data, single_group_batch_size,
                remaining_example, **kwargs)
            self.loader_memory, remaining_example = self._create_dataloaders(
                memory, single_group_batch_size,
                remaining_example, **kwargs)

        self.max_len = max([len(d) for d in chain(
            self.loader_data.values(), self.loader_memory.values())]
                           )

    def __iter__(self):
        iter_data_dataloaders = {}
        iter_buffer_dataloaders = {}

        for t in self.loader_data.keys():
            iter_data_dataloaders[t] = iter(self.loader_data[t])
        for t in self.loader_memory.keys():
            iter_buffer_dataloaders[t] = iter(self.loader_memory[t])

        max_len = max([len(d) for d in chain(iter_data_dataloaders.values(),
                                             iter_buffer_dataloaders.values())])
        try:
            for it in range(max_len):
                mb_curr = []
                self._get_mini_batch_from_data_dict(
                    self.data, iter_data_dataloaders,
                    self.loader_data, self.oversample_small_tasks,
                    mb_curr)

                self._get_mini_batch_from_data_dict(
                    self.memory, iter_buffer_dataloaders,
                    self.loader_memory, self.oversample_small_tasks,
                    mb_curr)

                yield self.collate_mbatches(mb_curr)
        except StopIteration:
            return

    def __len__(self):
        return self.max_len

    def _get_mini_batch_from_data_dict(self, data, iter_dataloaders,
                                       loaders_dict, oversample_small_tasks,
                                       mb_curr):
        # list() is necessary because we may remove keys from the
        # dictionary. This would break the generator.
        for t in list(iter_dataloaders.keys()):
            t_loader = iter_dataloaders[t]
            try:
                tbatch = next(t_loader)
            except StopIteration:
                # StopIteration is thrown if dataset ends.
                # reinitialize data loader
                if oversample_small_tasks:
                    # reinitialize data loader
                    iter_dataloaders[t] = iter(loaders_dict[t])
                    tbatch = next(iter_dataloaders[t])
                else:
                    del iter_dataloaders[t]
                    continue
            mb_curr.append(tbatch)

    def _create_dataloaders(self, data_dict, single_exp_batch_size,
                            remaining_example, **kwargs):
        loaders_dict: Dict[int, DataLoader] = {}
        for task_id in data_dict.task_set:
            data = data_dict.task_set[task_id]
            current_batch_size = single_exp_batch_size
            if remaining_example > 0:
                current_batch_size += 1
                remaining_example -= 1
            loaders_dict[task_id] = DataLoader(
                data, batch_size=current_batch_size, **kwargs)
        return loaders_dict, remaining_example


__all__ = [
    'TaskBalancedDataLoader',
    'GroupBalancedDataLoader',
    'ReplayDataLoader',
    'GroupBalancedInfiniteDataLoader'
]
