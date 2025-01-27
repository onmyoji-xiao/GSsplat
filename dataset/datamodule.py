import torch
import numpy as np
import random
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2 ** 32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2 ** 32 - 1))


class DataModule(LightningDataModule):

    def __init__(
            self,
            dataset,
            cfg,
            global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.global_rank = global_rank
        self.cfg = cfg

    def train_dataloader(self):
        dataset = self.dataset(is_train=True, cfg=self.cfg)
        if self.global_rank == 0:
            print(f"train samples number : {len(dataset.samples)} ")
        return DataLoader(
            dataset,
            self.cfg.batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=self.cfg.num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            # collate_fn=dataset.custom_collate
        )

    def val_dataloader(self):
        dataset = self.dataset(is_train=False, cfg=self.cfg)
        if self.global_rank == 0:
            print(f"val samples number : {len(dataset.samples)} ")
        return DataLoader(
            dataset,
            self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            pin_memory=True,
            # collate_fn=dataset.custom_collate
        )

    def test_dataloader(self, dataset_cfg=None):
        dataset = self.dataset(is_train=False, cfg=self.cfg)
        return DataLoader(
            dataset,
            self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            # persistent_workers=False,
            # collate_fn=dataset.custom_collate
        )
