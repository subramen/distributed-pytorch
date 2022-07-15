import torch
import torch.nn.functional as F
from utils import MyTrainDataset

import os
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def load_train_objs():
    train_set = MyTrainDataset(2048)   # load your dataset
    model = torch.nn.Linear(20, 1)     # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


class Trainer:
    def __init__(self, model, train_set, optimizer, device, batch_size):
        self.model = self.prepare_model_for_DDP(model, device)
        self.train_set = train_set
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size

    def prepare_model_for_DDP(self, model, device):
        model.to(device)
        model = DDP(model, [device])
        return model

    def run_batch(self, source, targets):
        self.optimizer.zero_grad()
        source = source.to(self.device)
        targets = targets.to(self.device)
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def run_epoch(self, nb_epochs):
        train_data = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=32,
            shuffle=False,
            sampler=DistributedSampler(self.train_set),
        )
        for epoch in range(nb_epochs):
            print(f"[GPU{self.device}] Epoch {epoch} | No. of batches: {len(train_data)} | Batchsize: {len(next(iter(train_data))[0])}")
            for source, targets in train_data:
                self.run_batch(source, targets)

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), "model_ddp.pth")
        print("Model state dict saved at model.pth")


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    trainer = Trainer(model, dataset, optimizer, device=rank, batch_size=32)  # effective batch_size *= world_size
    trainer.run_epoch(10)
    if rank == 0:
        trainer.save_checkpoint()
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
