import torch
from torch.utils.data import Dataset, DataLoader
from utils import MyTrainDataset


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
    ) -> None:
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.gpu_id = gpu_id

    def run_batch(self, source, targets):
        self.optimizer.zero_grad()
        source = source.to(self.gpu_id)
        targets = targets.to(self.gpu_id)
        output = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()

    def run_epoch(self, nb_epochs):
        for epoch in range(nb_epochs):
            print(f"[GPU{self.gpu_id}] Epoch {epoch}", end=" | ")
            print(f"Batchsize: {len(next(iter(self.train_data))[0])}", end=" | ")
            print(f"No. of steps: {len(self.train_data)}")
            for source, targets in self.train_data:
                self.run_batch(source, targets)

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), "model_single.pth")
        print("Model state dict saved at model.pth")


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )


def main(device):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size=32)
    trainer = Trainer(model, train_data, optimizer, gpu_id=device)
    trainer.run_epoch(10)
    trainer.save_checkpoint()


if __name__ == "__main__":
    device = 0  # shorthand for cuda:0
    main(device)
