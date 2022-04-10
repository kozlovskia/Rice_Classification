from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
import matplotlib.pyplot as plt

from data import RiceDataset, get_transform
from model import RiceClassifier


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("-o", "--output_dir", type=Path, default=Path("prediction_from_images/inference"),
                        help="Directory where outputs will be saved.")
    parser.add_argument("-i", "--input_dir", type=Path, default=Path('../data/Rice_Image_Dataset/'),
                        help="Dataset directory (where subdirs are directories named like classes).")

    parser.add_argument("-e", "--num_epochs", type=int, default=100, help="Epoch amount.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("-n", "--num_workers", type=int, default=6, help="Number of processes when reading samples.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of dataset used for validation.")

    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Neural net architecture. One of: resnet18 | efficientnet_b0.")

    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate.")

    return parser.parse_args()


def train(args):
    print('Start training. Config:', args, sep='\n')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RiceDataset(args.input_dir, transform=get_transform())
    val_amount = int(args.val_ratio * len(dataset))
    train_amount = len(dataset) - val_amount
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, (train_amount, val_amount),
                                                               generator=torch.Generator().manual_seed(42))
    print(f'Training dataset: {len(train_dataset)} | Validation dataset: {len(val_dataset)}')
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                               drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                             drop_last=False)

    model = RiceClassifier(args.backbone)
    print(model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    print(f'Using: {device}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fun = torch.nn.CrossEntropyLoss()

    loss_per_epoch = []
    accuracies_per_epoch = []
    for epoch in range(args.num_epochs):
        losses = []
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model.forward(imgs)
            batch_losses = loss_fun(preds, targets)
            batch_losses.backward()
            batch_losses = batch_losses.detach().cpu().numpy()
            losses.append(batch_losses)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = np.mean(np.array(losses))
        loss_per_epoch.append(epoch_loss)

        val_accuracy = 0
        for imgs, targets in val_loader:
            imgs.to(device)
            targets.to(device)

            with torch.no_grad():
                preds = model.forward(imgs)
                preds = torch.argmax(preds, -1)
                preds = preds.cpu().numpy()
                targets = targets.cpu().numpy()
                val_accuracy += (preds == targets).sum()

        val_accuracy /= len(val_dataset)
        accuracies_per_epoch.append(val_accuracy)

        print(f'Epoch: {epoch:2d}| Training loss: {epoch_loss:4f} | Validation accuracy: {val_accuracy:4f}.')

    torch.save(model.state_dict(), args.output_dir / 'model.pt')

    fig, axs = plt.subplots(1, 2, figsize=(30, 15))
    axs[0].plot(loss_per_epoch)
    axs[0].set_xlabel('Epoch number')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training loss')
    axs[1].plot(accuracies_per_epoch)
    axs[1].set_xlabel('Epoch number')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Validation Accuracy')
    plt.tight_layout()
    plt.savefig(str(args.output_dir / 'training_metrics.png'))

    print(f'Saved results to: {args.output_dir}.')


if __name__ == '__main__':
    train(parse_args())
