# train.py
# 单独负责模型训练

import torch
from torch.utils.tensorboard import SummaryWriter
from Dataloader import get_dataloader
from Model import my_POSSM
from engine import train_one_epoch, validate, masked_mse_loss
from utils import set_seed, save_checkpoint
from Config import my_POSSMConfig


def run_experiment(config, hyperparam, ckpt_path):
    set_seed(hyperparam["seed"])

    writer = SummaryWriter(
        log_dir = f"{hyperparam['log_dir']}/{config.backbone}"
        )
    train_loader, valid_loader = get_dataloader(
        batch_size = hyperparam["batch_size"]
    )

    model = my_POSSM(config).to(hyperparam["device"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparam["learning_rate"],
        weight_decay=hyperparam["weight_decay"]
    )

    train_losses, val_losses = [], []

    for epoch in range(hyperparam["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            masked_mse_loss,
            hyperparam["device"], writer, epoch
        )

        val_loss = validate(
            model, valid_loader,
            masked_mse_loss,
            hyperparam["device"], writer, epoch
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")

    save_checkpoint(
        ckpt_path,
        model,
        config,
        train_losses,
        val_losses
    )

    return train_losses, val_losses

    '''
    工具函数, 用于快速使用 dataloader
    '''

def build_model_and_dataloader(config, ckpt_path, device, batch_size):
    model = my_POSSM(config).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_loader = get_dataloader(batch_size)

    return model, val_loader