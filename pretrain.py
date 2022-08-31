import os
import argparse
import torch
import torch.nn as nn
from time import time
from os.path import join as osj
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoModelForMaskedLM, AutoConfig
from dataloader import CaptionMLMDataset
from utils import csv2list, str2bool as s2b
from utils import print_log, setup_logger


# --expt_d ./log --expt_n PT_Caps --data ../../Datasets/Captions
# --run bert_base --model bert-base-uncased --batch 128 --gpu 0


def main():
    parser = argparse.ArgumentParser(description='Pre-Train Captions')

    # Experiment params
    parser.add_argument('--expt_dir',       type=str,   help='root directory to save model & summaries')
    parser.add_argument('--expt_name',      type=str,   help='expt_dir/expt_name: organize experiments')
    parser.add_argument('--run_name',       type=str,   help='expt_dir/expt_name/run_name: organize training runs')

    # Model params
    parser.add_argument('--model',          type=str,   help='Transformer backbone', default='bert-base-uncased')
    parser.add_argument('--max_len',        type=int,   help='Max input text sequence length', default=64)

    # Data params
    parser.add_argument('--data_dir',       type=str,   help='raw dataset directory', required=True)

    # Training params
    parser.add_argument('--lr',             type=float, help='learning rate', default=1e-5)
    parser.add_argument('--epochs',         type=int,   help='number of epochs', default=20)
    parser.add_argument('--batch_size',     type=int,   help='batch size', default=8)
    parser.add_argument('--ckpt',           type=str,   help='path to model checkpoint .pth file')
    parser.add_argument('--save',           type=s2b,   help='whether to save models', default='T')
    parser.add_argument('--val_size',       type=int,   help='validation size for evaluating metrics', default=4096)
    parser.add_argument('--log_it',         type=int,   help='interval for logging training summaries', default=100)
    parser.add_argument('--save_it',        type=int,   help='num of weight updates to save model', default=30000)

    # GPU params
    parser.add_argument('--gpu_ids',        type=str,   help='GPU Device ID', default='0')
    parser.add_argument('--use_amp',        type=s2b,   help='Automatic-Mixed Precision (T/F)', default='T')

    # Misc params
    parser.add_argument('--num_workers',    type=int,   help='number of worker threads for Dataloader', default=1)

    # Parse Args
    args = parser.parse_args()

    # GPU device
    device_ids = csv2list(args.gpu_ids, cast=int)
    device = torch.device('cuda:{}'.format(device_ids[0]))

    print('GPUs: {}'.format(device_ids))

    # Configs
    lr = args.lr
    n_epochs = args.epochs
    batch_size = args.batch_size

    # Logging directory
    log_dir = osj(args.expt_dir, args.expt_name, args.run_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # TensorBoard summary  -->  /expt_dir/expt_name/run_name/
    writer = SummaryWriter(log_dir)

    # Train log file
    log_file = setup_logger(parser, log_dir)

    print('Training Log Directory: {}\n'.format(log_dir))

    # Dataset
    train_dataset = CaptionMLMDataset(args.data_dir, args.model, split='train')
    val_dataset = CaptionMLMDataset(args.data_dir, args.model, split='val')

    # Dataloader
    loader_params = dict(batch_size=batch_size, shuffle=True, drop_last=True,
                         num_workers=args.num_workers, collate_fn=train_dataset.get_collator())

    train_loader = DataLoader(train_dataset, **loader_params)
    val_loader = DataLoader(val_dataset, **loader_params)

    # Print split sizes
    train_size = train_dataset.__len__()
    val_size = val_dataset.__len__()

    log_msg = '\nTrain: {} \nValidation: {}\n\n'.format(train_size, val_size)

    # Validation set size
    val_used_size = min(val_size, args.val_size)
    log_msg += '** Validation Metrics are computed using {} samples. See --val_size\n'.format(val_used_size)

    # Model
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_config(cfg)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer.zero_grad()

    scaler = GradScaler(enabled=args.use_amp)

    # Step & Epoch
    start_epoch = 1
    curr_step = 1
    best_val_loss = 1e8

    # Multi-GPU
    # model = nn.DataParallel(model, device_ids)
    model.to(device)

    # Set mode
    model.train()

    # Log
    print_log(log_msg, log_file)

    # Train
    steps_per_epoch = len(train_loader)
    start_time = time()

    for epoch in range(start_epoch, n_epochs + 1):
        for batch in train_loader:
            # Load batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(args.use_amp):
                # Forward Pass
                out = model(**batch)

                # Loss
                loss = out.loss

            # Backward Pass
            scaler.scale(loss).backward()

            # Update Weights
            scaler.step(optimizer)
            scaler.update()

            # Clear
            optimizer.zero_grad()

            # Print Results - Loss
            if curr_step % args.log_it == 0 or curr_step == 1:
                # Validation Loss
                if val_dataset:
                    metrics = compute_eval_loss(model, val_loader, device, val_used_size)

                    # Reset the mode to training
                    model.train()
                    log_msg = 'Validation Loss: {:.4f}'.format(metrics['loss'])

                    print_log(log_msg, log_file)

                    # Add summaries to TensorBoard
                    writer.add_scalar('Val/Loss', metrics['loss'], curr_step)

                # Add summaries to TensorBoard
                writer.add_scalar('Train/Loss', loss.item(), curr_step)

                # Compute elapsed & remaining time for training to complete
                time_elapsed = (time() - start_time) / 3600
                total_time = (time_elapsed / curr_step) * steps_per_epoch * n_epochs
                time_left = total_time - time_elapsed

                log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h | time left: {:.2f}h'.format(
                    epoch, n_epochs, curr_step, steps_per_epoch, loss.item(), time_elapsed, time_left)

                print_log(log_msg, log_file)

            # Save the model
            if curr_step % args.save_it == 0:
                path = osj(log_dir, 'model_' + str(curr_step))

                if args.save:
                    model.save_pretrained(path)

                log_msg = 'Saving the model at the {} step to directory:{}'.format(curr_step, log_dir)
                print_log(log_msg, log_file)

            curr_step += 1

        # Validation Loss on the entire set
        if val_dataset:
            log_msg = '-------------------------------------------------------------------------\n'
            metrics = compute_eval_loss(model, val_loader, device, val_size)

            log_msg += '\nAfter {} epoch:\n'.format(epoch)
            log_msg += 'Validation Loss: {:.4f}\n'.format(metrics['loss'])

            # Save model after every epoch, if improved
            if metrics['loss'] < best_val_loss:
                best_val_loss = metrics['loss']

                step = '{:.1f}k'.format(curr_step / 1000) if curr_step > 1000 else f'{curr_step}'
                filename = 'ep_{}_stp_{}_loss_{:.2f}_model'.format(epoch, step, best_val_loss)

                path = osj(log_dir, filename)

                if args.save:
                    model.save_pretrained(path)

                log_msg += "\n** Best Performing Model: {:.4f} ** \nSaving weights at {}\n".format(best_val_loss, path)

            log_msg += '-------------------------------------------------------------------------\n\n'

            print_log(log_msg, log_file)

            # Reset mode
            model.train()

    writer.close()
    log_file.close()


@torch.inference_mode()
def compute_eval_loss(model, dataloader, device, size) -> dict:
    """
    For the given model, computes loss on validation set.
    :param model: model to evaluate
    :param dataloader: validation/test set dataloader
    :param device: cuda/cpu device where the model resides
    :param size: no. of samples (subset) to use
    :return: metrics {'loss', 'accuracy', ...}
    """
    model.eval()
    loss = []

    eval_samples = 0

    # Evaluate on mini-batches
    for batch in dataloader:
        # Load batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward Pass
        out = model(**batch)

        # Loss
        batch_loss = out.loss

        loss.append(batch_loss)

        # Samples evaluated
        eval_samples += dataloader.batch_size

        if eval_samples >= size:
            break

    # Compute Loss
    loss = torch.tensor(loss).mean()

    metrics = dict(loss=loss)

    return metrics


if __name__ == '__main__':
    main()
