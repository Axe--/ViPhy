"""
Implements Finetuning & Evaluation
"""
import os
import torch
import argparse
import numpy as np
from glob import glob
from time import time
from tqdm import tqdm
import torch.nn.functional as F
from os.path import join as osj
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import List, Dict, Union, Any
from utils import str2bool as s2b
from utils import print_log, csv2list, setup_logger, save_csv
from dataloader import ViPhyDataset
from models import MaskedLM, Text2TextLM, UnifiedQA


def main():
    parser = argparse.ArgumentParser(description='ViPhy: Train + Eval')

    # Experiment params
    parser.add_argument('--mode',           type=str,   help='train or test mode', choices=['train', 'eval'])
    parser.add_argument('--expt_dir',       type=str,   help='root directory to save model & summaries')
    parser.add_argument('--expt_name',      type=str,   help='expt_dir/expt_name: organize experiments')
    parser.add_argument('--run_name',       type=str,   help='expt_dir/expt_name/run_name: organize training runs')

    # Model params
    parser.add_argument('--model',          type=str,   help='Transformer backbone', required=True)

    # Data params
    parser.add_argument('--data_dir',       type=str,   help='path to dataset directory', required=True)

    # Training params
    parser.add_argument('--lr',             type=float, help='learning rate', default=1e-4)
    parser.add_argument('--epochs',         type=int,   help='number of epochs', default=50)
    parser.add_argument('--batch_size',     type=int,   help='batch size', default=8)
    parser.add_argument('--ckpt_path',      type=str,   help='checkpoint path for inference')
    parser.add_argument('--val_size',       type=int,   help='validation set size for evaluating metrics', default=512)
    parser.add_argument('--log_interval',   type=int,   help='interval size for logging summaries', default=1000)
    parser.add_argument('--save_all',       type=s2b,   help='if unset, saves only best.pth', default='F')

    # GPU params
    parser.add_argument('--gpus',           type=str,   help='GPU Device ID', default='0')
    parser.add_argument('--amp',            type=s2b,   help='Automatic-Mixed Precision (T/F)', default='T')

    # Misc params
    parser.add_argument('--num_workers',    type=int,   help='number of worker threads for Dataloader', default=1)
    parser.add_argument('--pred_csv',       type=str,   help='predictions on `Test`set (csv)')

    # Args
    args = parser.parse_args()

    # GPU device
    device_ids = csv2list(args.gpus, cast=int)
    device = torch.device('cuda:{}'.format(device_ids[0]))

    print('GPUs: {}'.format(device_ids))

    # Configs
    lr = args.lr
    n_epochs = args.epochs
    batch_size = args.batch_size
    num_classes = 11 if 'color' in args.data_dir else \
                  3 if 'spatial' in args.data_dir else 2

    # Model Type
    _type = ViPhyDataset.get_model_type(args.model)
    t2t = _type in ['CLM', 'QA']

    # AMP
    args.amp = False if 't5-large' in args.model else True

    # Expt dir
    log_dir = osj(args.expt_dir, args.expt_name, args.run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Logging file
    log_file = setup_logger(parser, log_dir)

    # Train
    if args.mode == 'train':
        # Summaries
        writer = SummaryWriter(log_dir)

        print('Training Log Directory: {}\n'.format(log_dir))

        # Model
        if t2t:
            if _type == 'QA':
                model = UnifiedQA(args.model, device, ckpt=args.ckpt_path)
            else:
                model = Text2TextLM(args.model, device, ckpt=args.ckpt_path)
        else:
            model = MaskedLM(args.model, num_classes, device, ckpt=args.ckpt_path)

        model.train()
        # model = nn.DataParallel(model, device_ids)

        # Dataset
        train_dataset = ViPhyDataset(args.data_dir, args.model, model.tokenizer, split='train')
        val_dataset = ViPhyDataset(args.data_dir, args.model, model.tokenizer, split='val')

        # Dataloader
        loader_params = dict(batch_size=batch_size,
                             shuffle=True,
                             drop_last=True,
                             num_workers=args.num_workers)

        train_loader = DataLoader(train_dataset, **loader_params)
        val_loader = DataLoader(val_dataset, **loader_params)

        # Split sizes
        train_size = train_dataset.__len__()
        val_size = val_dataset.__len__()

        log_msg = '\nTrain: {} \nValidation: {}\n\n'.format(train_size, val_size)

        val_used_size = min(val_size, args.val_size)
        log_msg += f'** Validation Metrics are computed using {val_used_size} samples. See --val_size\n'

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr)
        optimizer.zero_grad()

        scaler = GradScaler(enabled=args.amp)

        # Init params
        start_epoch = 1
        curr_step = 1
        best_acc = 0

        print_log(log_msg, log_file)

        # Train
        steps_per_epoch = len(train_loader)
        start_time = time()

        for epoch in range(start_epoch, n_epochs + 1):
            for batch in train_loader:
                # Load batch to device
                batch['inputs'] = {k: v.to(device) for k, v in batch['inputs'].items()}
                batch['labels'] = batch['labels'].to(device)

                with autocast(args.amp):
                    # Forward Pass
                    loss = model(**batch)

                # Backward Pass
                scaler.scale(loss).backward()

                # Update Weights
                scaler.step(optimizer)
                scaler.update()

                # Clear
                optimizer.zero_grad()

                # Interval Log
                if curr_step % args.log_interval == 0 or curr_step == 1:
                    # Validation set accuracy
                    metrics = compute_eval_metrics(model, val_loader, device, val_used_size, t2t)

                    # Reset
                    model.train()

                    log_msg = 'Validation Loss: {:.4f} || Accuracy: {:.4f}'.format(
                                    metrics['loss'], metrics['accuracy'])

                    print_log(log_msg, log_file)

                    # Add summaries to TensorBoard
                    writer.add_scalar('Val/Loss', metrics['loss'], curr_step)
                    writer.add_scalar('Val/Accuracy', metrics['accuracy'], curr_step)

                    # Add summaries to TensorBoard
                    writer.add_scalar('Train/Loss', loss.item(), curr_step)

                    # Compute elapsed & remaining time for training to complete
                    time_elapsed = (time() - start_time) / 3600
                    total_time = (time_elapsed / curr_step) * steps_per_epoch * n_epochs
                    time_left = total_time - time_elapsed

                    log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h | time left: {:.2f}h'.format(
                        epoch, n_epochs, curr_step, steps_per_epoch, loss.item(), time_elapsed, time_left)

                    print_log(log_msg, log_file)

                curr_step += 1

            # Validation set metrics
            metrics = compute_eval_metrics(model, val_loader, device, val_size, t2t)

            # Reset
            model.train()

            log_msg = '-' * 50 + '\n\n' + f'{epoch} epoch:\n'
            log_msg += 'Validation Loss: {:.4f} || Accuracy: {:.4f}\n'.format(
                            metrics['loss'], metrics['accuracy'])

            print_log(log_msg, log_file)

            # Save Best
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']

                step = '{:.1f}k'.format(curr_step / 1000) if curr_step > 1000 else f'{curr_step}'

                filename = 'ep_{}_stp_{}_acc_{:.2f}_model.pth'.format(epoch, step, best_acc * 100)
                path = osj(log_dir, filename)

                state_dict = dict(model_state_dict=model.state_dict(), curr_step=curr_step,
                                  loss=loss.item(), epoch=epoch, val_acc=best_acc)
                if args.save_all:
                    torch.save(state_dict, path)

                # replaced at each save
                path = osj(log_dir, 'best.pth')
                torch.save(state_dict, path)

                log_msg += f'\n** Best Model: {best_acc:.4f} ** \nSaving weights at {path}\n\n'
                log_msg += '-' * 50 + '\n\n'

                print_log(log_msg, log_file)

        writer.close()

    elif args.mode == 'eval':
        # Checkpoint
        if args.ckpt_path is None:
            args.ckpt_path = glob(osj(log_dir, 'best.pth'))[0]

        # Model
        if t2t:
            if _type == 'QA':
                model = UnifiedQA(args.model, device, ckpt=args.ckpt_path)
            else:
                model = Text2TextLM(args.model, device, ckpt=args.ckpt_path)
        else:
            model = MaskedLM(args.model, num_classes, device, ckpt=args.ckpt_path)

        model.eval()

        # Dataset
        dataset = ViPhyDataset(args.data_dir, args.model, model.tokenizer, split='test')

        # Dataloader
        loader = DataLoader(dataset, batch_size, num_workers=args.num_workers)

        data_len = len(dataset)
        print('Total Samples: {}'.format(data_len))

        # Inference
        metrics = compute_eval_metrics(model, loader, device, data_len, t2t, use_tqdm=True)

        meta = metrics.pop('meta')

        # Report
        for metric_name, score in metrics.items():
            print(f'{metric_name}: {score:.4f}')

        # Save
        if args.pred_csv:
            save_csv(meta, args.pred_csv, index=False)


@torch.inference_mode()
def compute_eval_metrics(model, loader, device, size, t2t, use_tqdm=False) -> Dict[str, Any]:
    """
    Computes evaluation metrics on validation/test set.

    :param model: model to evaluate
    :param loader: validation/test set Dataloader
    :param device: model device (cuda/cpu)
    :param size: no. of samples for eval
    :param t2t: indicates text-to-text model
    :param use_tqdm: show progress bar
    :return: metrics (loss, accuracy)
    """
    def unpack(_labels: str) -> List[Any]:
        """ Ex: '1,2,3' --> [1,2,3] """
        _labels = _labels.split(',')
        _labels = [int(l) if l.isdigit() else l
                   for l in _labels]
        return _labels

    batch_size = loader.batch_size

    if use_tqdm:
        loader = tqdm(loader)

    model.eval()

    d_acc = []
    d_conf = []
    d_loss = []

    m_probs = []
    m_trues = []
    m_preds = []

    n = 0
    for batch in loader:
        # Load
        labels_b = batch.pop('labels')

        batch['inputs'] = {k: v.to(device) for k, v in batch['inputs'].items()}

        # Inference
        logits_b = model(**batch)

        for logits, labels in zip(logits_b, labels_b):
            # Meta
            m_trues += [labels]

            # Labels
            labels = unpack(labels)

            # T2T
            if t2t:
                # Loss
                d_loss += [-1]

                # Prediction
                pred = logits

            # MLM
            else:
                labels = torch.tensor(labels).to(device)

                # Loss
                for label in labels:
                    loss = F.cross_entropy(logits, label).item()
                    d_loss += [loss]

                # Prediction
                pred = logits.argmax()

                # Distribution
                prob = F.softmax(logits)

                # Conf: sum(P(c) | c ∈ ground-truth)
                score = prob[labels].sum().item()
                d_conf += [score]

                # Meta
                m_preds += [pred.item()]
                m_probs += [','.join([str(p) for p in prob.tolist()])]

            # *Acc: pred ∈ ground-truth
            correct = (1 if pred in labels else 0)
            d_acc += [correct]

        # progress
        n += batch_size
        if n >= size:
            break

    # Metrics
    loss = np.mean(d_loss)
    acc = np.mean(d_acc)
    conf = np.mean(d_conf)

    meta = dict(true=m_trues, pred=m_preds, prob=m_probs)

    metrics = {'loss': loss, 'accuracy': acc, 'confidence': conf, 'meta': meta}

    return metrics


if __name__ == '__main__':
    main()
