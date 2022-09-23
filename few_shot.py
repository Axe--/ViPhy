"""
Trains model for a single epoch on `k` samples,
and reports the performance on test-set.
"""
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from dataloader import ViPhyDataset
from main import compute_eval_metrics
from models import MaskedLM, Text2TextLM, UnifiedQA
from utils import save_csv, csv2list, str2bool as s2b


def few_shot():
    parser = argparse.ArgumentParser(description='ViPhy: Few-Shot Eval')

    # Model params
    parser.add_argument('--model',      type=str, help='Transformer backbone', required=True)
    parser.add_argument('--VL',         type=s2b, help='vision + language input', default='F')

    # Data params
    parser.add_argument('--data_dir',   type=str, help='path to dataset directory', required=True)
    parser.add_argument('--k',          type=int, help='train set size `K`', required=True)

    # Train params
    parser.add_argument('--lr',         type=float, help='learning rate', default=1e-4)
    parser.add_argument('--epochs',     type=int,   help='number of epochs', default=1)
    parser.add_argument('--batch_size', type=int,   help='batch size', default=8)

    # GPU params
    parser.add_argument('--gpus',       type=str, help='GPU Device ID', default='0')
    parser.add_argument('--amp',        type=s2b, help='Automatic-Mixed Precision (T/F)', default='T')

    # Misc params
    parser.add_argument('--num_workers', type=int, help='worker threads for Dataloader', default=0)
    parser.add_argument('--pred_csv',    type=str, help='prediction on `Test`set (csv)')

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

    # Model
    if t2t:
        if _type == 'QA':
            model = UnifiedQA(args.model, device)
        else:
            model = Text2TextLM(args.model, device)
    else:
        model = MaskedLM(args.model, num_classes, device, vl=args.VL)

    model.train()

    # Dataset
    train_dataset = ViPhyDataset(args.data_dir, args.model, model.tokenizer, split=f'train_{args.k}')

    # Dataloader
    loader_params = dict(batch_size=batch_size,
                         shuffle=True,
                         drop_last=False,
                         num_workers=args.num_workers)

    train_loader = DataLoader(train_dataset, **loader_params)

    print(f'\n#Train: {len(train_dataset)}\n')

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer.zero_grad()

    scaler = GradScaler(enabled=args.amp)

    # Init params
    start_epoch = 1
    curr_step = 1

    # Train
    for epoch in range(start_epoch, n_epochs + 1):
        for batch in tqdm(train_loader):
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

            curr_step += 1

    # Dataset
    test_dataset = ViPhyDataset(args.data_dir, args.model, model.tokenizer, split='test')
    test_len = len(test_dataset)

    # Dataloader
    test_loader = DataLoader(test_dataset, batch_size**2, num_workers=args.num_workers)

    # Inference
    metrics = compute_eval_metrics(model, test_loader, device, test_len, t2t, use_tqdm=True)

    meta = metrics.pop('meta')

    # Report
    print(f'\nModel: {args.model}')
    print(f'#Test (k={args.k}): {test_len}\n')

    for metric_name, score in metrics.items():
        print(f'{metric_name}: {score:.4f}')

    # Save
    if args.pred_csv:
        save_csv(meta, args.pred_csv, index=False)


if __name__ == '__main__':
    few_shot()
