import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.optim
import torchtext.data.metrics
from hutils.arguments import get_argparser, resolve_runtime_env
from hutils.checkpoint import CheckpointManager
from hutils.logging import init_logging
from hutils.meters import AverageMeter
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import PackedSequence

from .dataset import EOS_INDEX, PAD_INDEX
from .dataset.cmn import CMN, get_splits
from .model import Seq2Seq

logger = logging.getLogger(__name__)


def accuracy(predict, target):
    non_pad_mask = target != PAD_INDEX
    correct_mask = torch.logical_and(non_pad_mask, predict.argmax(dim=2) == target)
    return torch.sum(correct_mask) / torch.sum(non_pad_mask)

def valid_ids(ids: torch.Tensor):
    ids = ids.tolist()
    try:
        ids = ids[:ids.index(EOS_INDEX)]
    except ValueError:
        pass
    return [str(i) for i in ids]

@torch.no_grad()
def evaluate(model: nn.Module, criterion, data_loader, save_path: Optional[Path] = None):
    device = next(model.parameters()).device
    loss_meter = AverageMeter('loss', device=device)
    acc_meter = AverageMeter('acc', device=device)
    model.eval()
    result_f = save_path.open('w') if save_path is not None else None
    all_decoded = []
    all_target = []
    for source, source_len, target in data_loader:
        source_device: torch.Tensor = source.to(device=device, non_blocking=True)
        target_device: torch.Tensor = target.to(device=device, non_blocking=True)

        predict, decoded = model(source_device, source_len, None)
        target_len = target.size(0)
        predict_len = predict.size(0)
        batch_size = source.size(1)
        vocab_len = predict.size(2)
        if predict_len > target_len:
            predict = predict[:target_len]
        elif predict_len < target_len:
            padding = torch.zeros((1,), device=predict.device).expand((target_len - predict_len, batch_size, vocab_len))
            predict = torch.cat((predict, padding))

        loss: torch.Tensor = criterion(predict.reshape(-1, vocab_len), target_device.reshape(-1))
        loss_meter.update(loss, n=batch_size)
        acc_meter.update(accuracy(predict, target_device), n=batch_size)

        decoded = decoded.cpu()
        for i in range(batch_size):
            all_decoded.append(valid_ids(decoded[:, i]))
            all_target.append([valid_ids(target[:, i])])

        if result_f is not None:
            dataset = data_loader.dataset.dataset
            for i in range(batch_size):
                source_sentence = dataset.source_lang.decode_sentence(source[:, i].tolist())
                translated_sentence = dataset.target_lang.decode_sentence(decoded[:, i].tolist())
                result_f.write(f'{source_sentence}\t{translated_sentence}\n')
    loss_meter.sync_distributed()
    acc_meter.sync_distributed()

    # print(all_decoded, all_target)
    bleu = torchtext.data.metrics.bleu_score(all_decoded, all_target)
    logger.info(f'evaluate finished\t{loss_meter}\t{acc_meter}\tBLEU={bleu}')
    return bleu

def main_worker(local_rank: int, args, config: Dict):
    if local_rank == 0:
        init_logging(args, tqdm=True)

    if args.cpu:
        device = torch.device('cpu')
    else:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=len(args.gpus),
            rank=local_rank,
        )
        device = torch.device(f'cuda:{args.gpus[local_rank]}')
        torch.cuda.set_device(device)

    dataset = CMN(config['dataset'])
    train_loader, eval_loader = get_splits(dataset, config=config['dataloader'])
    model = Seq2Seq(config['model'], dataset.source_lang.vocab_size, dataset.target_lang.vocab_size).to(device=device)
    model = DistributedDataParallel(
        module=model,
        device_ids=[device],
    )
    checkpoint = CheckpointManager(experiment_dir=args.experiment_dir, save_interval=5)
    best_bleu = 0
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX).to(device=device)
    log_interval = config['log_interval']
    for e in range(config['epoch']):
        loss_meter = AverageMeter('loss', device=device)
        acc_meter = AverageMeter('acc', device=device)
        model.train()
        for step, (source, source_len, target) in enumerate(train_loader):
            source: torch.Tensor = source.to(device=device, non_blocking=True)
            target: torch.Tensor = target.to(device=device, non_blocking=True)

            optimizer.zero_grad()
            predict, decoded = model(source, source_len, target)
            vocab_len = predict.size(2)
            loss: torch.Tensor = criterion(predict.reshape(-1, vocab_len), target.reshape(-1))
            loss.backward()
            optimizer.step()

            if step > 0 and (step - 1) % log_interval == 0:
                # Do logging as late as possible. this will force CUDA sync.
                # Log numbers from last iteration, just before update
                logger.info(f'train [{step - 1}]\t{loss_meter}\t{acc_meter}')

            batch_size = source.size(1)
            loss_meter.update(loss, n=batch_size)
            acc_meter.update(accuracy(predict, target), n=batch_size)

        logger.info(f'train epoch finished [{e}]\t{loss_meter}\t{acc_meter}')
        eval_bleu = evaluate(model, criterion, eval_loader)
        if local_rank == 0:
            is_best = eval_bleu > best_bleu
            if is_best:
                best_bleu = eval_bleu
            checkpoint.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': e,
                'best_bleu': best_bleu,
            }, is_best=is_best, epoch=e)

    best_checkpoint = torch.load(args.experiment_dir / 'model_best.pth.tar', map_location=device)
    model.load_state_dict(best_checkpoint['model'])
    evaluate(model, criterion, eval_loader, args.run_dir / f'result_rank_{local_rank}.txt')


def main():
    argparser = get_argparser()
    args = argparser.parse_args()
    args, config = resolve_runtime_env(args)

    if args.cpu:
        logger.info('单进程模式')
        main_worker(0, args, config)
    else:
        logger.info('多进程模式')
        mp.spawn(main_worker, args=(args, config), nprocs=len(args.gpus))


if __name__ == "__main__":
    main()
