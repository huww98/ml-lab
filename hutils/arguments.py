
import argparse
import json
import logging
import re
import shutil
import time
from pathlib import Path

import torch
import torch.cuda

from .checkpoint import CHECKPOINT_FILENAME
from .config import get_config
from .logging import init_logging

logger = logging.getLogger(__name__)


def get_timestamp(fmt: str = '%Y%m%d_%H%M%S') -> str:
    timestamp = time.strftime(fmt, time.localtime())
    return timestamp


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='Path to config file')
    parser.add_argument('-x', '--ext-config', nargs='*',
                        default=[], help='Extra jsonnet config')
    parser.add_argument('-e', '--experiment-dir', required=True,
                        const=Path('temp') / get_timestamp(),
                        nargs=argparse.OPTIONAL,
                        type=Path,
                        help='Used to keep checkpoint, tensorboard event log, etc.')
    parser.add_argument('--continue', action='store_true', dest='cont')
    parser.add_argument('--cp', '--load-checkpoint',
                        type=Path,
                        help='Continue from the specified checkpoint')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--run-dir',
                        required=False,
                        type=Path,
                        help='Used to keep log file, code back etc. We will automatically create one if not specified.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:36987')
    parser.add_argument('--gpus', nargs='*', default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--test', action='store_true', help='只跑测试')
    parser.add_argument('--checkpoint', help='checkpoint的路径')
    return parser


RUN_DIR_NAME_REGEX = re.compile('^run_(\d+)_?')


def resolve_runtime_env(args):
    if not args.cpu:
        if not torch.cuda.is_available():
            try:
                torch.cuda.init()
            except Exception as ex:
                raise RuntimeError(
                    'CUDA is not available. To use CPU, pass "--cpu"') from ex
            assert False, "CUDA is not available, but init succeed. shouldn't be possible."

        device_count = torch.cuda.device_count()
        if args.gpus is None:
            args.gpus = list(range(device_count))

        if any(int(gpu) >= device_count for gpu in args.gpus):
            raise ValueError(
                f'GPU {",".join(args.gpus)} requested, but only {device_count} GPUs available.')

    if (not args.cont and
            args.experiment_dir is not None and
            args.experiment_dir.exists() and
            (args.experiment_dir / CHECKPOINT_FILENAME).exists()):
        if not args.force:
            raise RuntimeError(
                f'Experiment directory {args.experiment_dir} exists and contains previous checkpoint. '
                'Pass "--force" to continue'
            )

    if args.experiment_dir is not None and args.run_dir is None:
        run_id = -1
        if args.experiment_dir.exists():
            for previous_runs in args.experiment_dir.iterdir():
                match = RUN_DIR_NAME_REGEX.match(previous_runs.name)
                if match is not None:
                    run_id = max(int(match.group(1)), run_id)
        run_id += 1
        args.run_dir = args.experiment_dir / f'run_{run_id}'

    if args.run_dir.exists():
        if args.force:
            shutil.rmtree(args.run_dir)
        else:
            raise RuntimeError(
                f'Run directory {args.run_dir} exists. '
                'Pass "--force" to replace it'
            )

    config = get_config(args)
    args.run_dir.mkdir(parents=True)

    init_logging(args)

    logger.info("Resolved arguments: %s", args)
    logger.info("Resolved config: %s", json.dumps(config, indent=4))

    return args, config
