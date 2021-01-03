import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_EPOCH_RE = re.compile(r'train epoch finished\s+\[\d+\]\s+loss\s+[\d.]+\s+\(([\d.]+)\)\s+acc\s+[\d.]+\s+\(([\d.]+)\)')
EVAL_EPOCH_RE = re.compile(r'evaluate finished\s+loss\s+[\d.]+\s+\(([\d.]+)\)\s+acc\s+[\d.]+\s+\(([\d.]+)\)\s+BLEU=([\d.]+)')

def plot(log_path: Path):
    with log_path.open('r') as f:
        logs = f.read()

    train_loss = []
    train_acc = []
    for loss, acc in TRAIN_EPOCH_RE.findall(logs):
        train_loss.append(float(loss))
        train_acc.append(float(acc))
    eval_loss = []
    eval_acc = []
    for loss, acc, bleu in EVAL_EPOCH_RE.findall(logs):
        eval_loss.append(float(loss))
        eval_acc.append(float(acc))

    x = range(1, len(train_loss) + 1)
    plt.figure(figsize=(4, 2.5))
    plt.xlabel('#epoch')
    plt.ylabel('loss')
    plt.plot(x, train_loss, label='train')
    plt.plot(x, eval_loss, label='test')
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig('report/figures/loss.pdf')
    plt.close()

    plt.figure(figsize=(4, 2.5))
    plt.xlabel('#epoch')
    plt.ylabel('Token accuracy (%)')
    plt.plot(x, [v * 100 for v in train_acc], label='train')
    plt.plot(x, [v * 100 for v in eval_acc], label='test')
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig('report/figures/acc.pdf')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type=Path)
    args = parser.parse_args()
    plot(args.log_path)
