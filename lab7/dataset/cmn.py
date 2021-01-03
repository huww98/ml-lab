import logging
from typing import List, Sequence
from dataclasses import dataclass
import re

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from . import EOS_INDEX, SPECIAL_TOKENS

logger = logging.getLogger(__name__)

def collate_fn(batch):
    langs = zip(*batch)
    source, target = (list(torch.tensor(sentence) for sentence in l) for l in langs)
    source_lens = torch.tensor(list(sentence.size(0) for sentence in source))
    return pad_sequence(source), source_lens, pad_sequence(target)

def build_loader(dataset, config, split):
    return DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=DistributedSampler(
            dataset=dataset,
            shuffle=(split == 'train'),
        )
    )

def get_splits(dataset, config):
    train_weight = config['train_weight']
    eval_weight = config['eval_weight']
    split_seed = config['split_seed']

    train_len = int(len(dataset) / (train_weight + eval_weight) * train_weight)
    eval_len = len(dataset) - train_len
    logger.info('Random split with seed %d, train %d samples, evaluate %d samples', split_seed, train_len, eval_len)
    train_split, eval_split = random_split(
        dataset=dataset,
        lengths=[train_len, eval_len],
        generator=torch.Generator().manual_seed(split_seed))
    return (build_loader(train_split, config['train'], 'train'),
        build_loader(eval_split, config['eval'], 'eval'))

ZH_REPLACE_END = {
    '.': '。',
}
ZH_REPLACE_ALL = {
    '?': '？',
    '!': '！',
    ',': '，',
}

EN_REPLACE_RE = [
    re.compile(r'(?<=\w)([,.?!]|\'s)(?=\s+(\w|")|$)'),
    re.compile(r'(?<=\d)(st|nd|th|%)'),
    re.compile(r'($)(?=\d)'),
]

class Lang:
    def __init__(self, config):
        self.name = config['name']
        self.split_method = config['split']
        self.word2index = {}
        self.index2word = list(SPECIAL_TOKENS)

    def add_sentence(self, sentence: str):
        if self.split_method == 'word':
            sentence = sentence.lower()
            sentence = sentence.replace('"', ' " ')
            for r in EN_REPLACE_RE:
                sentence = r.sub(r' \1', sentence)
            words = sentence.split()
        elif self.split_method == 'char':
            for k, v in ZH_REPLACE_END.items():
                if sentence.endswith(k):
                    sentence = sentence[:-1] + v
            for k, v in ZH_REPLACE_ALL.items():
                sentence = sentence.replace(k ,v)
            words = list(sentence)
        else:
            raise ValueError(f'Unkonwn split method "{self.split_method}"')

        indices = []
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
            indices.append(self.word2index[word])
        return indices

    def decode_sentence(self, ids: List[int]):
        try:
            ids = ids[:ids.index(EOS_INDEX)]
        except ValueError:
            pass
        if self.split_method == 'word':
            return ' '.join(self.index2word[i] for i in ids)
        else:
            return ''.join(self.index2word[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.index2word)

@dataclass
class Sample:
    source: List[int]
    target: List[int]

class CMN(Dataset):
    def __init__(self, config):
        self.langs = [Lang(l) for l in config['langs']]
        data_path = config['data_path']
        max_len = config['max_len']
        source_lang_idx = config['source_lang']
        target_lang_idx = 1 - source_lang_idx
        self.source_lang = self.langs[source_lang_idx]
        self.target_lang = self.langs[target_lang_idx]

        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                indices_pair = []
                for lang_str, l in zip(line.split('\t'), self.langs):
                    indices = l.add_sentence(lang_str)
                    indices.append(EOS_INDEX)
                    indices_pair.append(indices)
                source_ids = indices_pair[source_lang_idx]
                if len(source_ids) > max_len:
                    continue
                self.samples.append(Sample(source=source_ids, target=indices_pair[target_lang_idx]))

        for l in self.langs:
            logger.info('Language %s: vocabulary size %d', l.name, l.vocab_size)
            # print(l.index2word)
        logger.info('Translate from %s to %s', self.source_lang.name, self.target_lang.name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        return s.source, s.target
