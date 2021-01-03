import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .dataset import PAD_INDEX, SOS_INDEX, EOS_INDEX
MAX_LENGTH = 16

class EncoderRNN(nn.Module):
    def __init__(self, config, source_vocab_size):
        super().__init__()
        hidden_size = config['hidden_size']

        self.embedding = nn.Embedding(source_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input):
        embedded = PackedSequence(self.embedding(input[0]), *input[1:])
        output, hidden = self.gru(embedded)
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, config, target_vocab_size):
        super().__init__()
        hidden_size = config['hidden_size']
        dropout_p = config['dropout']
        self.max_length = config['max_source_len']
        self.max_decode_length = config['max_decode_len']

        self.embedding = nn.Embedding(target_vocab_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, encoded: PackedSequence, hidden: torch.Tensor, target):
        encoded, encoded_lens = pad_packed_sequence(encoded, total_length=self.max_length)
        encoded: torch.Tensor  # (max_length, batch_size, hidden_size)
        batch_size = encoded_lens.size(0)

        decoder_input = torch.tensor([SOS_INDEX], device=encoded.device).expand(batch_size)

        outputs = []
        decoded = []
        finished = torch.full((batch_size,), False, dtype=torch.bool, device=self.out.weight.device)
        max_length = self.max_decode_length if target is None else target.size(0)
        for i in range(max_length):
            embedded = self.embedding(decoder_input)  # (batch_size, hidden_size)
            embedded = self.dropout(embedded)

            attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden.view(batch_size, -1)), dim=1)), dim=1)  # (batch_size, max_length)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoded.transpose(0, 1)).squeeze(1)  # (batch_size, hidden_size)

            output = torch.cat((embedded, attn_applied), dim=1)
            output = self.attn_combine(output)  # (batch_size, hidden_size)

            output = F.relu(output)
            output, hidden = self.gru(output.unsqueeze(0), hidden)
            output = output.squeeze(0)
            output: torch.Tensor = self.out(output)  # (batch_size, target_vocab_size)
            outputs.append(output)

            if target is not None:
                decoder_input = target[i]
            else:
                topi = output.argmax(dim=1)
                finished = torch.logical_or(finished, topi == EOS_INDEX)
                decoded.append(topi)
                if finished.all():
                    break
                else:
                    decoder_input = topi

        decoded = torch.stack(decoded) if target is None else None   # (target_length, batch_size)
        outputs = torch.stack(outputs)  # (target_length, batch_size, target_vocab_size)
        return outputs, decoded

class Seq2Seq(nn.Module):
    def __init__(self, config, source_vocab_size: int, target_vocab_size: int):
        super().__init__()
        self.encoder = EncoderRNN(config['encoder'], source_vocab_size)
        self.decoder = AttnDecoderRNN(config['decoder'], target_vocab_size)

    def forward(self, source, source_lens, target):
        source = pack_padded_sequence(source, source_lens.cpu(), enforce_sorted=False)
        encoded, hidden = self.encoder(source)
        output = self.decoder(encoded, hidden, target)
        return output
