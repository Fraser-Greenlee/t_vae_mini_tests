import random
import string
from typing import List

import datasets


ALL_CHARS = [c for c in string.ascii_letters]


class LetterOrderDataset(datasets.GeneratorBasedBuilder):

    def __init__(self, num_chars=None, seq_len=None, **kwargs):
        assert(num_chars is not None)
        assert(seq_len is not None)
        assert num_chars <= len(ALL_CHARS)
        self.num_chars, self.seq_len = num_chars, seq_len
        self.tokens = ALL_CHARS[:self.num_chars]
        super().__init__(**kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    'input_ids': datasets.Value("int32"),
                }
            ),
        )

    def _split_generators(self):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN),
            datasets.SplitGenerator(name=datasets.Split.TEST),
        ]

    def decode(self, sample: List[int]):
        words = [ALL_CHARS[v] for v in sample]
        return ''.join(words)

    def generate_sample(self):
        chars = [random.sample(self.tokens) for _ in range(self.seq_len)]
        return sorted(chars)

    def generate_tokenized_sample(self):
        return [ALL_CHARS.index(c) for c in self.generate_sample()]

    def _generate_examples(self):
        row_num = 0
        while True:
            row = self.generate_tokenized_sample()
            yield row_num, {'input_ids': row}
