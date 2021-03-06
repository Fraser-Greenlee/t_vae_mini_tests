import random
import string

import datasets


ALL_CHARS = string.ascii_letters + string.digits


class LetterOrderDataset(datasets.GeneratorBasedBuilder):

    def __init__(self, num_chars=None, seq_len=None, num_rows=10_000, **kwargs):
        assert num_chars is not None
        assert seq_len is not None
        assert num_chars <= len(ALL_CHARS)
        assert num_rows > 0
        self.num_chars = num_chars
        self.seq_len = seq_len
        self.chars = ALL_CHARS[:self.num_chars]
        self.num_rows = num_rows
        super().__init__(**kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    'text': datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, _dl_manager, **kwargs):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION),
        ]

    def _generate_examples(self):
        row_num = 0
        for _ in range(self.num_rows):
            chars = [random.choice(self.chars) for _ in range(self.seq_len)]
            chars = sorted(chars)
            text = ''.join(chars)
            yield row_num, {'text': text}
            row_num += 1
