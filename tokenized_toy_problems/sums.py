import random
from typing import List


class SumsDataset():

    def __init__(self, max_units, operators):
        self.max_units, self.operators = max_units, operators
        self.n_tokens = 10 + len(self.operators) + 1

    def decode(self, sample: List[int]):
        words = [str(v) if v < 10 else (['='] + [op.__name__ for op in self.operators])[v-10] for v in sample]
        return ' '.join(words)

    def generate_tokenized_sample(self):
        result = 10 ** self.max_units
        while result > 10 ** self.max_units:
            a, b = random.randint(0, (10 ** self.max_units)-1)
            op = random.sample(self.operators)
            result = op(a, b)

        return [int(c) for c in str(a)] + [self.operators.index(op)] + [int(c) for c in str(b)] + [10 + len(self.operators) + 1] + [int(c) for c in str(result)]
