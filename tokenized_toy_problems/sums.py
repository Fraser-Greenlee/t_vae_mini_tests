import random
import operator
from typing import List


OPS = {
    '*': operator.mul,
    '/': operator.floordiv,
    '+': operator.add,
    '-': operator.sub,
}


class SumsDataset():

    def __init__(self, max_str_len, operators: List[str]):
        self.max_str_len = max_str_len
        self.operators = operators
        for op_key in self.operators:
            assert(op_key in OPS)

    def str_output(self, a, op_key, b, result):
        return f'{a}{op_key}{b}={result}'

    def generate_tokenized_sample(self):
        output = None

        while output is None or len(output) > self.max_str_len:
            a, b = random.randint(0, (10 ** self.max_units)-1)
            op_key = random.sample(self.operators)
            result = OPS[op_key](a, b)
            output = f'{a}{op_key}{b}={result}'

        return output
