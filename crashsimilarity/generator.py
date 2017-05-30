import random


class Generator(object):
    def __init__(self, model, vocab, pos2vocab):
        self._model = model
        self._vocab = vocab
        self._pos2vocab = pos2vocab

    def change(self, trace, idx):
        other = trace[:]
        for i in idx:
            t = trace[i]
            if t in self._vocab:
                similar = self._model.most_similar(str(self._vocab[t]))
                for w, _ in similar:
                    if int(w) in self._pos2vocab:
                        x = self._pos2vocab[int(w)]
                        if not x.endswith('________'):
                            other[i] = x
                            break
        return other

    def generate(self, trace, count=10, percent=0.33):
        idx = []
        new = int(len(trace) * percent)
        for i in range(count):
            cur = list(set([random.randint(0, len(trace) - 1) for _ in range(new)]))
            idx.append(cur)
        rv = [self.change(trace, i) for i in idx]
        return rv
