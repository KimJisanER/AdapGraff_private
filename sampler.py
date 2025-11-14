# sampler.py
import math, random
from torch.utils.data import Sampler

class ShardAwareBatchSampler(Sampler):
    """
    Dataset 요구 속성:
      - _shard_lengths: List[int]
    배치: 동일 shard 내 연속 offset으로 구성 (I/O 캐시 최적화)
    """
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True, seed=42):
        self.ds = dataset
        self.bs = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = int(seed)

        self.shard_lengths = list(dataset._shard_lengths)
        self.num_shards = len(self.shard_lengths)

    def __iter__(self):
        g = random.Random(self.seed)
        shard_ids = list(range(self.num_shards))
        if self.shuffle:
            g.shuffle(shard_ids)

        base = 0  # global index base for current shard
        # 미리 각 shard의 global 시작 인덱스 계산
        # (동적 누적합을 피하고 한번에 계산)
        bases = []
        acc = 0
        for L in self.shard_lengths:
            bases.append(acc)
            acc += L

        for sid in shard_ids:
            L = self.shard_lengths[sid]
            start_indices = list(range(0, (L // self.bs) * self.bs, self.bs))
            if self.shuffle:
                g.shuffle(start_indices)

            base = bases[sid]
            for s in start_indices:
                yield list(range(base + s, base + s + self.bs))

            tail = L % self.bs
            if (not self.drop_last) and tail > 0:
                yield list(range(base + (L - tail), base + L))

    def __len__(self):
        total = 0
        for L in self.shard_lengths:
            q, r = divmod(L, self.bs)
            total += q + (0 if self.drop_last or r == 0 else 1)
        return total