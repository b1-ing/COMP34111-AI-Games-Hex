import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, n=11):
        self.capacity = capacity
        self.boards = np.zeros((capacity, n, n), np.int8)
        self.pis = np.zeros((capacity, n, n), np.float16)
        self.zs = np.zeros(capacity, np.int8)
        self.next = 0
        self.size = 0


    def add(self, positions):
        for board, pi, z in positions:
            i = self.next
            self.boards[i], self.pis[i], self.zs[i] = board, pi, z
            self.next = (i + 1) % self.capacity  # wrap around: overwrite the oldest
            self.size = min(self.size + 1, self.capacity)


    def __len__(self):
        return self.size

    def sample_all(self):
        """Returns (x, target_pi, target_z) as training tensors."""
        b = self.boards[:self.size]
        x = np.stack([b == 1, b == 2, np.ones_like(b, bool)], axis=1).astype(np.float32)
        return (torch.from_numpy(x),
                torch.from_numpy(self.pis[:self.size].astype(np.float32)),
                torch.from_numpy(self.zs[:self.size].astype(np.float32)))


    def save(self, path):
        """Write to a temp file, then rename: a job killed mid-save leaves the old file intact."""
        tmp = path + ".tmp.npz"  # savez adds .npz if missing, so name it ourselves
        np.savez_compressed(tmp, boards=self.boards, players=self.players, pis=self.pis,
                            zs=self.zs, next=self.next, size=self.size)
        os.replace(tmp, path)

    def load(self, path):
        with np.load(path) as d:  # plain numeric arrays: no allow_pickle needed
            if d["boards"].shape != self.boards.shape:
                raise ValueError(f"Buffer file has shape {d['boards'].shape}, "
                                 f"expected {self.boards.shape} (capacity or board size changed?)")
            self.boards[:] = d["boards"]
            self.players[:] = d["players"]
            self.pis[:] = d["pis"]
            self.zs[:] = d["zs"]
            self.next = int(d["next"])
            self.size = int(d["size"])


