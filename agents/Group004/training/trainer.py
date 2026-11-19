import os
import time

import torch
import torch.nn.functional as F

from net import HexNet, N


class Trainer:
    def __init__(self, n=N, channels=64, blocks=6, lr=1e-3, weight_decay=1e-4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = HexNet(n=n, channels=channels, blocks=blocks).to(self.device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)


    def train(self, x, target_pi, target_v, epochs = 2, batch_size=256):
        self.net.train()
        M = x.shape[0]
        if M == 0: return []
        history = []
        for epoch in range(epochs):
            t0 = time.perf_counter()
            perm = torch.randperm(M)
            p_sum = v_sum = 0.0
            for i in range(0, M, batch_size):
                sample_ids = perm[i:i+batch_size]
                board = x[sample_ids].to(self.device)
                pis = target_pi[sample_ids].to(self.device)
                vs = target_v[sample_ids].to(self.device).float().view(-1)

                logits, v = self.net(board)
                p_loss = -(pis * F.log_softmax(logits, dim=1)).sum(1).mean()
                v_loss = F.mse_loss(v, vs)
                loss = p_loss + v_loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                p_sum += p_loss.item() * len(idx)
                v_sum += v_loss.item() * len(idx)
            row = {"epoch": epoch + 1, "p_loss": p_sum / M, "v_loss": v_sum / M,
                   "secs": time.perf_counter() - t0}
            history.append(row)
            if log:
                log(f"epoch {row['epoch']}/{epochs}  policy {row['p_loss']:.4f}  "
                    f"value {row['v_loss']:.4f}  ({row['secs']:.1f}s, {M} positions)")

        return history


    @torch.no_grad()
    def predict(self, x, mask):
        self.net.eval()
        x = x.to(self.device)
        mask = mask.to(self.device) #mask to check for legal moves
        logits, value = self.net(x)
        logits = logits.masked_fill(~legal_mask, float("-inf"))
        return F.softmax(logits, dim=1).cpu(), value.cpu()

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        torch.save({"net": self.net.state_dict(),
                    "opt": self.opt.state_dict(),
                    "config": self.config}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model at {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        if checkpoint.get("config", self.config) != self.config:
            raise ValueError(f"Checkpoint is {checkpoint['config']}, "
                             f"this Trainer is {self.config}")
        self.net.load_state_dict(checkpoint["net"])
        self.opt.load_state_dict(checkpoint["opt"])



