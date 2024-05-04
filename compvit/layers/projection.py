import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, depth=1, normalize=True):
        super().__init__()

        if hidden_dim is None:

            class LinearProjector(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.proj = nn.Linear(in_dim, out_dim)

                def forward(self, x):
                    x = self.proj(x)
                    if normalize:
                        x = F.normalize(x, dim=-1)
                    return x
            self.proj = LinearProjector()
        else:
            class Projector(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.proj = nn.Sequential(
                        nn.Linear(in_dim, in_dim, bias=False),
                        nn.BatchNorm1d(in_dim), 
                        nn.ReLU(),
                        nn.Linear(in_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=False), 
                    )
                def forward(self, x):
                    x = self.proj(x)
                    x = F.normalize(x, dim=-1)
                    return x
            self.proj = Projector()

    def forward(self, x):
        return self.proj(x)
