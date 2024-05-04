import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryBank(nn.Module):
    def __init__(self, dim, banksize) -> None:
        super().__init__()
        self.bank_size = banksize
        self.dim = dim
        self.bank = nn.Parameter(torch.randn(banksize, dim), requires_grad=True)

        self.selector = nn.Linear(dim, banksize)

    def forward(self, x, num_compressed_tokens):
        
        x, bank_token = x[:, 0:-1], x[:, -1]
        selector = self.selector(bank_token)
        selector = F.gumbel_softmax(selector, tau=1)
        selector = selector.argsort(descending=True)
        return x, self.bank[selector[:, :num_compressed_tokens]]