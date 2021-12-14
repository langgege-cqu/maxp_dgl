import torch
import torch.nn as nn

a = torch.ones((10, 100))
b = torch.ones((10, 100))
c = a + b
c = c.to('cuda')
print(c)