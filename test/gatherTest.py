import torch
import numpy as np

A_list = np.arange(1, 10).reshape((3, 3))

A = torch.tensor(A_list)

print(A)

number = torch.tensor([1, 2, 3])
index = torch.nonzero(number > 1).squeeze()
print(A.index_select(0, index))