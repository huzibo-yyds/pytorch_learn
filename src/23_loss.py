import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, [1, 1, 1, 3])
targets = torch.reshape(targets, [1, 1, 1, 3])

loss = L1Loss(reduction="mean")
result = loss(inputs, targets)
print(result)
# sum:tensor(2.)    mean:tensor(0.6667)

# MSE 均方误差
loss_mse = MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)
# tensor(1.3333)

#
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
# tensor(1.1019)
