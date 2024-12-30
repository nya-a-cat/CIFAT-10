import torch

ln1 = torch.nn.Linear(10, 10)
ln2 = torch.nn.Linear(2,2)
tor1 = torch.rand(1,2)
print(tor1.sum())

# m = torch.nn.GLU()
# input = torch.randn(4, 2)
# output = m(input)
# print(output)

