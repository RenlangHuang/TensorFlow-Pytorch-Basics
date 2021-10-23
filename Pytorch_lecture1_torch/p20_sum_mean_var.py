import torch

# sum
a = torch.ones((2, 3)).float()
print(torch.sum(a),a.sum())
print(torch.sum(a, dim=0))
print(torch.sum(a, dim=1))

# mean
a[1] += 1; print(a)
print(torch.mean(a),a.mean())
print(torch.mean(a, dim=0))
print(torch.mean(a, dim=1))

# var
print(torch.var(a),a.var())
print(torch.var(a, dim=0))
print(torch.var(a, dim=1))