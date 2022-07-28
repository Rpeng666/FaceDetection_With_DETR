from scipy.optimize import linear_sum_assignment
import torch

a = torch.rand((4, 5))

print(linear_sum_assignment(a))