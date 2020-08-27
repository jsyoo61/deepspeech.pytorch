def forward(inputs, input_sizes, edge_model):
    inputs = inputs.to(next(edge_model.parameters()).device)
    out, output_sizes = edge_model(inputs, input_sizes)
    return
import torch
import torch.nn as nn
from itertools import starmap
class m(nn.Module):
    def __init__(self):
        super(m, self).__init__()
        self.x = nn.Linear(5,10)

    def forward(self, x):
        return self.x(x), 2

edge_model_list = [m().cuda() for i in range(4)]
i=torch.ones(16,5)
i = list(i)
from multiprocessing import Pool
results = starmap(f, zip(i, edge_model_list))
results
results = map(f, zip(i, edge_model_list))
results = map(f, i, edge_model_list)

l=list(results)
l[0]
with Pool() as pool:
    results = pool.starmap(f, zip(i, edge_model_list))
def f(inputs, edge_model):
    inputs = inputs.to(next(edge_model.parameters()).device)
    output = edge_model(inputs)
    return output
a
