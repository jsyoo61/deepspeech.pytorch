# %%
import torch
import torch.nn as nn

# %%
def aggregation(model_source, model_target, weight = None):
    '''
    model_source: Single nn.Model instance
    model_target: List of nn.Model instances
    weights: default = None
        List of numbers. Must match the number of model_source.
        If None, then weight is set as 1/len(model_source)
    '''
    if weight == None:
        weight = [1/len(model_source)] * len(model_source)
    assert len(model_source) == len(weight), "length of model_source(%s) and weight(%s) does not match"%(len(model_source), len(weight))

    for parameters in zip(model_target.parameters(), *[model.parameters() for model in model_source]):
        p_trg = parameters[0]
        p_src_tuple = parameters[1:]
        # model_target's device
        device = p_trg.device

        # 1. Reset
        p_trg.data[:] = 0

        # 2. Add weighted sum
        for p_src, w in zip(p_src_tuple, weight):
            p_trg.data += w * p_src.data.to(device)

# %%
class m(nn.Module):
    def __init__(self):
        super(m, self).__init__()
        self.x = nn.Linear(10,10)
# %%
model_source = [m().cuda() for i in range(5)]
model_target = m()
for i, model in enumerate(model_source):
    model.x.weight.data[:] = i
# %%
for model in model_source:
    print(model.x.weight)
print(model_target.x.weight.data)
# %%
aggregation(model_source, model_target)
# %%
print(model_target.x.weight.data)
# %%

y.x.weight.data
weight = None
for i in zip([t.parameters() for t in model_source]):
    print(i)
torch.cuda.device_count()
