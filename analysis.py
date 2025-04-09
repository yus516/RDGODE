import torch.optim as optim
import util
from reaction_diffusion import *
import numpy as np


device = "cuda:1"
model = reaction_diffusion_gcn(device, True, 12)
model.load_state_dict(torch.load("save_models/PEMSD4copy/exp0_best_2.18.pth"))

weight_diff = model.weight_diff.cpu().detach().numpy()
# weight_diff_a = model.weight_diff_a.cpu().detach().numpy()
bias_diff = model.bias_diffusion.cpu().detach().numpy()

weight_react = model.weight_react.cpu().detach().numpy()
# weight_react_a = model.weight_react_a.cpu().detach().numpy()
bias_react = model.bias_diffusion.cpu().detach().numpy()

np.save("sym-weight-diff.npy", weight_diff)
# np.save("unsym-weight-a-diff.npy", weight_diff_a)
np.save("sym-bias-diff.npy", bias_diff)

np.save("sym-weight-react.npy", weight_react)
# np.save("unsym-weight-a-react.npy", weight_react_a)
np.save("sym-bias-react.npy", bias_react)

print("finish")
