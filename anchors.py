import torch

model = torch.load('best2.pt')['model']

m = model.model[-1]  # Detect()
m.anchors  # in stride units
m.anchor_grid  # in pixel units

print(m.anchor_grid.view(-1,2))
