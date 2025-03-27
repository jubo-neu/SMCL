import open3d as o3d
import numpy as np
import torch


pcd = o3d.io.read_point_cloud("./xxx.pcd")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors) if pcd.has_colors() else None

tensor_points = torch.tensor(points, dtype=torch.float32)
if colors is not None:
    tensor_colors = torch.tensor(colors, dtype=torch.float32)
    tensor_data = tensor_points
    print("1", tensor_data.shape)
else:
    tensor_data = tensor_points
torch.save(tensor_data, "xxx.pth")
