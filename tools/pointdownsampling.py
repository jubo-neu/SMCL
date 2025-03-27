import open3d as o3d
import numpy as np
import torch


def downsample_to_exact_count(pcd, target_count):
    while len(pcd.points) > target_count:
        points_to_remove = len(pcd.points) - target_count
        indices = np.random.choice(len(pcd.points), size=points_to_remove, replace=False)
        pcd.points = o3d.utility.Vector3dVector(np.delete(np.asarray(pcd.points), indices, axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.delete(np.asarray(pcd.colors), indices, axis=0))
    return pcd

pcd = o3d.io.read_point_cloud(".../point_cloud/xxx.pcd")

original_point_count = len(pcd.points)

target_point_count = 10000

initial_ratio_estimate = int(np.floor(original_point_count / target_point_count))


downsampled_pcd = pcd.uniform_down_sample(initial_ratio_estimate)

if len(downsampled_pcd.points) <= target_point_count:
    print("no need")
else:
    downsampled_pcd = downsample_to_exact_count(downsampled_pcd, target_point_count)

final_point_count = len(downsampled_pcd.points)
print(f"include {final_point_count} points")

o3d.io.write_point_cloud("xxx.pcd", downsampled_pcd)

print("over")
o3d.visualization.draw_geometries([downsampled_pcd])
