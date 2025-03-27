import open3d as o3d


def convert_ply_to_pcd(ply_file, pcd_file):
    point_cloud = o3d.io.read_point_cloud(ply_file)

    o3d.io.write_point_cloud(pcd_file, point_cloud)


# 使用示例
ply_file_path = '.../point_cloud/xxx.ply'
pcd_file_path = 'D:/pyt_chenjb/papr-main/point_cloud/xxx.pcd'
convert_ply_to_pcd(ply_file_path, pcd_file_path)
