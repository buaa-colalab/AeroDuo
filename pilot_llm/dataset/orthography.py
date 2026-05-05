import math

import numpy as np
from scipy.ndimage import distance_transform_edt

FOV = 120


class Orthophoto:

    def __init__(self, granularity=0.3):
        self.granularity = granularity

    @staticmethod
    def project_images_to_3d(depth_images, bev_camera_positions):
        """ Projecting multiple depth maps onto a 3D world coordinate system """
        n, h, w = depth_images.shape
        c_x, c_y = w / 2, h / 2
        f_x = w / (2 * math.tan(math.radians(FOV / 2)))
        f_y = h / (2 * math.tan(math.radians(FOV / 2)))

        # Pre-computed intrinsic parameter inverse matrix
        intrinsic_inv = np.linalg.inv(
            np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]))

        # Inverse of the rotation matrix
        r_inv = np.linalg.inv(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))

        i, j = np.indices((h, w))  # i corresponds to y, j corresponds to x

        z = depth_images.astype(np.float32)  # (n, h, w)
        pixels_homogeneous = np.stack((j * z, i * z, z),
                                      axis=-1)  # (n, h, w, 3)
        pixels_homogeneous = pixels_homogeneous.reshape(n, h * w, 3).transpose(
            0, 2, 1)  # (n, 3, h*w)

        # 计算 3D 坐标
        transformed_points = r_inv @ intrinsic_inv @ pixels_homogeneous  # (3, 3) @ (3, 3) @ (n, 3, h*w) -> (n, 3, h*w)
        coord_3d_clouds = bev_camera_positions[:, :,
                                               None] + transformed_points  # (n, 3, 1) + (n, 3, h*w) -> (n, 3, h*w)
        coord_3d_clouds = coord_3d_clouds.transpose(0, 2, 1).reshape(
            n, h, w, 3)  # (n, h*w, 3) -> (n, h, w, 3)

        return coord_3d_clouds

    def orthorectify(self, rgb_images, coord_3d_clouds, depth_images=None):
        """
        :param rgb_images: (n, H, W, 3)
        :param coord_3d_clouds: (n, H, W, 3)
        :return: merged_orthophoto, merged_coor_map
        """
        all_x, all_y, all_z = coord_3d_clouds[..., 0].ravel(), coord_3d_clouds[
            ..., 1].ravel(), coord_3d_clouds[..., 2].ravel()
        all_rgb = rgb_images.reshape(-1, 3)
        if depth_images is not None:
            all_depth = depth_images.reshape(-1, 1)

        z_mean, z_std = np.mean(all_z), np.std(all_z)
        valid_mask = (all_z > z_mean - 4 * z_std) & (all_z
                                                     < z_mean + 4 * z_std)
        all_x, all_y, all_z, all_rgb = all_x[valid_mask], all_y[
            valid_mask], all_z[valid_mask], all_rgb[valid_mask]
        if depth_images is not None:
            all_depth = all_depth[valid_mask]

        if all_x.size == 0:
            raise ValueError("orthorectify: empty point cloud after z-filter")

        # Get global coordinate range
        min_x, min_y = np.min(all_x), np.min(all_y)
        max_x, max_y = np.max(all_x), np.max(all_y)
        start_x = np.floor(min_x / self.granularity) * self.granularity
        start_y = np.floor(min_y / self.granularity) * self.granularity
        end_x = np.ceil(max_x / self.granularity) * self.granularity
        end_y = np.ceil(max_y / self.granularity) * self.granularity

        # Create a uniform coordinate grid
        x_coords = np.arange(end_x + self.granularity, start_x,
                             -self.granularity)
        y_coords = np.arange(start_y, end_y + self.granularity,
                             self.granularity)
        coor_map = np.stack(np.meshgrid(x_coords, y_coords), axis=-1)

        # Calculate the index of each point in the grid.
        h, w = coor_map.shape[:-1]
        i = np.round((all_y - start_y) / self.granularity).astype(int)
        i = np.clip(i, 0, h - 1)
        j = w - np.round((all_x - start_x) / self.granularity).astype(int) + 1
        j = np.clip(j, 0, w - 1)
        linear_indices = i * w + j

        # Sort by z-value, ensuring that higher-value points cover lower-value points
        sort_order = np.argsort(all_z)
        sorted_linear = linear_indices[sort_order]
        sorted_rgb = all_rgb[sort_order]
        if depth_images is not None:
            sorted_depth = all_depth[sort_order]

        # Select a single pixel (the highest point)
        unique_linear, first_idx = np.unique(sorted_linear, return_index=True)
        selected_rgb = sorted_rgb[first_idx]
        if depth_images is not None:
            selected_depth = sorted_depth[first_idx]

        i_indices = unique_linear // w
        j_indices = unique_linear % w

        merged_orthophoto = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=bool)  # 标记已填充的像素点
        merged_orthophoto[i_indices, j_indices] = selected_rgb
        mask[i_indices, j_indices] = False
        if depth_images is not None:
            merged_depth = np.zeros((h, w, 1), dtype=np.float32)
            merged_depth[i_indices, j_indices] = selected_depth

        # Fill holes using nearest neighbor interpolation.
        dist, nearest_idx = distance_transform_edt(mask, return_indices=True)
        merged_orthophoto = merged_orthophoto[nearest_idx[0], nearest_idx[1]]
        if depth_images is not None:
            merged_depth = merged_depth[nearest_idx[0], nearest_idx[1]]

        # Swap axes to match the original format
        merged_orthophoto = np.swapaxes(merged_orthophoto, 0, 1)
        coor_map = np.swapaxes(coor_map, 0, 1)
        if depth_images is not None:
            merged_depth = np.swapaxes(merged_depth, 0, 1)
            return merged_orthophoto, coor_map, merged_depth

        assert merged_orthophoto.shape[:-1] == coor_map.shape[:-1]
        return merged_orthophoto, coor_map

    @staticmethod
    def world_to_pixel(world_point, coor_map):
        """
        :param world_point: (x, y, z)
        :param coor_map: orthorectify return coor_map
        :return: (i, j)
        """
        x, y, _ = world_point
        distances = np.sqrt((coor_map[..., 0] - x)**2 +
                            (coor_map[..., 1] - y)**2)
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        return i, j
