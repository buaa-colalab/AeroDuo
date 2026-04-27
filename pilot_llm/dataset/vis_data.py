import cv2
import numpy as np


def world_to_pixel(world_point, coor_map):
    x, y, _ = world_point
    distances = np.sqrt((coor_map[..., 0] - x)**2 + (coor_map[..., 1] - y)**2)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    return i, j


def visualize_waypoints(waypoints,
                        coor_map,
                        merged_ortho,
                        color=[0, 255, 0],
                        radius=8):
    for idx, point in enumerate(waypoints):
        i, j = world_to_pixel(point, coor_map)
        cv2.circle(
            merged_ortho,
            center=(j, i),  # OpenCV 的坐标是 (x, y) = (width, height)
            radius=radius,
            color=color,
            thickness=-1)
    return merged_ortho
