import math
from queue import PriorityQueue
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


class A_star:

    def __init__(self,
                 heuristic_type: str = 'manhattan',
                 grid_distance: int = 1):
        if heuristic_type == 'euclidean':
            self.heuristic = self.euclidean_heuristic
        elif heuristic_type == 'manhattan':
            self.heuristic = self.manhattan_heuristic
        else:
            raise NotImplementedError

        self.init_neighbors_with_cost(grid_distance)

    def init_neighbors_with_cost(self, grid_distance: int):
        self.relative_neighbors_with_cost = [
            (grid_distance, 0, grid_distance),
            (-grid_distance, 0, grid_distance),
            (0, grid_distance, grid_distance),
            (0, -grid_distance, grid_distance)
        ]

    def visualize(self, searched_path: List, global_ortho, save_path):
        mark_color = [0, 255, 0]  # green
        for point in searched_path:
            half_size = 1
            i, j = point
            for x in range(i - half_size, i + half_size + 1):
                for y in range(j - half_size, j + half_size + 1):
                    global_ortho[x, y] = mark_color
        cv2.imwrite(save_path, global_ortho)

    @staticmethod
    def euclidean_heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    @staticmethod
    def manhattan_heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def from_pixel_to_3d(searched_path, uav_height, coor_map):
        searched_path_in_3d = []
        for pixel in searched_path:
            x, y = coor_map[pixel]
            searched_path_in_3d.append([x, y, uav_height])

        return searched_path_in_3d

    def fetch_neighbors(self, current: Tuple[int, int]):
        cost_to_neighbors = {}
        for dx, dy, cost in self.relative_neighbors_with_cost:
            new_position = (current[0] + dx, current[1] + dy)
            if new_position[0] < 0 or new_position[
                    0] >= self.occupancy_map.shape[0]:
                continue
            if new_position[1] < 0 or new_position[
                    1] >= self.occupancy_map.shape[1]:
                continue
            # skip the obstacle point
            if self.occupancy_map[new_position[0], new_position[1]] == 0:
                continue
            cost_to_neighbors[new_position] = cost

        return cost_to_neighbors

    @staticmethod
    def world_to_pixel(world_point, coor_map):
        """
        将世界坐标 (x, y, z) 转换为正射图像素坐标 (i, j)
        :param world_point: 世界坐标 (x, y, z)
        :param coor_map: orthorectify 返回的 coor_map
        :return: (i, j) 像素坐标
        """
        x, y, _ = world_point
        # 计算每个像素与目标点的欧氏距离（仅x,y）
        distances = np.sqrt((coor_map[..., 0] - x)**2 +
                            (coor_map[..., 1] - y)**2)
        # 找到最近像素的索引
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        return i, j

    def find_nearest_valid_goal(self, goal):
        # 如果目标点本身可行，直接返回
        if self.occupancy_map[goal] == 1:
            return goal

        # 从小到大按半径搜索
        goal_i, goal_j = goal
        h, w = self.occupancy_map.shape

        # 使用扩展环搜索而不是整个区域搜索，提高效率
        for radius in range(1, 1000):
            # 候选点列表
            candidates = []

            # 搜索当前半径的圆环上的点
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    # 只考虑圆环上的点(近似)
                    if abs(di) + abs(dj) == radius:
                        i, j = goal_i + di, goal_j + dj

                        # 检查边界
                        if 0 <= i < h and 0 <= j < w:
                            # 如果点可行，添加为候选
                            if self.occupancy_map[i, j] == 1:
                                candidates.append((i, j))

            # 如果找到候选点，返回距离最近的一个
            if candidates:
                # 计算每个候选点到目标的欧几里得距离
                distances = [((i - goal_i)**2 + (j - goal_j)**2)
                             for i, j in candidates]
                # 返回距离最小的点

                return candidates[np.argmin(distances)]

    def search(self, global_depth, global_coor_map, start_world, goal_world,
               delta_altitude):
        # prepare the start, end and the obstacle map
        start = self.world_to_pixel(start_world, global_coor_map)
        goal = self.world_to_pixel(goal_world, global_coor_map)
        start_depth = min(delta_altitude, int(global_depth[start]))
        raw_occupancy_map = (global_depth >= start_depth
                             )  # True for free space, False for obstacle

        # inflate the obstacle
        # pixel_size = 0.25m, we assume that drone_size = 1m
        dilate_kernel = np.ones((3, 3), np.uint8)
        self.occupancy_map = cv2.erode(raw_occupancy_map.astype(np.uint8),
                                       dilate_kernel,
                                       iterations=1)
        if self.occupancy_map[start] == 0:
            self.occupancy_map = np.squeeze(raw_occupancy_map.astype(np.uint8))

        goal = self.find_nearest_valid_goal(goal)

        # prepare the state dict for A* search
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = dict()
        came_from[start] = None
        cost_so_far = dict()
        cost_so_far[start] = 0

        while not frontier.empty():
            current = frontier.get()

            if current == goal:
                break

            for next, cost in self.fetch_neighbors(current).items():
                new_cost = cost_so_far[current] + cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current

        if goal not in came_from.keys():
            # 返回came_from中与goal最接近的位置
            closest_point = None
            min_distance = float('inf')

            for point in came_from.keys():
                dist = self.euclidean_heuristic(point, goal)
                if dist < min_distance:
                    min_distance = dist
                    closest_point = point

            if closest_point is not None:
                print(
                    f"Using closest reachable point instead: {closest_point}, distance to goal: {min_distance}"
                )
                goal = closest_point

        # extract the searched path
        searched_path = [goal]
        current = came_from[goal]
        while current is not None:
            searched_path.append(current)
            current = came_from[current]
        searched_path.reverse()

        # transform the searched path to 3d
        # searched_path_3d = self.from_pixel_to_3d(searched_path, start_world[-1], global_coor_map)

        return searched_path

    def jump_point_search(self, global_depth, global_coor_map, start_world,
                          goal_world):
        pass


if __name__ == "__main__":
    import json
    import os

    from PIL import Image

    test_traj_path = "/home/wrp/airdrone_proj/AirVLN/A_star/test_case/1d7962f1-f502-4014-9351-9fc0893eb079"
    depth_path = os.path.join(test_traj_path, 'bevcamera_depth/000010.png')
    depth = np.asarray(Image.open(depth_path).convert('L'))
    a_star = A_star()

    with open(os.path.join(test_traj_path, 'gt_waypoints.json')) as f:
        gt_waypoints = json.load(f)
    print("traj len:", len(gt_waypoints))

    with open(os.path.join(test_traj_path, 'log/000010.json')) as f:
        bev_camera_pos = json.load(f)['sensors']['state']['position']
    start = gt_waypoints[20]
    goal = gt_waypoints[40]
    start[2] = goal[2] = -10
    start = tuple(start)
    goal = tuple(goal)
    print("bev_camera_pos:", bev_camera_pos)
    print("start:", start)
    print("goal:", goal)

    occupancy_mask = np.zeros_like(depth)
    occupancy_mask[depth >= 70] = 1
    a_star.init_task_info(occupancy_mask, start, goal, bev_camera_pos)

    searched_path = a_star.search()
    print("searched path:\n", searched_path)
    a_star.visualize(searched_path)
    refined_path = a_star.from_grid_to_3d(searched_path, 70, bev_camera_pos)
    print("3d path:\n", refined_path)
