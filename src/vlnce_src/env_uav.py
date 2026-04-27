import copy
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import airsim
import fastdtw
import numpy as np
import torch
import tqdm

from src.common.param import args
from utils.logger import logger

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from airsim import MultirotorClient

from airsim_plugin.AirVLNSimulatorClientTool import (AirVLNSimulatorClientTool,
                                                     Imu, State)
from navigator import Navigator
from utils.env_utils_uav import SimState
from utils.env_vector_uav import VectorEnvUtil

RGB_FOLDER = [
    'frontcamera', 'leftcamera', 'rightcamera', 'rearcamera', 'downcamera'
]
DEPTH_FOLDER = [name + '_depth' for name in RGB_FOLDER]

from scipy.spatial.transform import Rotation as R


def project_target_state2global_state_axis(this_target_state, target_state):

    def to_eularian_angles(q):
        x, y, z, w = q
        ysqr = y * y
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        yaw = math.atan2(t3, t4)
        return (pitch, roll, yaw)

    def euler_to_rotation_matrix(e):
        rotation = R.from_euler('xyz', e, degrees=False)
        return rotation.as_matrix()

    start_pos = target_state['position']
    start_eular = to_eularian_angles(target_state['orientation'])
    this_pos = this_target_state['position']
    this_eular = to_eularian_angles(this_target_state['orientation'])
    rot = euler_to_rotation_matrix(start_eular)
    this_global_pos = np.linalg.inv(rot).T @ np.array(this_pos) + np.array(
        start_pos)
    this_global_eular = np.array(this_eular) + np.array(start_eular)
    return {
        'position': this_global_pos.tolist(),
        'orientation': this_global_eular.tolist()
    }


def prepare_object_map():
    map_config_dir = 'data/config/label_area'
    map_dict = {
        'Carla_Town01': 'Carla_Town01_test',
        'Carla_Town02': 'Carla_Town02_test',
        'Carla_Town03': 'Carla_Town03_test',
        'Carla_Town04': 'Carla_Town04_test',
        'Carla_Town05': 'Carla_Town05_test',
        'Carla_Town06': 'Carla_Town06_test',
        'Carla_Town07': 'Carla_Town07_test',
        'Carla_Town10HD': 'Carla_Town10HD_test',
        'Carla_Town15': 'Carla_Town15_test',
        'ModernCityMap': 'ModernCityMap_test',
        'NewYorkCity': 'NewYorkCity_test',
        'ModularPark': 'ModularPark_test',
        'NYCEnvironmentMegapa': 'NYC_test',
        'TropicalIsland': 'TropicalIsland_test'
    }
    for map_name, file_name in map_dict.items():
        file_name = file_name + '.json'
        map_file = os.path.join(map_config_dir, file_name)
        with open(map_file, 'r') as f:
            data = json.load(f)
        map_dict[map_name] = data['areas']
    return map_dict


def find_closest_area(coord, areas):

    def euclidean_distance(coord1, coord2):
        return np.sqrt(sum((np.array(coord1) - np.array(coord2))**2))

    min_distance = float('inf')
    closest_area = None
    closest_area_info = None
    for area in areas:
        if len(area) < 18:
            continue
        true_area = [area[0] + 1, area[1] + 1, area[2] + 0.5]
        distance = euclidean_distance(coord, true_area)
        if distance < min_distance:
            min_distance = distance
            closest_area = true_area
            closest_area_info = area
    return closest_area, closest_area_info


class AirVLNENV:

    def __init__(self,
                 batch_size=8,
                 dataset_path=None,
                 save_path=None,
                 seed=1,
                 dataset_group_by_scene=True,
                 activate_maps=[]):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.seed = seed
        self.collected_keys = set()
        self.dataset_group_by_scene = dataset_group_by_scene
        self.activate_maps = set(activate_maps)
        self.map_area_dict = prepare_object_map()
        self.exist_save_path = save_path
        load_data = self.load_my_datasets(dataset_path)
        self.ori_raw_data = load_data
        logger.info('Loaded dataset {}.'.format(len(self.dataset_path)))
        self.index_data = 0
        self.data = self.ori_raw_data

        if dataset_group_by_scene:
            self.data = self._group_scenes()
            logger.warning('dataset grouped by scene, ')

        scenes = [item['map_name'] for item in self.data]
        self.scenes = set(scenes)
        # 一个batch中每一条轨迹的当前状态记录，使用SimState类记录
        self.sim_states: Optional[List[SimState]] = [
            None for _ in range(batch_size)
        ]
        self.drone2_state = [[] for _ in range(batch_size)]
        self.last_using_map_list = []
        self.one_scene_could_use_num = 5000
        self.this_scene_used_cnt = 0
        self.init_VectorEnvUtil()

        self.is_NYC = False

    @staticmethod
    def load_drone2_traj(traj_path):
        save_path = os.path.join(traj_path, 'drone2_traj.json')
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                drone2_traj = json.load(f)
            return drone2_traj

        log_path = os.path.join(traj_path, 'log')
        point_list = os.listdir(log_path)
        point_list = sorted(point_list)
        drone2_traj = []
        for point in point_list:
            with open(os.path.join(log_path, point)) as f:
                log = json.load(f)
                drone2_traj.append(log['sensors']['state']['position'])

        with open(save_path, 'w') as f:
            json.dump(drone2_traj, f)

        return drone2_traj

    def load_my_datasets(self, path):
        # TODO: need to modify
        list_data_dict = json.load(open(path, "r"))
        trajectorys_path = set()
        skipped_trajectory_set = set()
        data = []
        old_state = random.getstate()
        for item in list_data_dict:
            trajectorys_path.add(item)
        # for item in os.listdir(self.exist_save_path):
        if not os.path.exists(os.path.join(self.exist_save_path)):
            os.makedirs(os.path.join(self.exist_save_path))
        for item in os.listdir(os.path.join(self.exist_save_path)):
            item = item.replace('success_', '').replace('oracle_', '')
            skipped_trajectory_set.add(item)

        print('Loading dataset metainfo...')
        trajectorys_path = sorted(trajectorys_path)

        invalid_cnt = 0
        for traj_path in tqdm.tqdm(trajectorys_path):
            if not os.path.exists(traj_path):
                invalid_cnt += 1
                continue
            path_parts = traj_path.strip('/').split('/')
            map_name, seq_name = path_parts[-2], path_parts[-1]
            if (len(self.activate_maps) > 0
                    and map_name not in self.activate_maps
                ) or seq_name in skipped_trajectory_set:
                continue
            # load target object info
            mark_json = os.path.join(traj_path, 'mark.json')
            if not os.path.exists(mark_json):
                invalid_cnt += 1
                continue
            with open(mark_json, 'r') as f:
                mark_json = json.load(f)
                asset_name = mark_json['object_name']
                object_position = mark_json['target']['position']
                _, closest_area_info = find_closest_area(
                    object_position, self.map_area_dict[map_name])
                object_position = [
                    closest_area_info[9], closest_area_info[10],
                    closest_area_info[11]
                ]
                obj_pose = airsim.Pose(
                    airsim.Vector3r(closest_area_info[9],
                                    closest_area_info[10],
                                    closest_area_info[11]),
                    airsim.Quaternionr(closest_area_info[13],
                                       closest_area_info[14],
                                       closest_area_info[15],
                                       closest_area_info[12]))
                obj_scale = airsim.Vector3r(closest_area_info[17],
                                            closest_area_info[17],
                                            closest_area_info[17])
                asset_name = closest_area_info[16]

            # load traj info
            traj_info = {}
            traj_info['map_name'] = map_name
            traj_info['seq_name'] = seq_name
            if not os.path.exists(os.path.join(traj_path,
                                               'gt_waypoints.json')):
                invalid_cnt += 1
                continue
            with open(os.path.join(traj_path, 'gt_waypoints.json'), 'r') as f:
                gt_waypoints = json.load(f)
            traj_info['drone1_traj'] = gt_waypoints
            traj_info['drone1_start'] = gt_waypoints[0]
            traj_info['length'] = len(gt_waypoints)

            # load drone2 traj
            drone2_traj = self.load_drone2_traj(traj_path)
            traj_info['drone2_traj'] = drone2_traj
            if len(drone2_traj) == 0:
                invalid_cnt += 1
                continue
            traj_info['drone2_start'] = drone2_traj[0]

            with open(
                    os.path.join(traj_path,
                                 'object_description_with_help.json')) as f:
                raw_instruction = json.load(f)[0]
            traj_info['instruction'] = raw_instruction
            traj_info['object'] = {
                'pose': obj_pose,
                'scale': obj_scale,
                'asset_name': asset_name
            }
            traj_info['object_position'] = object_position
            data.append(traj_info)
        print('empty_cnt:', invalid_cnt)
        random.setstate(old_state)  # Recover the state of the random generator
        return data

    def _group_scenes(self):
        assert self.dataset_group_by_scene, 'error args param'
        scene_sort_keys: OrderedDict[str, int] = {}
        for item in self.data:
            if str(item['map_name']) not in scene_sort_keys:
                scene_sort_keys[str(item['map_name'])] = len(scene_sort_keys)
        return sorted(self.data,
                      key=lambda e: (scene_sort_keys[str(e['map_name'])]))

    def init_VectorEnvUtil(self):
        self.delete_VectorEnvUtil()
        self.VectorEnvUtil = VectorEnvUtil(self.scenes, self.batch_size)

    def delete_VectorEnvUtil(self):
        if hasattr(self, 'VectorEnvUtil'):
            del self.VectorEnvUtil
        import gc
        gc.collect()

    def next_minibatch(self, skip_scenes=[], data_it=0):
        batch = []
        while True:
            if self.index_data >= len(self.data):
                # 结束一轮dagger时重洗数据并设置场景
                random.shuffle(self.data)
                logger.warning('random shuffle data')
                if self.dataset_group_by_scene:
                    self.data = self._group_scenes()
                    logger.warning('dataset grouped by scene')

                # 结束条件
                if len(batch) == 0:
                    self.index_data = 0
                    self.batch = None
                    return

                # 最后一个batch，补全整个batch
                self.index_data = self.batch_size - len(batch)
                batch += self.data[:self.index_data]
                break

            new_trajectory = self.data[self.index_data]

            if "NewYorkCity" in new_trajectory['map_name']:
                self.is_NYC = True
            else:
                self.is_NYC = False

            if new_trajectory['map_name'] in skip_scenes:
                self.index_data += 1
                continue

            batch.append(new_trajectory)
            self.index_data += 1

            if len(batch) == self.batch_size:
                break

        self.batch = copy.deepcopy(batch)
        assert len(self.batch) == self.batch_size, 'next_minibatch error'
        self.VectorEnvUtil.set_batch(self.batch)
        return self.batch
        # return [b['trajectory_dir'] for b in self.batch]

    #
    def changeToNewTrajectorys(self):
        self._changeEnv(need_change=False)

        self._setTrajectorys()

        self._setObjects()

        self.update_measurements()

    def _setObjects(self, ):
        objects_info = [item['object'] for item in self.batch]
        return self.simulator_tool.setObjects(objects_info)

    def _changeEnv(self, need_change: bool = True):
        using_map_list = [item['map_name'] for item in self.batch]

        assert len(using_map_list) == self.batch_size, '错误'

        machines_info_template = copy.deepcopy(args.machines_info)
        total_max_scene_num = 0
        for item in machines_info_template:
            total_max_scene_num += item['MAX_SCENE_NUM']
        assert self.batch_size <= total_max_scene_num, 'error args param: batch_size'

        machines_info = []
        ix = 0
        for index, item in enumerate(machines_info_template):
            machines_info.append(item)
            delta = min(self.batch_size, item['MAX_SCENE_NUM'],
                        len(using_map_list) - ix)
            machines_info[index]['open_scenes'] = using_map_list[ix:ix + delta]
            # TODO: gpus !!!
            machines_info[index]['gpus'] = [args.gpu_id] * 8
            ix += delta

        cnt = 0
        for item in machines_info:
            cnt += len(item['open_scenes'])
        assert self.batch_size == cnt, 'error create machines_info'

        #
        if self.this_scene_used_cnt < self.one_scene_could_use_num and \
                len(set(using_map_list)) == 1 and len(set(self.last_using_map_list)) == 1 and \
                using_map_list[0] is not None and self.last_using_map_list[0] is not None and using_map_list[0] == self.last_using_map_list[0] and \
                need_change == False:
            self.this_scene_used_cnt += 1
            logger.warning('no need to change env: {}'.format(using_map_list))
            return
        else:
            logger.warning('to change env: {}'.format(using_map_list))

        #
        self.machines_info = copy.deepcopy(machines_info)
        print('machines_info:', self.machines_info)
        self.simulator_tool = AirVLNSimulatorClientTool(
            machines_info=self.machines_info)
        self.simulator_tool.run_call()

        self.last_using_map_list = using_map_list.copy()
        self.this_scene_used_cnt = 1

    def _setTrajectorys(self):
        # TODO: also set pose for drone2
        drone1_start_position_list = [
            item['drone1_start'] for item in self.batch
        ]
        drone2_start_position_list = [
            item['drone2_start'] for item in self.batch
        ]

        # setpose
        poses = []
        poses2 = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            poses2.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                pose = airsim.Pose(position_val=airsim.Vector3r(
                    x_val=drone1_start_position_list[cnt][0],
                    y_val=drone1_start_position_list[cnt][1],
                    z_val=max(-10, drone1_start_position_list[cnt][2] - 5),
                ))
                pose2 = airsim.Pose(position_val=airsim.Vector3r(
                    x_val=drone2_start_position_list[cnt][0],
                    y_val=drone2_start_position_list[cnt][1],
                    z_val=-150 if self.is_NYC else -80,
                ))
                poses[index_1].append(pose)
                poses2[index_2].append(pose2)
                cnt += 1
        for _ in range(3):
            results = self.simulator_tool.setPoses(poses=poses,
                                                   vehicle_name='Drone_1')
            results2 = self.simulator_tool.setPoses(poses=poses2,
                                                    vehicle_name='Drone_2')

        # TODO: why this set pose func can make the drone persist at this pose
        state_info_results = self.simulator_tool.getSensorInfo()

        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2, _ in enumerate(item['open_scenes']):
                self.sim_states[cnt] = SimState(
                    index=cnt, step=0, raw_trajectory_info=self.batch[cnt])
                self.sim_states[cnt].trajectory = [
                    state_info_results[index_1][index_2]
                ]
                self.sim_states[cnt].drone2_traj = [
                    state_info_results[index_1][index_2]['sensors']['state2']
                ]
                cnt += 1

    def get_obs(self):
        time_start = time.time()
        obs_states = self._getStates()
        print('get_states time:', time.time() - time_start)
        obs, states = self.VectorEnvUtil.get_obs(obs_states)
        self.sim_states = states
        return obs

    def _getStates(self):
        # logger.info('getting states from server...')
        responses = self.simulator_tool.getImageResponses()
        # responses_for_record = self.simulator_tool.getImageResponsesForRecord()
        # logger.info('received states from server...')
        cnt = 0
        for item in responses:
            cnt += len(item)
        assert len(responses) == len(self.machines_info), 'error'
        assert cnt == self.batch_size, 'error'

        states = [None for _ in range(self.batch_size)]
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2 in range(len(item['open_scenes'])):
                rgb_images = responses[index_1][index_2][0]
                depth_images = responses[index_1][index_2][1]
                bev_images = responses[index_1][index_2][2]
                bev_depth = responses[index_1][index_2][3]
                state = self.sim_states[cnt]
                # import ipdb;ipdb.set_trace()
                states[cnt] = (rgb_images, depth_images, state, bev_images,
                               bev_depth)
                cnt += 1
        return states

    def _get_current_state(self) -> list:
        states = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            states.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                s = self.sim_states[cnt].state
                state = airsim.KinematicsState()
                state.position = airsim.Vector3r(*s['position'])
                state.orientation = airsim.Quaternionr(*s['orientation'])
                state.linear_velocity = airsim.Vector3r(*s['linear_velocity'])
                state.angular_velocity = airsim.Vector3r(
                    *s['angular_velocity'])
                states[index_1].append(state)
                # states[index_1].append(
                #     airsim.KinematicsState(position=airsim.Vector3r(*s['position']),
                #                            orientation=airsim.Quaternionr(*s['orientation']),
                #                            linear_velocity=airsim.Vector3r(*s['linear_velocity']),
                #                            angular_velocity=airsim.Vector3r(*s['angular_velocity']))
                # )
                cnt += 1
        return states

    def _get_current_pose(self) -> list:
        poses = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                poses[index_1].append(self.sim_states[cnt].pose)
                cnt += 1
        return poses

    def reset(self):
        self.changeToNewTrajectorys()
        return self.get_obs()

    def revert2frame(self, index):
        self.sim_states[index].revert2frames()

    def makeActions(self, waypoint1, waypoint2, navigator: Navigator,
                    traj_status):
        waypoints_args = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            waypoints_args.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                waypoints_args[index_1].append(waypoint1[cnt])
                cnt += 1
        # start_states = self._get_current_state()
        # TODO: deal with the waypoint by batches
        temp_waypoint1 = torch.tensor(waypoint1[0],
                                      device=navigator.device,
                                      dtype=torch.float32)
        temp_waypoint2 = torch.tensor(waypoint2[0], device=navigator.device)
        results, delta_time, delta_distance, end_pos = navigator.navigate(
            temp_waypoint1, temp_waypoint2, traj_status)
        if results is None:
            raise Exception('move on path error.')
        states = results['states']
        drone2_states = results['drone2_states']
        collision = results['collision'][0]
        drone2_collision = results['collision'][1]
        airsim_collision = results['airsim_collision']
        for index, waypoint in enumerate(waypoint1):
            if np.linalg.norm(
                    np.array(waypoint) -
                    np.array(self.batch[index]['object_position'])
            ) < self.sim_states[index].SUCCESS_DISTANCE:
                self.sim_states[index].oracle_success = True
            elif self.sim_states[index].step >= int(args.maxWaypoints):
                self.sim_states[index].is_end = True

            if self.sim_states[index].is_end == True:
                waypoint = [self.sim_states[index].pose[0:3]] * len(waypoint)
            self.sim_states[index].step += 1
            self.sim_states[index].trajectory.extend(states)  # [xyzxyzw]...
            self.sim_states[index].pre_waypoints = waypoint
            self.sim_states[index].is_collisioned = collision
            self.sim_states[index].drone2_traj.append(drone2_states)

        self.update_measurements()
        return collision, drone2_collision, delta_time, delta_distance, end_pos, airsim_collision

    def Teleport(self, waypoint1, waypoint2):
        airsim_client: MultirotorClient = self.simulator_tool.airsim_clients[
            0][0]
        state_sensor = State(airsim_client, drone_name='Drone_1')
        imu_sensor = Imu(airsim_client, imu_name='Imu', drone_name='Drone_1')
        drone2_state_sensor = State(airsim_client, drone_name='Drone_2')
        results = []
        delta_distance = None
        target_yaw = None

        # calculate the orientation according to the target direction
        if len(waypoint1) > 0:
            start_pos = airsim_client.getMultirotorState(
                vehicle_name="Drone_1").kinematics_estimated.position
            start_pos = np.array([p_i for p_i in start_pos])
            target_pos = np.array(waypoint1[0])
            target_direction = target_pos - start_pos
            delta_distance = np.linalg.norm(target_pos - start_pos)
            target_yaw = math.atan2(target_direction[1], target_direction[0])

            target_pose = airsim.Pose(
                airsim.Vector3r(*target_pos),
                airsim.to_quaternion(0, 0, math.radians(target_yaw)))
            airsim_client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
            airsim_client.simSetVehiclePose(target_pose,
                                            True,
                                            vehicle_name="Drone_1")

        # set Drone2_pos
        drone2_pos = np.array(waypoint2[0]).squeeze()
        drone2_pose = airsim.Pose(airsim.Vector3r(*drone2_pos))
        airsim_client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        airsim_client.simSetVehiclePose(drone2_pose,
                                        True,
                                        vehicle_name="Drone_2")

        # update state and imu
        state_info = copy.deepcopy(state_sensor.retrieve())
        state_info2 = copy.deepcopy(drone2_state_sensor.retrieve())
        imu_info = copy.deepcopy(imu_sensor.retrieve())
        results.append({'sensors': {'state': state_info, 'imu': imu_info}})

        # update sim_states
        results = {'states': results, 'drone2_states': state_info2}
        states = results['states']
        drone2_states = results['drone2_states']
        for index, waypoint in enumerate(waypoint1):
            if np.linalg.norm(
                    np.array(waypoint) -
                    np.array(self.batch[index]['object_position'])
            ) < self.sim_states[index].SUCCESS_DISTANCE:
                self.sim_states[index].oracle_success = True
            elif self.sim_states[index].step >= int(args.maxWaypoints):
                self.sim_states[index].is_end = True

            if self.sim_states[index].is_end == True:
                waypoint = [self.sim_states[index].pose[0:3]] * len(waypoint)
            self.sim_states[index].step += 1
            self.sim_states[index].trajectory.extend(states)  # [xyzxyzw]...
            self.sim_states[index].pre_waypoints = waypoint
            self.sim_states[index].drone2_traj.append(drone2_states)

        self.update_measurements()
        return delta_distance, target_yaw

    def update_measurements(self):
        self._update_distance_to_target()
        # self._update_DistanceToGoal()
        # self._updata_Success()
        # self._updata_NDTW()
        # self._updata_SDTW()
        # self._update_PathLength()
        # self._update_OracleSuccess()
        # self._update_StepsTaken()

    def _update_distance_to_target(self):
        target_positions = [item['object_position'] for item in self.batch]
        for idx, target_position in enumerate(target_positions):
            current_position = self.sim_states[idx].pose[0:3]
            current_drone2_position = self.sim_states[idx].drone2_traj[-1][
                'position']
            distance = np.linalg.norm(
                np.array(current_position) - np.array(target_position))
            print(
                f'batch[{idx}/{len(self.batch)}]| distance: {round(distance, 2)}\n position: {current_position[0]:.03f}, {current_position[1]:.03f}, {current_position[2]:.03f}\n position_drone2: {current_drone2_position[0]:.03f}, {current_drone2_position[1]:.03f}, {current_drone2_position[2]:.03f}\n target: {target_position[0]:.03f}, {target_position[1]:.03f}, {target_position[2]:.03f}'
            )

    def _updata_Success(self):
        for i, state in enumerate(self.sim_states):
            distance_to_target = self.sim_states[i].DistanceToGoal['_metric']
            if (self.sim_states[i].is_end and distance_to_target
                    <= self.sim_states[i].SUCCESS_DISTANCE):
                self.sim_states[i].Success['_metric'] = 1.0
            else:
                self.sim_states[i].Success['_metric'] = 0.0

    def _updata_NDTW(self):

        def euclidean_distance(
            position_a,
            position_b,
        ) -> float:
            return np.linalg.norm(np.array(position_b) - np.array(position_a),
                                  ord=2)

        for i, state in enumerate(self.sim_states):

            current_position = np.array(
                [state.pose[0], state.pose[1], state.pose[2]])

            if len(state.NDTW['locations']) == 0:
                self.sim_states[i].NDTW['locations'].append(current_position)
            else:
                if current_position.tolist(
                ) == state.NDTW['locations'][-1].tolist():
                    continue
                self.sim_states[i].NDTW['locations'].append(current_position)

            dtw_distance = fastdtw.fastdtw(
                self.sim_states[i].NDTW['locations'],
                self.sim_states[i].NDTW['gt_locations'],
                dist=euclidean_distance)[0]

            nDTW = np.exp(-dtw_distance /
                          (len(self.sim_states[i].NDTW['gt_locations']) *
                           self.sim_states[i].SUCCESS_DISTANCE))
            self.sim_states[i].NDTW['_metric'] = nDTW

    def _updata_SDTW(self):
        for i, state in enumerate(self.sim_states):
            ep_success = self.sim_states[i].Success['_metric']
            nDTW = self.sim_states[i].NDTW['_metric']
            self.sim_states[i].SDTW['_metric'] = ep_success * nDTW

    def _update_OracleSuccess(self):
        for i, state in enumerate(self.sim_states):
            d = self.sim_states[i].DistanceToGoal['_metric']
            self.sim_states[i].OracleSuccess['_metric'] = float(
                self.sim_states[i].OracleSuccess['_metric']
                or d <= self.sim_states[i].SUCCESS_DISTANCE)

    def _update_StepsTaken(self):
        for i, state in enumerate(self.sim_states):
            self.sim_states[i].StepsTaken['_metric'] = self.sim_states[i].step
