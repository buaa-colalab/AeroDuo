import json
import math
import os
import re
import time
import traceback
import warnings

import cv2
import msgpackrpc
import numpy as np
import torch
import tqdm

from navigator import Navigator
from pilot_llm.api_for_airsim import CallPilot
from src.common.param import args, data_args, model_args
from src.vlnce_src.dino_monitor_online import DinoMonitor
from src.vlnce_src.env_uav import AirVLNENV
from utils.logger import logger
from utils.utils import is_dist_avail_and_initialized

warnings.filterwarnings("ignore")


class BatchIterator:

    def __init__(self, env: AirVLNENV):
        self.env = env

    def __len__(self):
        return len(self.env.data)

    def __next__(self):
        # import ipdb;ipdb.set_trace()
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch

    def __iter__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch


class Metrics:

    def __init__(self, log_path: str = None):
        self.log_path = log_path
        # ultimate metrics
        self.OSR = 0
        self.SR = 0
        self.CR = 0
        self.airsim_CR = 0
        self.SPL = 0
        self.SST = 0
        self.TL = 0  # average trajectory length
        self.NE = 0

        # intermediate statistics
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                data = json.load(f)
            self.total_cnt = data["total_cnt"]
            self.success_cnt = data["success_cnt"]
            self.oracle_success_cnt = data["oracle_success_cnt"]
            self.collision_cnt = data["collision_cnt"]
            self.airsim_collision_cnt = data["airsim_collision_cnt"]
            self.spl_cnt = data["spl_cnt"]
            self.sst_cnt = data["sst_cnt"]
            self.total_length = data["total_length"]
            self.total_error = data["total_error"]
            # Navigator extra statistics
            self.nav_time_stat = data["nav_time_stat"]
            self.nav_velocity_stat = data["nav_velocity_stat"]
            self.nav_stat_cnt = data["nav_stat_cnt"]
        else:
            self.total_cnt = 0
            self.success_cnt = 0
            self.oracle_success_cnt = 0
            self.collision_cnt = 0
            self.airsim_collision_cnt = 0
            self.spl_cnt = 0
            self.sst_cnt = 0
            self.total_length = 0
            self.total_error = 0
            # Navigator extra statistics
            self.nav_time_stat = 0
            self.nav_velocity_stat = 0
            self.nav_stat_cnt = 0

    def update_ultimate_metrics(self):
        self.OSR = self.oracle_success_cnt / self.total_cnt
        self.SR = self.success_cnt / self.total_cnt
        self.CR = self.collision_cnt / self.total_cnt
        self.airsim_CR = self.airsim_collision_cnt / self.total_cnt
        self.SPL = self.spl_cnt / self.total_cnt
        self.SST = self.sst_cnt / self.total_cnt
        self.TL = self.total_length / self.total_cnt
        self.NE = self.total_error / self.total_cnt
        print("OSR: ", self.OSR, " SR: ", self.SR, " CR: ", self.CR,
              "airsim_CR: ", self.airsim_CR, " SPL: ", self.SPL, " SST: ",
              self.SST, "TL:", self.TL, " NE: ", self.NE)

        # save intermeidate statistics
        with open(self.log_path, 'w') as f:
            json.dump(
                {
                    "total_cnt": self.total_cnt,
                    "success_cnt": self.success_cnt,
                    "oracle_success_cnt": self.oracle_success_cnt,
                    "collision_cnt": self.collision_cnt,
                    "airsim_collision_cnt": self.airsim_collision_cnt,
                    "spl_cnt": self.spl_cnt,
                    "sst_cnt": self.sst_cnt,
                    "total_length": self.total_length,
                    "total_error": self.total_error,
                    "nav_time_stat": self.nav_time_stat,
                    "nav_velocity_stat": self.nav_velocity_stat,
                    "nav_stat_cnt": self.nav_stat_cnt
                }, f)


class TrajectoryStatus:

    def __init__(self, env_batches, train_env, object_desc_dict, args):
        self.ori_data_dirs = [b['seq_name'] for b in env_batches]
        self.map_names = [b['map_name'] for b in env_batches]
        self.target_positions = [b['object_position'] for b in env_batches]
        self.object_infos = [
            object_desc_dict.get(
                b['object']['asset_name'].replace("AA", ""),
                re.sub(r'(SM_|AASM_)?\d*([a-zA-Z]+)\d*', r'\2',
                       b['object']['asset_name'])) for b in env_batches
        ]
        self.gt_drone1_trajs = [b['drone1_traj'] for b in env_batches]
        self.gt_drone2_trajs = [b['drone2_traj'] for b in env_batches]
        self.drone1_trajs = [[] for _ in range(train_env.batch_size)]
        self.drone2_trajs = [[] for _ in range(train_env.batch_size)]
        self.raw_instructions = [b['instruction'] for b in env_batches]

        self.train_env = train_env
        self.episodes = [[] for _ in range(train_env.batch_size)]
        self.skips = [False for _ in range(train_env.batch_size)]
        self.dones = [False for _ in range(train_env.batch_size)]
        self.collisions = [False for _ in range(train_env.batch_size)]
        self.distance_to_ends = [[] for _ in range(train_env.batch_size)]
        self.dino_results = [False for _ in range(train_env.batch_size)]
        self.success = [False for _ in range(train_env.batch_size)]
        self.oracle_success = [False for _ in range(train_env.batch_size)]
        self.early_end = [False for _ in range(train_env.batch_size)]
        self.envs_to_pause = []

        self.maxWaypoints = args.maxWaypoints  # 50
        self.is_end = False
        self.eval_save_dir = args.eval_save_path

        self.pointclouds = [[] for _ in range(train_env.batch_size)]
        self.bevs = [[] for _ in range(train_env.batch_size)]
        self.bev_depths = [[] for _ in range(train_env.batch_size)]

        # traj statistics
        self.steps = 0
        self.distance = 0
        self.time = 0
        self.airsim_collision = False

    def calculate_traj_stats(self, metrics: Metrics):
        # calculate optimal distance
        tot_distance = 0
        traj_waypoint = self.gt_drone1_trajs[0]
        for i in range(len(traj_waypoint) - 1):
            p1 = np.array(traj_waypoint[i])
            p2 = np.array(traj_waypoint[i + 1])
            tot_distance += np.linalg.norm(p2 - p1)
        tot_distance -= 20

        # retrieve optimal time
        root_path = "data/HaL-13k"
        time_json_path = os.path.join(root_path, self.map_names[0],
                                      self.ori_data_dirs[0], "time.json")
        with open(time_json_path, 'r') as f:
            tot_time = json.load(f)["gt_time"]

        return tot_distance, tot_time

    def init_dino_monitor(self, monitor: DinoMonitor):
        self.monitor = monitor

    @staticmethod
    def project_image_to_3d(depth_image, bev_camera_pos):
        """ 将深度图投影到 3D 世界坐标系 """
        h, w = depth_image.shape
        c_x, c_y = w / 2, h / 2
        FOV = 120
        f_x = w / (2 * math.tan(math.radians(FOV / 2)))
        f_y = h / (2 * math.tan(math.radians(FOV / 2)))

        # 预计算内参逆矩阵
        intrinsic_inv = np.linalg.inv(
            np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]))

        # 旋转矩阵逆矩阵 (从相机坐标系转换到世界坐标系)
        r_inv = np.linalg.inv(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))

        # 生成图像像素坐标 (u, v)
        i, j = np.indices(depth_image.shape)
        z = depth_image.astype(np.float32)
        bev_camera_pos = np.array(bev_camera_pos).reshape(3, 1)  # (3, 1)

        # 计算 3D 坐标
        pixels_homogeneous = np.stack((j * z, i * z, z),
                                      axis=-1)  # (1024, 1024, 3)
        pixels_homogeneous = np.expand_dims(pixels_homogeneous,
                                            -1)  # (1024, 1024, 3, 1)
        coor_3d = bev_camera_pos + r_inv @ intrinsic_inv @ pixels_homogeneous  # r_inv @ intrinsic_inv @ pixels_homogeneous shape: (1024, 1024, 3, 1)

        return coor_3d.squeeze(-1)

    def update_pointclouds(self, bev_camera_pos):
        bev_depth = self.fetch_from_observations("bev_depth")
        for i in range(self.train_env.batch_size):
            self.pointclouds[i].append(
                self.project_image_to_3d(bev_depth[i], bev_camera_pos[i]))
        return self.pointclouds

    def update_bevs(self):
        bev = self.fetch_from_observations("bev")
        bev_depth = self.fetch_from_observations("bev_depth")
        for i in range(self.train_env.batch_size):
            self.bevs[i].append(bev[i])
            self.bev_depths[i].append(bev_depth[i])

    def update_observation(self, outputs, pos_list, airsim_collision):
        observations, dones, collisions, oracle_success = [
            list(x) for x in zip(*outputs)
        ]
        self.observations = observations
        self.update_bevs()

        # TODO: we assume batch=1 here to simplify the code
        self.airsim_collision = airsim_collision

        for i in range(self.train_env.batch_size):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            self.dones[i] = dones[i]
            self.collisions[i] = collisions[i]
            self.oracle_success[i] = oracle_success[i]
            self.drone1_trajs[i].extend(pos_list)
            if len(pos_list) > 0:
                new_distance_to_ends = [
                    np.linalg.norm(
                        np.array(pos) - np.array(self.target_positions[i]))
                    for pos in pos_list
                ]
                for distance in new_distance_to_ends:
                    if distance <= 20:
                        self.oracle_success[i] = True
                        break
                self.distance_to_ends[i].extend(new_distance_to_ends)
            if len(self.drone1_trajs[i]) == 0:
                self.drone1_trajs[i].append(
                    observations[i][-1]['sensors']['state']['position'])
            if len(self.drone2_trajs[i]) > 0:
                last_point = self.drone2_trajs[i][-1]
                now_point = self.train_env.sim_states[i].drone2_traj[-1][
                    'position']
                direction = np.array(now_point) - np.array(last_point)
                distance = np.linalg.norm(direction)
                unit_direction = direction / distance if distance > 0 else np.zeros_like(
                    direction)
                # new point every 5 meters
                new_point = last_point
                while distance > 5:
                    new_point = np.array(new_point) + unit_direction * 5
                    self.drone2_trajs[i].append(new_point.tolist())
                    distance -= 5
            else:
                self.drone2_trajs[i].append(
                    self.train_env.sim_states[i].drone2_traj[-1]['position'])

            if self.oracle_success[i]:
                self.check_deviation(i)

    def save_to_dataset(self, root_path, i):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        # TODO: save bev, bev_depth, drone1_traj, drone2_traj and link mark.json and object_description_with_help.json
        folder_names = ['bevcamera', 'log', 'log2']
        for folder_name in folder_names:
            os.makedirs(os.path.join(root_path, folder_name), exist_ok=True)

        # TODO: finish and reactivate
        # self.save_images(root_path, i)
        self.save_logs(root_path, i)

    def save_images(self, trajectory_dir, i):
        episodes = self.episodes[i]
        for idx, episode in enumerate(episodes):
            # if 'rgb' in episode:
            #     for cid, camera_name in enumerate(['frontcamera']):
            #         image = episode['rgb'][cid]
            #         cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)
            # if 'depth' in episode:
            #     for cid, camera_name in enumerate(['frontcamera_depth']):
            #         image = episode['depth'][cid]
            #         cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)
            if 'bev' in episode:
                image = episode['bev']
                cv2.imwrite(
                    os.path.join(trajectory_dir, 'bevcamera',
                                 str(idx).zfill(6) + '.png'), image)
            # if 'bev_depth' in episode:
            #     image = episode['bev_depth']
            #     cv2.imwrite(os.path.join(trajectory_dir, 'bevcamera_depth', str(idx).zfill(6) + '.png'), image)

    def save_logs(self, trajectory_dir, i):
        # TODO: save dagger info
        drone1_traj = self.drone1_trajs[i]
        save_dir = os.path.join(trajectory_dir, 'log')
        for idx, point in enumerate(drone1_traj):
            with open(os.path.join(save_dir,
                                   str(idx).zfill(6) + '.json'), 'w') as f:
                json.dump(point, f)

        # TODO: save drone2 traj
        drone2_traj = self.drone2_trajs[i]
        drone2_save_dir = os.path.join(trajectory_dir, 'log2')
        for idx, point in enumerate(drone2_traj):
            with open(
                    os.path.join(drone2_save_dir,
                                 str(idx).zfill(6) + '.json'), 'w') as f:
                json.dump(point, f)

    def check_traj_status(self, metrics):
        for i in range(self.train_env.batch_size):
            if len(self.drone1_trajs[0]) - 1 > self.maxWaypoints:
                self.dones[i] = True
            if len(self.drone1_trajs[i]) > 1:
                delta_distance = np.linalg.norm(
                    np.array(self.drone1_trajs[i][-1]) -
                    np.array(self.drone1_trajs[i][-2]))
                if delta_distance < 0.1:
                    self.collisions[i] = True
            # 检验每一条轨迹有没有结束
            if self.collisions[i]:
                self.dones[i] = True
            if self.early_end[i] and self.oracle_success[i]:
                self.dones[i] = True
            if self.dones[i] and not self.skips[i]:
                prex = ""
                self.envs_to_pause.append(i)
                if self.success[i]:
                    prex = 'success_'
                    print(i, " has succeed!")
                elif self.oracle_success[i]:
                    prex = "oracle_"
                    print(i, " has oracle succeed!")
                new_traj_name = prex + self.ori_data_dirs[
                    i]  # str(uuid.uuid4())
                new_traj_dir = os.path.join(self.eval_save_dir, new_traj_name)
                self.save_to_dataset(new_traj_dir, i)
                self.skips[i] = True
                print(i, " has finished!")
            if np.array(self.skips).all():
                self.is_end = True

        if self.dones[0]:
            self.update_metrics(metrics, new_traj_dir)
            return True
        return False

    def calculate_spl(self, gt_distance):
        spl = gt_distance / max(gt_distance, self.distance)
        assert spl > 0, "SPL should be greater than 0"

        return spl

    def calculate_sst(self, gt_time):
        sst = gt_time / max(gt_time, self.time)
        assert sst > 0, "SST should be greater than 0"

        return sst

    def update_metrics(self, metrics: Metrics, new_traj_dir):
        # we temporarily take bs=1
        metrics.total_cnt += 1
        spl_cnt = 0
        sst_cnt = 0

        metrics.total_length += self.distance
        metrics.total_error += self.distance_to_ends[0][-1]

        if self.success[0]:
            metrics.success_cnt += 1
            metrics.oracle_success_cnt += 1
            gt_distance, gt_time = self.calculate_traj_stats(metrics)
            spl_cnt = self.calculate_spl(gt_distance)
            sst_cnt = self.calculate_sst(gt_time)
            metrics.spl_cnt += spl_cnt
            metrics.sst_cnt += sst_cnt
        elif self.oracle_success[0]:
            metrics.oracle_success_cnt += 1
        if self.collisions[0] and not self.success[0]:
            metrics.collision_cnt += 1
        if self.airsim_collision:
            metrics.airsim_collision_cnt += 1

        state_log = {
            "success": self.success[0],
            "oracle_success": self.oracle_success[0],
            "collision": self.collisions[0],
            "airsim_collision": self.airsim_collision,
            "spl": spl_cnt,
            "sst": sst_cnt,
            "steps": len(self.drone1_trajs[0]) - 1,
            "path_length": self.distance,
            "time": self.time,
            "distance_to_end": self.distance_to_ends[0][-1],
        }
        with open(os.path.join(new_traj_dir, "state_log.json"), 'w') as f:
            json.dump(state_log, f)

        metrics.update_ultimate_metrics()

    def search_target_with_monitor(self, cur_pos, rgb, depth):
        for i in range(self.train_env.batch_size):
            if self.dones[i]:
                continue

            self.dino_results[i] = self.monitor.get_dino_results(
                rgb, depth, self.object_infos[i])
            distance_to_end = np.linalg.norm(
                np.array(cur_pos) - np.array(self.target_positions[i]))
            if self.dino_results[i] and not self.skips[i]:
                if distance_to_end <= 25:  # TODO: use 25 to compensate the dino interval
                    self.success[i] = True
                elif distance_to_end > 20:
                    self.early_end[i] = True

                if self.oracle_success[i] and self.early_end[i]:
                    self.dones[i] = True
                elif self.success[i]:
                    self.dones[i] = True

        return self.dones[0]

    def check_deviation(self, i):

        def target_distance_increasing_for_10frames(lst):
            if len(lst) < 10:
                return False
            sublist = lst[-10:]
            for i in range(1, len(sublist)):
                if sublist[i] < sublist[i - 1]:
                    return False
            return True

        if target_distance_increasing_for_10frames(self.distance_to_ends[i]):
            self.dones[i] = True

    def prepare_llm_inputs(self):
        raw_instructions = self.fetch_instruction()
        bev_images = self.bevs
        bev_depths = self.bev_depths
        drone1_start = [batch[-1] for batch in self.drone1_trajs]
        drone2_start = [batch[-1] for batch in self.drone2_trajs]
        point_clouds = self.update_pointclouds(drone2_start)
        drone1_traj = self.drone1_trajs
        drone2_traj = self.drone2_trajs
        target_positions = self.target_positions
        # TODO: in future, use drone2_pred direction to replace target position

        llm_inputs = {
            "bev_images": bev_images,
            "bev_depths": bev_depths,
            "point_clouds": point_clouds,
            "high_uav_locations": drone2_start,
            "low_uav_locations": drone1_start,
            "history_waypoints": drone1_traj,
            "history_waypoints2": drone2_traj,
            "description": raw_instructions,
            "target_positions": target_positions,
        }

        return llm_inputs

    def fetch_instruction(self):
        result = []
        for i in range(self.train_env.batch_size):
            result.append(self.raw_instructions[i])
        return result

    def fetch_from_observations(self, key):
        result = []
        for i in range(self.train_env.batch_size):
            result.append(self.observations[i][-1][key])
        return result

    def update_navigator_extra_statistics(self, metrics: Metrics, nav_time,
                                          nav_velocity):
        metrics.nav_time_stat += nav_time
        metrics.nav_velocity_stat += nav_velocity
        metrics.nav_stat_cnt += 1
        print("-----------------------------")
        print(
            f"mean velocity: {nav_velocity:.2f} m/s, navigation time: {nav_time:.2f} s"
        )
        print("-----------------------------")


def init_llm_planner(args):
    return CallPilot(checkpoint_path=args.llm_checkpoint_path,
                     device=args.device,
                     use_a_star=args.use_a_star)


# @hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def load_hydra_cfg():
    from hydra import compose, initialize

    file_path = "scripts/cfg"
    with initialize(config_path=file_path):
        cfg = compose(config_name="train")

    return cfg


# TODO: deal with the case when batch_size > 1
def init_navigator(cfg, train_env: AirVLNENV):
    navigator = Navigator(cfg, airsim_client=None)

    return navigator


def main():
    cfg = load_hydra_cfg()
    save_path = args.eval_save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()
    args.DistributedDataParallel = False

    # init env to deal with the communication between the simulator and the model
    train_env = AirVLNENV(batch_size=args.batchSize,
                          dataset_path=args.dataset_path,
                          save_path=save_path)

    # init planner, navigator and monitor
    llm_planner = init_llm_planner(args)
    navigator = init_navigator(cfg, train_env)
    dino_monitor = DinoMonitor(device=args.device)

    # init metrics
    log_path = os.path.join(save_path, "metrics_log.json")
    metrics = Metrics(log_path=log_path)

    # init object list
    object_desc_dict = dict()
    with open("data/config/object_new_name.json") as f:
        file = json.load(f)
        for item in file:
            object_desc_dict[item['object_name']] = item['new_name']

    drone2_collision_cnt = 0
    with torch.no_grad():
        dataset = BatchIterator(train_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        while True:
            env_batches = train_env.next_minibatch()
            if env_batches is None:
                break
            # states to record
            traj_status = TrajectoryStatus(env_batches, train_env,
                                           object_desc_dict, args)
            traj_status.init_dino_monitor(dino_monitor)

            pbar.update(n=train_env.batch_size)

            seq_name = traj_status.ori_data_dirs[0]
            llm_planner.update_traj_info(seq_name)

            # reset设置轨迹的初始状态
            # 我们的output应当包括：图片（第5帧）、位置（1-5帧）、姿态（1-5帧）、语言指令
            # reset时的output则是第0帧的上述内容
            outputs = train_env.reset()
            if train_env.is_NYC == True:
                llm_planner.reset_ortho_granularity(0.6)

            navigator.update_airsim_client(
                train_env.simulator_tool.airsim_clients[0][0])
            # 轨迹中添加初始状态
            traj_status.update_observation(outputs, [], airsim_collision=False)

            pre_time = None
            t = -1
            while True:
                t += 1
                if t >= args.maxWaypoints:
                    traj_status.dones[0] = True

                logger.info('Step: {} \t Completed: {} / {}'.format(
                    t,
                    int(train_env.index_data) - int(train_env.batch_size),
                    end_iter))

                cur_time = time.time()
                if pre_time is None:
                    pre_time = cur_time
                else:
                    logger.info('Time Cost : {} s'.format(
                        round(cur_time - pre_time, 2)))
                    pre_time = cur_time

                # check traj status
                if traj_status.check_traj_status(metrics):
                    break

                # waypoint + LLM预测轨迹点
                t1 = time.time()
                inputs = traj_status.prepare_llm_inputs()
                waypoint_list, _ = llm_planner(**inputs)
                t2 = time.time()
                print(f"llm prediction time: {t2-t1}")

                # navigate
                pos_list = []
                airsim_collision = False
                accumulate_distance = 0
                accumulate_time = 0
                start_time = time.time()
                for waypoint in waypoint_list:
                    # TODO: predict drone2 traj
                    drone2_waypoint = traj_status.target_positions
                    drone1_waypoint = [waypoint]
                    collision, drone2_collision, delta_time, delta_distance, end_pos, airsim_collision = train_env.makeActions(
                        drone1_waypoint, drone2_waypoint, navigator,
                        traj_status)
                    # update intermediate metrics
                    traj_status.steps += 1
                    traj_status.distance += delta_distance
                    traj_status.time += delta_time
                    accumulate_distance += delta_distance
                    accumulate_time += delta_time
                    # 细化轨迹，沿前进方向每隔5m添加一个点，多余的距离算到最后一个点上
                    if len(pos_list) > 0:
                        previous_pos = pos_list[-1]
                    else:
                        previous_pos = traj_status.drone1_trajs[0][-1]
                    delta = np.array(end_pos) - np.array(previous_pos)
                    length = np.linalg.norm(delta)
                    delta_unit = delta / length
                    cur_pos = previous_pos
                    while length >= 10:
                        length -= 5
                        cur_pos = cur_pos + delta_unit * 5
                        pos_list.append(cur_pos.tolist())
                    pos_list.append(end_pos.tolist())
                    if drone2_collision:
                        drone2_collision_cnt += 1
                        print(f"Drone2 collides! {drone2_collision_cnt} times")
                        break
                    if collision:
                        break
                    if traj_status.dones[0]:
                        break
                if accumulate_time > 1:
                    end_time = time.time()
                    total_nav_time = end_time - start_time
                    mean_velocity = accumulate_distance / accumulate_time
                    traj_status.update_navigator_extra_statistics(
                        metrics, total_nav_time, mean_velocity)

                outputs = train_env.get_obs()
                traj_status.update_observation(outputs, pos_list,
                                               airsim_collision)


if __name__ == "__main__":
    # Airsim simulator server is very unstable, so we need to retry if it fails
    while True:
        try:
            main()
            break
        except (msgpackrpc.error.TimeoutError, ConnectionError,
                TimeoutError) as e:
            print(f"AirSim connection error, retrying: {e}")
            traceback.print_exc()
            time.sleep(3)
            continue
