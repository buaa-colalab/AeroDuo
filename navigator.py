import copy
import json
import os
import time
from collections import deque

import airsim
import cv2
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.utils import ExplorationType, set_exploration_type

from airsim_plugin.AirVLNSimulatorClientTool import Imu, State
from scripts.ppo import PPO
from scripts.utils import vec_to_new_frame


class Navigator:

    def __init__(self,
                 cfg,
                 airsim_client: airsim.MultirotorClient,
                 simple_action: bool = False,
                 activate_drone2: bool = True):
        self.cfg = cfg
        self.device = cfg.device
        self.airsim_client = airsim_client
        self.lidar_hbeams = int(360 / self.cfg.sensor.lidar_hres)
        self.pos_queue = deque(maxlen=20)

        self.apply_simple_action = simple_action
        if not self.apply_simple_action:
            print("-------------------------")
            print("Using NavRL policy")
            print("-------------------------")
            self.policy = self.init_model()
            self.policy.eval()

        self.activate_drone2 = activate_drone2
        if activate_drone2:
            self.drone2_pos_queue = deque(maxlen=10)
            self.drone2_collision = False

        self.drone1_img_idx = 0
        self.drone2_img_idx = 0

    def update_airsim_client(self, airsim_client: airsim.MultirotorClient):
        self.airsim_client = airsim_client

    def get_Image_Response(self):
        image_check_log = "image_check_log"
        Drone1_ImageRequest = []
        Drone2_ImageRequest = []
        Drone1_ImageRequest.append(
            airsim.ImageRequest("FrontCamera",
                                airsim.ImageType.Scene,
                                pixels_as_float=False,
                                compress=False))
        # Drone1_ImageRequest.append(airsim.ImageRequest("FrontCamera", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))
        Drone2_ImageRequest.append(
            airsim.ImageRequest('BEVCamera',
                                airsim.ImageType.Scene,
                                pixels_as_float=False,
                                compress=False))
        # Drone2_ImageRequest.append(airsim.ImageRequest('BEVCamera', airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))
        drone1_resp = self.airsim_client.simGetImages(Drone1_ImageRequest,
                                                      vehicle_name="Drone_1")
        drone2_resp = self.airsim_client.simGetImages(Drone2_ImageRequest,
                                                      vehicle_name="Drone_2")
        front_img = drone1_resp[0]
        front_img = np.frombuffer(front_img.image_data_uint8,
                                  dtype=np.uint8).reshape(
                                      front_img.height, front_img.width, 3)
        bev_img = drone2_resp[0]
        bev_img = np.frombuffer(bev_img.image_data_uint8,
                                dtype=np.uint8).reshape(
                                    bev_img.height, bev_img.width, 3)
        cv2.imwrite(
            os.path.join(image_check_log, f"drone1_{self.drone1_img_idx}.png"),
            front_img)
        cv2.imwrite(
            os.path.join(image_check_log, f"drone2_{self.drone2_img_idx}.png"),
            bev_img)
        self.drone1_img_idx += 1
        self.drone2_img_idx += 1

    def get_image_for_monitor(self, cmd_vel):
        yaw = np.arctan2(cmd_vel[1].item(), cmd_vel[0].item())
        yaw = int(np.rad2deg(yaw) // 10 * 10) % 360
        camera_name = f"Camera_{yaw}"
        ImageRequest = [
            airsim.ImageRequest(camera_name,
                                airsim.ImageType.Scene,
                                pixels_as_float=False,
                                compress=False),
            airsim.ImageRequest(camera_name,
                                airsim.ImageType.DepthPerspective,
                                pixels_as_float=True,
                                compress=False)
        ]
        image_response = self.airsim_client.simGetImages(
            ImageRequest, vehicle_name="Drone_1")
        rgb = image_response[0]
        rgb = np.frombuffer(rgb.image_data_uint8,
                            dtype=np.uint8).reshape(rgb.height, rgb.width, 3)
        depth = image_response[1]
        depth = airsim.list_to_2d_float_array(depth.image_data_float,
                                              depth.width, depth.height)

        return rgb, depth

    def init_model(self) -> PPO:
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10
        observation_spec = CompositeSpec(
            {
                "agents":
                CompositeSpec({
                    "observation":
                    CompositeSpec({
                        "state":
                        UnboundedContinuousTensorSpec(
                            (observation_dim, ), device=self.cfg.device),
                        "lidar":
                        UnboundedContinuousTensorSpec(
                            (1, self.lidar_hbeams,
                             self.cfg.sensor.lidar_vbeams),
                            device=self.cfg.device),
                        "direction":
                        UnboundedContinuousTensorSpec(
                            (1, 3), device=self.cfg.device),
                        "dynamic_obstacle":
                        UnboundedContinuousTensorSpec(
                            (1, self.cfg.algo.feature_extractor.dyn_obs_num,
                             num_dim_each_dyn_obs_state),
                            device=self.cfg.device),
                    }),
                }).expand(1)
            },
            shape=[1],
            device=self.cfg.device)

        action_dim = 3
        action_spec = CompositeSpec({
            "agents":
            CompositeSpec({
                "action":
                UnboundedContinuousTensorSpec((action_dim, ),
                                              device=self.cfg.device),
            })
        }).expand(1, action_dim).to(self.cfg.device)

        policy = PPO(self.cfg.algo, observation_spec, action_spec,
                     self.cfg.device)

        file_dir = "scripts/ckpts"
        checkpoint = "navrl_checkpoint.pt"

        policy.load_state_dict(torch.load(os.path.join(file_dir, checkpoint)))
        return policy

    def get_distance_data(self) -> torch.Tensor:
        # TODO: whether to use a finer granularity to reduce the error caused by innacurate start angle
        v_angles = [-10, 0, 10, 20]
        # h_angles_ori = list(range(0, 360, 10))
        start_angle = np.rad2deg(
            np.arctan2(self.target_dir[1].cpu().numpy(),
                       self.target_dir[0].cpu().numpy()))
        # raw_start_angle = round(start_angle / 10) * 10
        raw_start_angle = start_angle // 10 * 10
        h_angles = [
            int((raw_start_angle + i * 10) % 360)
            for i in range(self.lidar_hbeams)
        ]

        distance_data = []

        for h in h_angles:
            for v in v_angles:
                sensor_name = f"DistanceSensor_v{v}_h{h}"
                data = self.airsim_client.getDistanceSensorData(
                    sensor_name, vehicle_name="Drone_1")
                distance = data.distance
                distance_data.append(distance)

        distance_data = torch.tensor(distance_data, device=self.device)
        return distance_data

    def check_obstacle(self, lidar_scan, dyn_obs_states):
        # return true if there is obstacles in the range
        # has_static = not torch.all(lidar_scan == 0.)
        # has_static = not torch.all(lidar_scan[..., 1:] < 0.2) # hardcode to tune
        quarter_size = lidar_scan.shape[2] // 4
        first_quarter_check, last_quarter_check = torch.all(
            lidar_scan[:, :, :quarter_size,
                       1:] < 0.2), torch.all(lidar_scan[:, :, -quarter_size:,
                                                        1:] < 0.2)
        has_static = (not first_quarter_check) or (not last_quarter_check)
        has_dynamic = not torch.all(dyn_obs_states == 0.)
        return has_static or has_dynamic

    def get_action(self, pos: torch.Tensor, vel: torch.Tensor,
                   goal: torch.Tensor,
                   lidar: torch.Tensor) -> torch.Tensor:  # use world velocity
        rpos = goal - pos
        distance = rpos.norm(dim=-1, keepdim=True)
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)

        target_dir_2d = self.target_dir.clone()
        target_dir_2d[2] = 0.

        rpos_clipped = rpos / distance.clamp(1e-6)  # start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped,
                                          target_dir_2d).squeeze(0).squeeze(0)

        # "relative" velocity
        vel_g = vec_to_new_frame(vel, target_dir_2d).squeeze(0).squeeze(
            0)  # goal velocity

        # drone_state = torch.cat([rpos_clipped, orientation, vel_g], dim=-1).squeeze(1)
        drone_state = torch.cat(
            [rpos_clipped_g, distance_2d, distance_z, vel_g],
            dim=-1).unsqueeze(0)

        # Lidar States
        # TODO: check the lidar format in airsim
        # lidar_scan = torch.tensor(lidar, device=self.cfg.device)
        # lidar_scan = (lidar_scan - pos).norm(dim=-1).clamp_max(self.cfg.sensor.lidar_range).reshape(1, 1, self.lidar_hbeams, self.cfg.sensor.lidar_vbeams)
        lidar_scan = lidar.clamp_max(self.cfg.sensor.lidar_range).reshape(
            1, 1, self.lidar_hbeams, self.cfg.sensor.lidar_vbeams)
        lidar_scan = self.cfg.sensor.lidar_range - lidar_scan

        # dynamic obstacle states (no obstacle, init as a zero tensor)
        dyn_obs_states = torch.zeros(
            1,
            1,
            self.cfg.algo.feature_extractor.dyn_obs_num,
            10,
            device=self.cfg.device)

        # states
        target_dir_2d = target_dir_2d.to(device=self.device)
        drone_state = drone_state.to(device=self.device)
        obs = TensorDict({
            "agents":
            TensorDict({
                "observation":
                TensorDict({
                    "state": drone_state,
                    "lidar": lidar_scan,
                    "direction": target_dir_2d,
                    "dynamic_obstacle": dyn_obs_states
                })
            })
        })

        has_obstacle_in_range = self.check_obstacle(lidar_scan, dyn_obs_states)
        if (has_obstacle_in_range):
            with set_exploration_type(ExplorationType.MEAN):
                output = self.policy(obs)
            vel_world = output["agents", "action"]
        else:
            vel_world = (goal - pos) / torch.norm(
                goal - pos) * self.cfg.algo.actor.action_limit

        return vel_world

    def simple_action(self, pos: torch.Tensor, goal: torch.Tensor,
                      drone1_pos: torch.Tensor) -> torch.Tensor:
        # TODO: use the distance between high and low uav to constrain the velocity
        vel_world = (goal[:-1] - pos[:-1]) / torch.norm(
            goal[:-1] - pos[:-1]
        ) * self.cfg.algo.actor.action_limit  #* min(10 / delta_pos, 1)
        assert vel_world.shape == (2, )
        return vel_world

    def simple_action_drone1(self, pos: torch.Tensor,
                             goal: torch.Tensor) -> torch.Tensor:
        vel_world = (goal -
                     pos) / torch.norm(goal -
                                       pos) * self.cfg.algo.actor.action_limit
        assert vel_world.shape == (3, )
        return vel_world

    def navigate_drone2(self, goal: torch.Tensor, drone1_pos) -> bool:
        current_kinematics_state = self.airsim_client.getMultirotorState(
            vehicle_name="Drone_2").kinematics_estimated
        new_state = current_kinematics_state
        # TODO: a more accurate way to determine the height
        height = -150 if new_state.position.z_val < -120 else -80
        new_state.position.z_val = height
        self.airsim_client.simSetKinematics(new_state,
                                            ignore_collision=True,
                                            vehicle_name="Drone_2")
        self.airsim_client.simContinueForFrames(1)

        pos = torch.Tensor([p_i for p_i in new_state.position])
        pos[-1] = height
        pos = pos.to(self.device)
        self.drone2_pos_queue.append(pos)

        # check if the drone2 has reached the goal
        distance = torch.norm(pos[:-1] - goal[:-1])
        if distance < 1:
            self.airsim_client.hoverAsync(vehicle_name="Drone_2")
            return pos

        # check if the drone2 is stuck
        if len(self.drone2_pos_queue) == self.drone2_pos_queue.maxlen:
            recent_loc = pos
            history_loc = self.drone2_pos_queue.popleft()
            delta_distance = torch.norm(history_loc - recent_loc)
            if delta_distance < 0.1:
                print('Drone2 move on path api: stuck max len')
                # raise RuntimeError('Drone2 stuck')
                self.drone2_collision = True
                self.airsim_client.simPause(True)
                return pos

        cmd_vel = self.simple_action(pos, goal, drone1_pos)
        self.airsim_client.simPause(False)
        self.airsim_client.moveByVelocityZAsync(vx=cmd_vel[0].item(),
                                                vy=cmd_vel[1].item(),
                                                z=pos[-1].item(),
                                                duration=1,
                                                vehicle_name="Drone_2")
        # path = [airsim.Vector3r(pos[0].item(), pos[1].item(), pos[2].item()), airsim.Vector3r(goal[0].item(), goal[1].item(), goal[2].item())]
        # self.airsim_client.moveOnPathAsync(
        #     path=path,
        #     velocity=2,
        #     drivetrain=airsim.DrivetrainType.ForwardOnly,
        #     yaw_mode=airsim.YawMode(is_rate=False),
        #     lookahead=3,
        #     adaptive_lookahead=1
        # )

        return pos

    def dino_search(self, cmd_vel, traj_status, cur_pos):
        # TODO: run dino in a certain frequency
        # more camera and choose one camera according to the orientation
        self.pause_cnt += 1
        done = False
        if self.pause_cnt == 3:
            self.pause_cnt = 0
            # self.get_Image_Response()
            rgb, depth = self.get_image_for_monitor(cmd_vel)
            image_request = [
                airsim.ImageRequest('BEVCamera',
                                    airsim.ImageType.Scene,
                                    pixels_as_float=False,
                                    compress=False),
                airsim.ImageRequest('BEVCamera',
                                    airsim.ImageType.DepthPlanar,
                                    pixels_as_float=True,
                                    compress=False)
            ]
            drone2_resp = self.airsim_client.simGetImages(
                image_request, vehicle_name="Drone_2")
            bev_img = drone2_resp[0]
            bev_img = np.frombuffer(bev_img.image_data_uint8,
                                    dtype=np.uint8).reshape(
                                        bev_img.height, bev_img.width, 3)

            done = traj_status.search_target_with_monitor(cur_pos, rgb, depth)

        return done

    # TODO: implement navigator into AirVLNENV
    def navigate(self,
                 goal: torch.Tensor,
                 drone2_goal: torch.Tensor = None,
                 traj_status=None) -> bool:
        self.pause_cnt = 0
        self.current_steps = 0

        results = []
        state_sensor = State(self.airsim_client, drone_name='Drone_1')
        imu_sensor = Imu(self.airsim_client,
                         imu_name='Imu',
                         drone_name='Drone_1')
        drone2_state_sensor = State(self.airsim_client, drone_name='Drone_2')

        self.pos_queue.clear()
        self.collision = False
        self.airsim_client.enableApiControl(True, vehicle_name='Drone_1')
        self.airsim_client.armDisarm(True, vehicle_name='Drone_1')
        if self.activate_drone2:
            self.drone2_pos_queue.clear()
            self.drone2_collision = False
            self.airsim_client.enableApiControl(True, vehicle_name='Drone_2')
            self.airsim_client.armDisarm(True, vehicle_name='Drone_2')

        self.airsim_client.simPause(True)
        start_time = self.airsim_client.getMultirotorState(
            vehicle_name="Drone_1").timestamp * 1e-10
        start_pos = self.airsim_client.getMultirotorState(
            vehicle_name="Drone_1").kinematics_estimated.position
        start_pos = np.array([p_i for p_i in start_pos])

        airsim_collision_flag = False

        while True:
            # obtain the current state of drone
            current_kinematics_state = self.airsim_client.getMultirotorState(
                vehicle_name="Drone_1").kinematics_estimated
            pos = torch.Tensor(
                [p_i for p_i in current_kinematics_state.position])
            vel = torch.Tensor(
                [v_i for v_i in current_kinematics_state.linear_velocity])
            pos = pos.to(self.device)
            vel = vel.to(self.device)
            self.target_dir = goal - pos

            # update the position queue
            self.pos_queue.append(pos)

            # check if the drone is stuck
            if len(self.pos_queue) == self.pos_queue.maxlen:
                recent_loc = pos
                history_loc = self.pos_queue.popleft()
                delta_distance = torch.norm(history_loc - recent_loc)
                if delta_distance < 0.1:
                    print('Drone1 move on path api: collision')
                    self.collision = True
                    airsim_collision_flag = self.airsim_client.simGetCollisionInfo(
                        vehicle_name="Drone_1").has_collided
                    self.airsim_client.simPause(True)
                    break

            # check if the drone has reached the goal
            distance = torch.norm(pos - goal)
            if distance < 1:
                self.airsim_client.simPause(True)
                break

            # get the lidar scan (we temporally use airsim distance sensor to achieve our goal)
            raw_lidar = self.get_distance_data()

            # get action
            # TODO: implement the VO safety shield
            if self.apply_simple_action:
                cmd_vel = self.simple_action_drone1(pos, goal)
            else:
                cmd_vel = self.get_action(pos, vel, goal, raw_lidar).squeeze()

            # move by velocity
            # TODO: use threading to optimize the logic here
            if self.activate_drone2:
                pos2 = self.navigate_drone2(drone2_goal, pos)
                if self.drone2_collision:
                    break
            # self.airsim_client.simPause(False)
            self.airsim_client.moveByVelocityAsync(vx=cmd_vel[0].item(),
                                                   vy=cmd_vel[1].item(),
                                                   vz=cmd_vel[2].item(),
                                                   duration=1,
                                                   vehicle_name="Drone_1")

            time.sleep(0.1)
            self.airsim_client.simPause(True)
            # airsim_collision_flag = self.airsim_client.simGetCollisionInfo(vehicle_name="Drone_1").has_collided
            # if airsim_collision_flag:
            #     print("Collision detected!")
            #     self.collision = True
            #     break

            cur_pos = self.airsim_client.getMultirotorState(
                vehicle_name="Drone_1").kinematics_estimated.position
            cur_pos = np.array([p_i for p_i in cur_pos])
            done = self.dino_search(cmd_vel, traj_status, cur_pos)
            if done:
                break
            self.current_steps += 1
            if self.current_steps == 1000:
                break

        end_time = self.airsim_client.getMultirotorState(
            vehicle_name="Drone_1").timestamp * 1e-10
        end_pos = self.airsim_client.getMultirotorState(
            vehicle_name="Drone_1").kinematics_estimated.position
        end_pos = np.array([p_i for p_i in end_pos])
        delta_time = end_time - start_time
        delta_distance = np.linalg.norm(end_pos - start_pos)

        # update state and imu
        state_info = copy.deepcopy(state_sensor.retrieve())
        state_info2 = copy.deepcopy(drone2_state_sensor.retrieve())
        imu_info = copy.deepcopy(imu_sensor.retrieve())
        results.append({'sensors': {'state': state_info, 'imu': imu_info}})

        return {
            'states': results,
            'drone2_states': state_info2,
            'collision': (self.collision, self.drone2_collision),
            'airsim_collision': airsim_collision_flag
        }, delta_time, delta_distance, end_pos
