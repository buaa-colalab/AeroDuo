import copy
import math
import multiprocessing
import os
import random
import sys
import threading
import time
from collections import deque

import airsim
import cv2
import msgpackrpc
import numpy as np
import tqdm

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")

# from utils.logger import logger


class BaseSensor:

    def __init__(self) -> None:
        pass

    def retrieve(self):
        raise NotImplementedError()


class State(BaseSensor):

    def __init__(self, client, drone_name=''):
        self.data = {
            'position': None,
            'linear_velocity': None,
            'linear_acceleration': None,
            'orientation': None,
            'angular_velocity': None,
            'angular_acceleration': None
        }
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name

    def retrieve(self):
        data = self.client.getMultirotorState(vehicle_name=self.drone_name)
        collision = {}
        collision_info = self.client.simGetCollisionInfo(
            vehicle_name=self.drone_name)
        collision['has_collided'] = collision_info.has_collided
        # collision['has_collided'] = data.collision.has_collided
        # collision['normal'] = list(data.collision.normal)
        # collision['impact_point'] = list(data.collision.impact_point)
        # collision['position'] = list(data.collision.position)
        # collision['penetration_depth'] = data.collision.penetration_depth
        # collision['time_stamp'] = data.collision.time_stamp
        collision['object_name'] = data.collision.object_name
        # collision['object_id'] = data.collision.object_id
        gps_location = [
            data.gps_location.latitude, data.gps_location.longitude,
            data.gps_location.altitude
        ]
        timestamp = data.timestamp
        position = list(data.kinematics_estimated.position)
        linear_velocity = list(data.kinematics_estimated.linear_velocity)
        linear_acceleration = list(
            data.kinematics_estimated.linear_acceleration)
        orientation = list(data.kinematics_estimated.orientation)
        angular_velocity = list(data.kinematics_estimated.angular_velocity)
        angular_acceleration = list(
            data.kinematics_estimated.angular_acceleration)

        self.data.update({
            'collision': collision,
            'gps_location': gps_location,
            'timestamp': timestamp,
            'position': position,
            'linear_velocity': linear_velocity,
            'linear_acceleration': linear_acceleration,
            'orientation': orientation,
            'angular_velocity': angular_velocity,
            'angular_acceleration': angular_acceleration
        })
        return self.data


class Imu(BaseSensor):

    def __init__(self, client, drone_name='', imu_name=''):
        self.data = {}
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name
        self.imu_name = imu_name

    def retrieve(self):
        data = self.client.getImuData(imu_name=self.imu_name,
                                      vehicle_name=self.drone_name)
        time_stamp = data.time_stamp
        orientation = data.orientation
        angular_velocity = list(data.angular_velocity)
        linear_acceleration = list(data.linear_acceleration)
        q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        rotation_matrix = np.array(([
            1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q3 * q0),
            2 * (q1 * q3 + q2 * q0)
        ], [
            2 * (q1 * q2 + q3 * q0), 1 - 2 * (q1 * q1 + q3 * q3),
            2 * (q2 * q3 - q1 * q0)
        ], [
            2 * (q1 * q3 - q2 * q0), 2 * (q2 * q3 + q1 * q0),
            1 - 2 * (q1 * q1 + q2 * q2)
        ])).tolist()
        self.data.update({
            'time_stamp': time_stamp,
            'rotation': rotation_matrix,
            'orientation': list(data.orientation),
            'linear_acceleration': linear_acceleration,
            'angular_velocity': angular_velocity
        })
        return self.data


class MyThread(threading.Thread):

    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False

    def run(self):
        # try:
        self.result = self.func(*self.args)
        # except Exception as e:
        #     print(e)
        #     self.flag_ok = False
        # else:
        self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None


class AirVLNSimulatorClientTool:

    def __init__(self, machines_info) -> None:
        self.machines_info = copy.deepcopy(machines_info)
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])]
                               for item in machines_info]
        self.airsim_ports = []
        self.airsim_ip = '127.0.0.1'
        self._init_check()
        self.objects_name_cnt = [[0 for _ in list(item['open_scenes'])]
                                 for item in machines_info]
        self.test_cnt = 0

    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    def _confirmSocketConnection(self,
                                 socket_client: msgpackrpc.Client) -> bool:
        try:
            socket_client.call('ping')
            print("Connected\t{}:{}".format(socket_client.address._host,
                                            socket_client.address._port))
            return True
        except:
            try:
                print("Ping returned false\t{}:{}".format(
                    socket_client.address._host, socket_client.address._port))
            except:
                print('Ping returned false')
            return False

    def _confirmConnection(self) -> bool:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    confirmed = False
                    count = 0
                    while not confirmed and count < 30:
                        try:
                            self.airsim_clients[index_1][
                                index_2].confirmConnection()
                            confirmed = True
                        except Exception as e:
                            time.sleep(1)
                            print('failed', e)
                            count += 1
                            pass

        return confirmed

    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients

        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass

        self.socket_clients = []
        return

    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass

        self.airsim_clients = [[None for _ in list(item['open_scenes'])]
                               for item in self.machines_info]
        return

    def run_call(self, airsim_timeout: int = 300) -> None:
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'],
                                                     item['SOCKET_PORT']),
                                  timeout=300))

        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                print('cannot establish socket')
                raise Exception('cannot establish socket')

        self.socket_clients = socket_clients

        before = time.time()
        self._closeConnection()

        def _run_command(index, socket_client: msgpackrpc.Client):
            print(
                f'开始打开场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}'
            )
            print(f'gpus: {self.machines_info[index]}')
            result = socket_client.call(
                'reopen_scenes', socket_client.address._host,
                list(
                    zip(self.machines_info[index]['open_scenes'],
                        self.machines_info[index]['gpus'])))
            if result[0] == False:
                print(
                    f'打开场景失败，机器: {socket_client.address._host}:{socket_client.address._port}'
                )
                raise Exception('打开场景失败')
            assert len(result[1]) == 2, '打开场景失败'
            print('waiting for airsim connection...')
            time.sleep(3 * len(self.machines_info[index]['open_scenes']) + 35)
            ip = result[1][0]
            if type(ip) is not str:
                ip = ip.decode('utf-8')
            ports = result[1][1]
            self.airsim_ip = ip
            self.airsim_ports = ports
            assert str(ip) == str(socket_client.address._host), '打开场景失败'
            assert len(ports) == len(
                self.machines_info[index]['open_scenes']), '打开场景失败'
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                else:
                    self.airsim_clients[index][i] = airsim.MultirotorClient(
                        ip=ip, port=port, timeout_value=airsim_timeout)
                    print(port)

            print(
                f'打开场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}'
            )
            return ports

        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(MyThread(_run_command, (index, socket_client)))
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            thread.get_result()
            thread_results.append(thread.flag_ok)
        threads = []

        if not (np.array(thread_results) == True).all():
            raise Exception('打开场景失败')

        after = time.time()
        diff = after - before
        print(f"启动时间：{diff}")

        assert self._confirmConnection(), 'server connect failed'
        self._closeSocketConnection()

    def collect_DDP(self, data_dir, workers):

        def init_worker(index, lock):
            with lock:
                multiprocessing.current_process(
                ).client_port = self.machines_info[0]['SOCKET_PORT']
                multiprocessing.current_process(
                ).machine_ip = self.machines_info[0]['MACHINE_IP']
                multiprocessing.current_process(
                ).client = self.airsim_clients[0][index.value]
                multiprocessing.current_process().port = self.airsim_ports[
                    index.value]
                index.value += 1

        index = multiprocessing.Value('i', 0)
        lock = multiprocessing.Lock()
        with multiprocessing.Pool(workers,
                                  initializer=init_worker,
                                  initargs=(index, lock)) as p:
            r = list(
                tqdm.tqdm(p.imap_unordered(collect, data_dir),
                          total=len(data_dir)))

    def closeScenes(self):
        try:
            socket_clients = []
            for index, item in enumerate(self.machines_info):
                socket_clients.append(
                    msgpackrpc.Client(msgpackrpc.Address(
                        item['MACHINE_IP'], item['SOCKET_PORT']),
                                      timeout=300))

            for socket_client in socket_clients:
                if not self._confirmSocketConnection(socket_client):
                    print('cannot establish socket')
                    raise Exception('cannot establish socket')

            self.socket_clients = socket_clients

            self._closeConnection()

            def _run_command(index, socket_client: msgpackrpc.Client):
                print(
                    f'开始关闭所有场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}'
                )
                result = socket_client.call('close_scenes',
                                            socket_client.address._host)
                print(
                    f'关闭所有场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}'
                )
                return

            threads = []
            for index, socket_client in enumerate(socket_clients):
                threads.append(MyThread(_run_command, (index, socket_client)))
            for thread in threads:
                thread.setDaemon(True)
                thread.start()
            for thread in threads:
                thread.join()
            threads = []

            self._closeSocketConnection()
        except Exception as e:
            print(e)

    def move_path_by_waypoints(self,
                               waypoints_list,
                               start_states,
                               use_bev=True,
                               camera_pose=None):
        velocity = 1
        drivetrain = airsim.DrivetrainType.ForwardOnly
        yaw_mode = airsim.YawMode(is_rate=False)
        lookahead = 3
        adaptive_lookahead = 1

        def move_path(airsim_client: airsim.VehicleClient,
                      waypoints,
                      start_state,
                      vehicle_name="Drone_1"):
            results = []
            state_sensor = State(airsim_client, drone_name=vehicle_name)
            imu_sensor = Imu(airsim_client,
                             imu_name='Imu',
                             drone_name=vehicle_name)
            path = [airsim.Vector3r(*waypoint[0:3]) for waypoint in waypoints]
            airsim_client.enableApiControl(True, vehicle_name=vehicle_name)
            airsim_client.armDisarm(True, vehicle_name=vehicle_name)
            airsim_client.simPause(False)
            # TODO: simSetKinematics/Pose
            airsim_client.simSetKinematics(start_state,
                                           ignore_collision=False,
                                           vehicle_name=vehicle_name)
            # airsim_client.simPause(True)
            # airsim_client.simPause(False)
            # print("start_pose", start_pose)
            # state_info = state_sensor.retrieve()
            # print('pre', state_info)
            airsim_client.moveOnPathAsync(
                path=path,
                velocity=velocity,
                drivetrain=drivetrain,
                yaw_mode=yaw_mode,
                lookahead=lookahead,
                adaptive_lookahead=adaptive_lookahead,
                vehicle_name=vehicle_name)
            # state_info = state_sensor.retrieve()
            # print('post',state_info['position'])
            target_idx = 5  # 5 / 7
            current_idx = 0
            pos_queue = deque(maxlen=20)
            start_time = time.perf_counter()
            collision = False
            distance = 10000
            while True:
                time.sleep(0.005)  # simcontinuetime xxxs
                if time.perf_counter() - start_time > 5:
                    return None
                target = path[current_idx]
                state_info = copy.deepcopy(state_sensor.retrieve())
                imu_info = copy.deepcopy(imu_sensor.retrieve())
                position = np.array(state_info['position'])
                # collision = state_info['collision']['has_collided']
                # env collision not used
                pos_queue.append(position)
                if len(pos_queue) == pos_queue.maxlen:
                    recent_loc = position
                    history_loc = pos_queue.popleft()
                    delta_distance = np.linalg.norm(history_loc - recent_loc)
                    if delta_distance < 0.1:
                        print('move on path api: stuck max len')
                        collision = True
                        break
                new_distance = np.linalg.norm(
                    position -
                    np.array([target.x_val, target.y_val, target.z_val]))
                if new_distance > distance:
                    results.append(
                        {'sensors': {
                            'state': state_info,
                            'imu': imu_info
                        }})
                    current_idx += 1
                    if current_idx == target_idx:
                        airsim_client.simPause(True)
                        break
                    else:
                        distance = 10000
                else:
                    distance = new_distance
            # if collision:
            #     print("collision: ", state_info['collision'])
            return {'states': results, 'collision': collision}

        def swarm_move_path(airsim_client: airsim.VehicleClient,
                            waypoints_list, start_state):
            if use_bev:
                waypoints_list2 = [[x, y, z - 10]
                                   for x, y, z in waypoints_list]
                result = move_path(airsim_client,
                                   waypoints_list,
                                   start_state,
                                   vehicle_name='Drone_1')
                result2 = move_path(airsim_client,
                                    waypoints_list2,
                                    start_state,
                                    vehicle_name='Drone_2')
                result["collistion_uav2"] = result2["collision"]

                return result
            else:
                return move_path(airsim_client, waypoints_list, start_state)

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(swarm_move_path,
                             (self.airsim_clients[index_1][index_2],
                              waypoints_list[index_1][index_2],
                              start_states[index_1][index_2])))
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        result_poses_list = []
        error_flag = False
        for index_1, _ in enumerate(threads):
            result_poses_list.append([])
            for index_2, _ in enumerate(threads[index_1]):
                result = threads[index_1][index_2].get_result()
                result_poses_list[index_1].append(result)
                if result is None:
                    error_flag = True
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            print('move path by waypoints failed.')
            return None
        if error_flag:
            return None
        return result_poses_list

    def setPoses(self, poses: list, vehicle_name: str = 'Drone_1') -> bool:

        def _setPoses(airsim_client: airsim.VehicleClient,
                      pose: airsim.Pose) -> None:
            if airsim_client is None:
                raise Exception('error')
                return

            airsim_client.simSetKinematics(state=pose,
                                           ignore_collision=True,
                                           vehicle_name=vehicle_name)
            airsim_client.simContinueForFrames(1)
            airsim_client.simPause(True)

            return

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_setPoses, (self.airsim_clients[index_1][index_2],
                                         poses[index_1][index_2])))
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            print('setPoses失败')
            return False

        return True

    def setObjects(self, object_list: list):

        def _setObject(airsim_client: airsim.VehicleClient,
                       object_info: dict) -> None:
            if airsim_client is None:
                raise Exception('error')
                return
            asset_name = object_info['asset_name']
            pose = object_info['pose']
            scale = object_info['scale']
            object_cnt = object_info['object_cnt']
            if object_cnt > 0:
                airsim_client.simDestroyObject('my_object_' +
                                               str(object_cnt - 1))
            success = airsim_client.simSpawnObject('my_object_' +
                                                   str(object_cnt),
                                                   asset_name,
                                                   pose,
                                                   scale,
                                                   physics_enabled=False,
                                                   is_blueprint=False)
            airsim_client.simContinueForFrames(1)
            airsim_client.simPause(True)
            return success

        threads = []
        thread_results = []
        cnt = 0
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                object_list[cnt]['object_cnt'] = self.objects_name_cnt[
                    index_1][index_2]
                threads[index_1].append(
                    MyThread(_setObject,
                             (self.airsim_clients[index_1][index_2],
                              object_list[cnt])))
                self.objects_name_cnt[index_1][index_2] += 1
                cnt += 1

        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            print('set Object失败')
            return False
        return True

    @staticmethod
    def project3d_to_image(airsim_client: airsim.VehicleClient):
        import math

        # intrinsic parameters
        resolution = (1024, 1024)
        FoV = 120
        w, h = resolution
        c_x, c_y = w / 2, h / 2
        f_x = w / (2 * math.tan(math.radians(FoV / 2)))
        f_y = h / (2 * math.tan(math.radians(FoV / 2)))
        intri_mat = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

        vehicle_kinematics = airsim_client.simGetGroundTruthKinematics(
            'Drone_2')
        r_rela = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        r = r_rela

        t = np.array([val for val in vehicle_kinematics.position]).transpose()
        drone_location = airsim_client.simGetGroundTruthKinematics(
            'Drone_1').position
        homogeneous_3d = np.array([val for val in drone_location] + [1])

        # calculate the relative coordinate in frame of camera
        camera_frame_3d = r @ (homogeneous_3d[:3] - t)
        homogeneous_2d = intri_mat @ camera_frame_3d
        homogeneous_2d = homogeneous_2d / homogeneous_2d[-1]

        return homogeneous_2d

    def mark_projection(self,
                        bev_image,
                        frd_coordinate,
                        flu_coordinate=None,
                        single_image=True):
        color_list = [[0, 255, 0], [255, 0, 0], [0, 0,
                                                 255]]  # green, blue, red
        x_mark, y_mark, _ = frd_coordinate
        x_mark, y_mark = round(x_mark), round(y_mark)
        half_size = 2
        bev_copy = bev_image.copy()
        for i in range(x_mark - half_size, x_mark + half_size + 1):
            for j in range(y_mark - half_size, y_mark + half_size + 1):
                if 0 <= i < bev_copy.shape[1] and 0 <= j < bev_copy.shape[0]:
                    bev_copy[j, i] = color_list[0]
        if flu_coordinate is not None:
            x_mark_2, y_mark_2, _ = flu_coordinate
            x_mark_2, y_mark_2 = round(x_mark_2), round(y_mark_2)
            half_size = 2
            if x_mark_2 == x_mark and y_mark_2 == y_mark:
                color = color_list[2]
            else:
                color = color_list[1]
            for i in range(x_mark_2 - half_size, x_mark_2 + half_size + 1):
                for j in range(y_mark_2 - half_size, y_mark_2 + half_size + 1):
                    if 0 <= i < bev_copy.shape[1] and 0 <= j < bev_copy.shape[
                            0]:
                        bev_copy[j, i] = color
        if single_image == True:
            cv2.imwrite('/home/wrp/test/test.png', bev_copy)
        else:
            cv2.imwrite(
                '/home/wrp/test/test_image/test_{:04d}.png'.format(
                    self.test_cnt), bev_copy)
            self.test_cnt += 1

    def getImageResponses(self, cameras=['FrontCamera'], use_bev=True):

        def _getImages(airsim_client: airsim.MultirotorClient):
            if airsim_client is None:
                raise Exception('client is None.')
            ImageRequest = []
            for camera_name in cameras:
                ImageRequest.append(
                    airsim.ImageRequest(camera_name,
                                        airsim.ImageType.Scene,
                                        pixels_as_float=False,
                                        compress=False))
                ImageRequest.append(
                    airsim.ImageRequest(camera_name,
                                        airsim.ImageType.DepthPerspective,
                                        pixels_as_float=True,
                                        compress=False))
            image_datas = airsim_client.simGetImages(vehicle_name='Drone_1',
                                                     requests=ImageRequest)
            images, depth_images = [], []
            for idx, camera_name in enumerate(cameras):
                rgb_resp = image_datas[2 * idx]
                image = np.frombuffer(rgb_resp.image_data_uint8,
                                      dtype=np.uint8).reshape(
                                          rgb_resp.height, rgb_resp.width, 3)
                depth_resp = image_datas[2 * idx + 1]
                depth_img_in_meters = airsim.list_to_2d_float_array(
                    depth_resp.image_data_float, depth_resp.width,
                    depth_resp.height)
                depth_image = (np.clip(depth_img_in_meters, 0, 100) / 100 *
                               255).astype(np.uint8)
                images.append(image)
                depth_images.append(depth_image)
            if use_bev:
                bev_request = []
                bev_request.append(
                    airsim.ImageRequest('BEVCamera',
                                        airsim.ImageType.Scene,
                                        pixels_as_float=False,
                                        compress=False))
                bev_request.append(
                    airsim.ImageRequest('BEVCamera',
                                        airsim.ImageType.DepthPlanar,
                                        pixels_as_float=True,
                                        compress=False))
                bev_datas = airsim_client.simGetImages(vehicle_name='Drone_2',
                                                       requests=bev_request)
                bev_image = bev_datas[0]
                bev_image = np.frombuffer(bev_image.image_data_uint8,
                                          dtype=np.uint8).reshape(
                                              bev_image.height,
                                              bev_image.width, 3)
                bev_depth = bev_datas[1]
                bev_depth_in_meters = airsim.list_to_2d_float_array(
                    bev_depth.image_data_float, bev_depth.width,
                    bev_depth.height)
                bev_depth = bev_depth_in_meters.astype(np.uint8)
            if use_bev:
                return images, depth_images, bev_image, bev_depth
            return images, depth_images

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_getImages,
                             (self.airsim_clients[index_1][index_2], )))
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
                # threads[index_1][index_2].join() # debug
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result())
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            print('getImageResponses失败')
            return None

        return responses

    def getSensorInfo(self):

        def get_sensor_info(airsim_client: airsim.VehicleClient, ):
            state_sensor = State(airsim_client, 'Drone_1')
            imu_sensor = Imu(airsim_client, 'Drone_1')
            state_sensor2 = State(airsim_client, 'Drone_2')
            state_info = state_sensor.retrieve()
            state_info2 = state_sensor2.retrieve()
            imu_info = imu_sensor.retrieve()
            return {
                'sensors': {
                    'state': state_info,
                    'state2': state_info2,
                    'imu': imu_info
                }
            }

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(get_sensor_info,
                             (self.airsim_clients[index_1][index_2], )))
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        results = []
        for index_1, _ in enumerate(threads):
            results.append([])
            for index_2, _ in enumerate(threads[index_1]):
                results[index_1].append(threads[index_1][index_2].get_result())
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            print('getSensorInfo failed.')
            return None
        return results
