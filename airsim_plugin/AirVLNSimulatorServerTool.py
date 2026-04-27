import argparse
import copy
import errno
import glob
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import msgpackrpc

AIRSIM_SETTINGS_TEMPLATE = {
    "SeeDocsAt": "https://microsoft.github.io/AirSim/settings/",
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "ClockSpeed": 10,
    "ViewMode": "NoDisplay",
    "PhysiceEngineName": "ExternalPhysicsEngine",
    "Recording": {
        "RecordInterval": 1,
        "Enabled": False,
        "Cameras": []
    },
    "Vehicles": {
        "Drone_1": {
            "VehicleType": "SimpleFlight",
            "UseSerial": False,
            "LockStep": True,
            "AutoCreate": True,
            "X": 0,
            "Y": 0,
            "Z": 0,
            "Roll": 0,
            "Pitch": 0,
            "Yaw": 0,
            "Cameras": {
                "FrontCamera": {
                    "X":
                    1,
                    "Y":
                    0,
                    "Z":
                    0,
                    "Pitch":
                    0,
                    "Roll":
                    0,
                    "Yaw":
                    0,
                    "CaptureSettings": [{
                        "ImageType": 0,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }, {
                        "ImageType": 2,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }]
                },
                "RearCamera": {
                    "X":
                    -1,
                    "Y":
                    0,
                    "Z":
                    0,
                    "Pitch":
                    0,
                    "Roll":
                    0,
                    "Yaw":
                    180,
                    "CaptureSettings": [{
                        "ImageType": 0,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }, {
                        "ImageType": 2,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }]
                },
                "LeftCamera": {
                    "X":
                    0,
                    "Y":
                    -1,
                    "Z":
                    0,
                    "Pitch":
                    0,
                    "Roll":
                    0,
                    "Yaw":
                    -90,
                    "CaptureSettings": [{
                        "ImageType": 0,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }, {
                        "ImageType": 2,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }]
                },
                "RightCamera": {
                    "X":
                    0,
                    "Y":
                    1,
                    "Z":
                    0,
                    "Pitch":
                    0,
                    "Roll":
                    0,
                    "Yaw":
                    90,
                    "CaptureSettings": [{
                        "ImageType": 0,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }, {
                        "ImageType": 2,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }]
                },
                "DownCamera": {
                    "X":
                    0,
                    "Y":
                    0,
                    "Z":
                    0,
                    "Pitch":
                    -90,
                    "Roll":
                    0,
                    "Yaw":
                    0,
                    "CaptureSettings": [{
                        "ImageType": 0,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }, {
                        "ImageType": 2,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }]
                }
            },
            "Sensors": {
                "Imu": {
                    "SensorType": 2,
                    "Enabled": True,
                    "AngularRandomWalk": 0.3,
                    "GyroBiasStabilityTau": 500,
                    "GyroBiasStability": 4.6,
                    "VelocityRandomWalk": 0.24,
                    "AccelBiasStabilityTau": 800,
                    "AccelBiasStability": 36
                }
            }
        }
    }
}

AIRSIM_SETTINGS_TEMPLATE_2UAV = {
    "SeeDocsAt": "https://microsoft.github.io/AirSim/settings/",
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "ClockSpeed": 10,
    "ViewMode": "NoDisplay",
    "PhysiceEngineName": "ExternalPhysicsEngine",
    "Recording": {
        "RecordInterval": 1,
        "Enabled": False,
        "Cameras": []
    },
    "Vehicles": {
        "Drone_1": {
            "VehicleType": "SimpleFlight",
            "UseSerial": False,
            "LockStep": True,
            "AutoCreate": True,
            "X": 0,
            "Y": 0,
            "Z": 0,
            "Roll": 0,
            "Pitch": 0,
            "Yaw": 0,
            "Cameras": {
                "FrontCamera": {
                    "X":
                    1,
                    "Y":
                    0,
                    "Z":
                    0,
                    "Pitch":
                    0,
                    "Roll":
                    0,
                    "Yaw":
                    0,
                    "CaptureSettings": [{
                        "ImageType": 0,
                        "Width": 1024,
                        "Height": 1024,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }, {
                        "ImageType": 2,
                        "Width": 1024,
                        "Height": 1024,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }]
                }
            },
            "Sensors": {
                "Lidar": {
                    "SensorType": 6,
                    "Enabled": True,
                    "NumberOfChannels": 4,
                    "RotationsPerSecond": 10,
                    "PointsPerSecond": 1440,
                    "Range": 4,
                    "X": 0,
                    "Y": 0,
                    "Z": -1,
                    "Roll": 0,
                    "Pitch": 90,
                    "Yaw": 0,
                    "VerticalFOVUpper": 20,
                    "VerticalFOVLower": -10,
                    "HorizontalFOVStart": -180,
                    "HorizontalFOVEnd": 180,
                    "DrawDebugPoints": True,
                    "DataFrame": "SensorLocalFrame"
                },
                "Imu": {
                    "SensorType": 2,
                    "Enabled": True,
                    "AngularRandomWalk": 0.3,
                    "GyroBiasStabilityTau": 500,
                    "GyroBiasStability": 4.6,
                    "VelocityRandomWalk": 0.24,
                    "AccelBiasStabilityTau": 800,
                    "AccelBiasStability": 36
                }
            }
        },
        "Drone_2": {
            "VehicleType": "SimpleFlight",
            "UseSerial": False,
            "LockStep": True,
            "AutoCreate": True,
            "X": 0,
            "Y": 0,
            "Z": 0,
            "Roll": 0,
            "Pitch": 0,
            "Yaw": 0,
            "Cameras": {
                "FrontCamera": {
                    "X":
                    1,
                    "Y":
                    0,
                    "Z":
                    0,
                    "Pitch":
                    0,
                    "Roll":
                    0,
                    "Yaw":
                    0,
                    "CaptureSettings": [{
                        "ImageType": 0,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }, {
                        "ImageType": 2,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }, {
                        "ImageType": 5,
                        "Width": 256,
                        "Height": 256,
                        "FOV_Degrees": 90,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }]
                },
                "BEVCamera": {
                    "X":
                    0,
                    "Y":
                    0,
                    "Z":
                    0,
                    "Pitch":
                    -90,
                    "Roll":
                    0,
                    "Yaw":
                    0,
                    "CaptureSettings": [{
                        "ImageType": 0,
                        "Width": 1024,
                        "Height": 1024,
                        "FOV_Degrees": 120,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }, {
                        "ImageType": 1,
                        "Width": 1024,
                        "Height": 1024,
                        "FOV_Degrees": 120,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }, {
                        "ImageType": 5,
                        "Width": 1024,
                        "Height": 1024,
                        "FOV_Degrees": 120,
                        "AutoExposureMaxBrightness": 1,
                        "AutoExposureMinBrightness": 0.03
                    }],
                    "Gimbal": {
                        "Stabilization": 1,
                        "Pitch": -90,
                        "Roll": 0,
                        "Yaw": 0
                    }
                }
            },
            "Sensors": {
                "Imu": {
                    "SensorType": 2,
                    "Enabled": True,
                    "AngularRandomWalk": 0.3,
                    "GyroBiasStabilityTau": 500,
                    "GyroBiasStability": 4.6,
                    "VelocityRandomWalk": 0.24,
                    "AccelBiasStabilityTau": 800,
                    "AccelBiasStability": 36
                }
            }
        }
    }
}

known_env_dict = {
    "NYC_dev": {
        'bash_name': 'NYCEnvironmentMegapa',
        'exec_path': '/nfs/airport/airdrone/envs',
    },
    "NYCEnvironmentMegapa": {
        'bash_name': 'NYCEnvironmentMegapa',
        'exec_path': '/nfs/airport/airdrone/envs',
    },
    "TropicalIsland": {
        'bash_name': 'TropicalIsland',
        'exec_path': '/nfs/airport/airdrone/envs',
    },
    "NewYorkCity": {
        'bash_name': 'NewYorkCity',
        'exec_path': '/nfs/airport/airdrone/envs',
    },
    "ModularPark": {
        'bash_name': 'ModularPark',
        'exec_path': '/nfs/airport/airdrone/envs',
    },
    "ModularEuropean": {
        'bash_name': 'ModularEuropean',
        'exec_path': '/nfs/airport/airdrone/envs',
    },
    "Carla_Town01": {
        'bash_name': 'CarlaUE4',
        'exec_path':
        '/nfs/airport/airdrone/carla_town_envs/Town01/LinuxNoEditor',
    },
    "Carla_Town02": {
        'bash_name': 'CarlaUE4',
        'exec_path':
        '/nfs/airport/airdrone/carla_town_envs/Town02/LinuxNoEditor',
    },
    "Carla_Town03": {
        'bash_name': 'CarlaUE4',
        'exec_path':
        '/nfs/airport/airdrone/carla_town_envs/Town03/LinuxNoEditor',
    },
    "Carla_Town04": {
        'bash_name': 'CarlaUE4',
        'exec_path':
        '/nfs/airport/airdrone/carla_town_envs/Town04/LinuxNoEditor',
    },
    "Carla_Town05": {
        'bash_name': 'CarlaUE4',
        'exec_path':
        '/nfs/airport/airdrone/carla_town_envs/Town05/LinuxNoEditor',
    },
    "Carla_Town06": {
        'bash_name': 'CarlaUE4',
        'exec_path':
        '/nfs/airport/airdrone/carla_town_envs/Town06/LinuxNoEditor',
    },
    "Carla_Town07": {
        'bash_name': 'CarlaUE4',
        'exec_path':
        '/nfs/airport/airdrone/carla_town_envs/Town07/LinuxNoEditor',
    },
    "Carla_Town10HD": {
        'bash_name':
        'CarlaUE4',
        'exec_path':
        '/nfs/airport/airdrone/carla_town_envs/Town10HD/LinuxNoEditor',
    },
    "Carla_Town15": {
        'bash_name': 'CarlaUE4',
        'exec_path':
        '/nfs/airport/airdrone/carla_town_envs/Town15/LinuxNoEditor',
    },
}


def create_drones(drone_num_per_env=1,
                  show_scene=False,
                  uav_mode=True) -> dict:
    if NUM_UAV == 1:
        airsim_settings = copy.deepcopy(AIRSIM_SETTINGS_TEMPLATE)
    elif NUM_UAV == 2:
        airsim_settings = copy.deepcopy(AIRSIM_SETTINGS_TEMPLATE_2UAV)
    airsim_settings = generate_distance_sensor_setting(airsim_settings)
    airsim_settings = generate_camera_setting(airsim_settings)
    return airsim_settings


def generate_distance_sensor_setting(airsim_config):
    v_angles = [-10, 0, 10, 20]
    h_angles = list(range(0, 360, 10))

    for v in v_angles:
        for h in h_angles:
            sensor_name = f"DistanceSensor_v{v}_h{h}"
            airsim_config["Vehicles"]["Drone_1"]["Sensors"][sensor_name] = {
                "SensorType": 5,  # Distance Sensor类型
                "Enabled": True,
                "Range": 4.0,  # 最大探测范围
                "X": 0,
                "Y": 0,
                "Z": 0,
                "Roll": 0,
                "Pitch": v,  # 垂直角度
                "Yaw": h,  # 水平角度
                "DataFrame": "SensorLocalFrame"
            }

    return airsim_config


def generate_camera_setting(airsim_config):
    yaw_list = list(range(0, 360, 10))
    for yaw in yaw_list:
        x, y = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        sensor_name = f"Camera_{yaw}"
        airsim_config["Vehicles"]["Drone_1"]["Cameras"][sensor_name] = {
            "X":
            x,
            "Y":
            y,
            "Z":
            0,
            "Pitch":
            0,
            "Roll":
            0,
            "Yaw":
            yaw,
            "CaptureSettings": [{
                "ImageType": 0,
                "Width": 1024,
                "Height": 1024,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03
            }, {
                "ImageType": 2,
                "Width": 1024,
                "Height": 1024,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03
            }]
        }

    return airsim_config


def pid_exists(pid) -> bool:
    """
    Check whether pid exists in the current process table.
    UNIX only.
    """
    if pid < 0:
        return False

    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True


def FromPortGetPid(port: int):
    subprocess_execute = "netstat -nlp | grep {}".format(port, )

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print("{}\t{}\t{}".format(
            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            'FromPortGetPid',
            e,
        ))
        return None
    except:
        return None

    pid = None
    for line in iter(p.stdout.readline, b''):
        # import pdb;pdb.set_trace()
        line = str(line, encoding="utf-8")
        if 'tcp' in line:
            pid = line.strip().split()[-1].split('/')[0]
            try:
                pid = int(pid)
            except:
                pid = None
            break

    try:
        # os.system(("kill -9 {}".format(p.pid)))
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    return pid


def KillPid(pid) -> None:
    if pid is None or not isinstance(pid, int):
        # print('pid is not int')
        return

    while pid_exists(pid):
        try:
            # os.system(("kill -9 {}".format(pid)))
            print('pid {} is killed'.format(pid))
            os.kill(pid, signal.SIGKILL)
        except Exception as e:
            pass
        time.sleep(0.5)

    return


def KillPorts(ports) -> None:
    threads = []

    def _kill_port(index, port):
        pid = FromPortGetPid(port)
        KillPid(pid)

    for index, port in enumerate(ports):
        thread = threading.Thread(target=_kill_port,
                                  args=(index, port),
                                  daemon=True)
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    threads = []

    return


def KillAirVLN() -> None:
    subprocess_execute = "pkill -9 AirVLN"

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print("{}\t{}\t{}".format(
            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            'KillAirVLN',
            e,
        ))
        return
    except:
        return

    try:
        # os.system(("kill -9 {}".format(p.pid)))
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    time.sleep(1)
    return


class EventHandler(object):

    def __init__(self):
        scene_ports = []
        for i in range(1000):
            scene_ports.append(int(args.port) + (i + 1))
        self.scene_ports = scene_ports

        scene_gpus = []
        while len(scene_gpus) < 100:
            scene_gpus += GPU_IDS.copy()
        self.scene_gpus = scene_gpus

        self.scene_used_ports = []

        self.port_to_scene = {}

    def ping(self) -> bool:
        return True

    def _open_scenes(self, ip: str, scen_id_gpu_list: list):
        print("{}\t关闭场景中".format(
            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), ))
        KillPorts(self.scene_used_ports)
        self.scene_used_ports = []
        # KillAirVLN()
        print("{}\t已关闭所有场景".format(
            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), ))

        # Occupied airsim port 1
        ports = []
        index = 0
        while len(ports) < len(scen_id_gpu_list):
            pid = FromPortGetPid(self.scene_ports[index])
            if pid is None or not isinstance(pid, int):
                ports.append(self.scene_ports[index])
            index += 1

        KillPorts(ports)

        # Occupied GPU 2
        gpus = [
            scen_id_gpu_list[index][-1]
            for index in range(len(scen_id_gpu_list))
        ]
        print(scen_id_gpu_list)

        # search scene path 3
        choose_env_exe_paths = []
        for scen_id, gpu_id in scen_id_gpu_list:
            if str(scen_id).lower() == 'none':
                choose_env_exe_paths.append(None)
                continue
            if 'Carla' in scen_id:
                idx = scen_id.split('Town')[-1]
                SEARCH_ENVs_PATH = Path(
                    f'envs/carla_town_envs/Town{idx}/LinuxNoEditor')
                res = glob.glob((str(SEARCH_ENVs_PATH / 'CarlaUE4.sh')))
            else:
                SEARCH_ENVs_PATH = Path('envs/closeloop_envs')
                res = glob.glob((str(SEARCH_ENVs_PATH / (scen_id + '.sh'))))
            print(str(SEARCH_ENVs_PATH / (scen_id + '.sh')))
            if len(res) > 0:
                choose_env_exe_paths.append(res[0])
            elif scen_id in known_env_dict:
                env_info = known_env_dict.get(scen_id)
                res = os.path.join(env_info['exec_path'],
                                   env_info['bash_name'] + '.sh')
                choose_env_exe_paths.append(res)
            else:
                prefix_flag = False
                for map_name in known_env_dict.keys():
                    if str(scen_id).startswith(map_name):
                        prefix_flag = True
                        env_info = known_env_dict.get(map_name)
                        res = os.path.join(env_info['exec_path'],
                                           env_info['bash_name'] + '.sh')
                        choose_env_exe_paths.append(res)
                if not prefix_flag:
                    print(f'can not find sCene file: {scen_id}')
                    raise KeyError

        p_s = []
        for index, (scen_id, gpu_id) in enumerate(scen_id_gpu_list):
            # airsim settings 4
            airsim_settings = create_drones()
            # print(airsim_settings['SimMode'])
            airsim_settings['ApiServerPort'] = int(ports[index])
            self.port_to_scene[ports[index]] = (scen_id, gpu_id)
            airsim_settings_write_content = json.dumps(airsim_settings)
            if not os.path.exists(
                    str(CWD_DIR / 'airsim_plugin/settings' /
                        str(ports[index]))):
                os.makedirs(str(CWD_DIR / 'airsim_plugin/settings' /
                                str(ports[index])),
                            exist_ok=True)
            with open(str(CWD_DIR / 'airsim_plugin/settings' /
                          str(ports[index]) / 'settings.json'),
                      'w',
                      encoding='utf-8') as dump_f:
                dump_f.write(airsim_settings_write_content)

            # open scene 5
            if choose_env_exe_paths[index] is None:
                p_s.append(None)
                continue
            else:
                subprocess_execute = "bash {} -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter={} -settings={} ".format(
                    choose_env_exe_paths[index],
                    gpu_id,
                    str(CWD_DIR / 'airsim_plugin/settings' /
                        str(ports[index]) / 'settings.json'),
                )
                time.sleep(1)
                print(subprocess_execute)

                try:
                    p = subprocess.Popen(
                        subprocess_execute,
                        stdin=None,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        shell=True,
                    )
                    p_s.append(p)
                except Exception as e:
                    print("{}\t{}".format(
                        str(
                            time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime())),
                        e,
                    ))
                    return False, None
                except:
                    return False, None
        time.sleep(10)
        # ChangeNice(ports)
        self.scene_used_ports += copy.deepcopy(ports)

        print("finished", ip)

        return True, (ip, ports)

    def reopen_scene_from_port(self, port):

        KillPorts([port])

        scene_id, gpu_id = self.port_to_scene[port]
        res = glob.glob((str(SEARCH_ENVs_PATH / (scene_id + '.sh'))))
        env_path = res[0]

        subprocess_execute = "bash {} -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter={} -settings={} ".format(
            env_path,
            gpu_id,
            str(CWD_DIR / 'airsim_plugin/settings' / str(port) /
                'settings.json'),
        )
        time.sleep(1)
        print(subprocess_execute)

        p = subprocess.Popen(
            subprocess_execute,
            stdin=None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            shell=True,
        )

    def reopen_scenes(self, ip: str, scen_id_gpu_list: list):
        print("{}\tSTART reopen_scenes".format(
            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), ))
        try:
            print(scen_id_gpu_list)
            ip = ip
            for item in scen_id_gpu_list:
                # print(item)
                # TODO: don't know why item is int
                if type(item[0]) is not str:
                    item[0] = item[0].decode('utf-8')
            result = self._open_scenes(ip, scen_id_gpu_list)
        except Exception as e:
            print(e)
            exe_type, exe_value, exe_traceback = sys.exc_info()
            exe_info_list = traceback.format_exception(exe_type, exe_value,
                                                       exe_traceback)
            tracebacks = ''.join(exe_info_list)
            print('traceback:', tracebacks)
            result = False, None
        print("{}\tEND reopen_scenes".format(
            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), ))
        return result

    def close_scenes(self, ip: str) -> bool:
        print("{}\tSTART close_scenes".format(
            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), ))

        try:
            KillPorts(self.scene_used_ports)
            self.scene_used_ports = []
            # KillPorts(self.scene_ports)
            # KillAirVLN()

            result = True
        except Exception as e:
            print(e)
            result = False

        print("{}\tEND close_scenes".format(
            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), ))
        return result


def serve_background(server, daemon=False):

    def _start_server(server):
        server.start()
        server.close()

    t = threading.Thread(target=_start_server, args=(server, ))
    t.setDaemon(daemon)
    t.start()
    return t


def serve(daemon=False):
    try:
        server = msgpackrpc.Server(EventHandler())
        addr = msgpackrpc.Address(HOST, PORT)
        server.listen(addr)

        thread = serve_background(server, daemon)

        return addr, server, thread
    except Exception as err:
        print("error", err)
        pass


if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=str,
        default='3',
    )
    parser.add_argument("--port", type=int, default=50500, help='server port')
    parser.add_argument("--num_uav",
                        type=int,
                        default=2,
                        help="the number of UAVs")

    args = parser.parse_args()

    HOST = '127.0.0.1'
    PORT = int(args.port)
    CWD_DIR = Path(str(os.getcwd())).resolve()
    PROJECT_ROOT_DIR = CWD_DIR.parent.parent.parent
    print("PROJECT_ROOT_DIR", PROJECT_ROOT_DIR)
    # SEARCH_ENVs_PATH = PROJECT_ROOT_DIR / 'envs/anew_test/ghx/BrushifyCountryRoads_612_zichan/'  # TODO
    SEARCH_ENVs_PATH = Path('envs/closeloop_envs')
    assert os.path.exists(str(SEARCH_ENVs_PATH)), 'error'

    gpu_list = []
    gpus = str(args.gpus).split(',')
    for gpu in gpus:
        gpu_list.append(int(gpu.strip()))
    GPU_IDS = gpu_list.copy()

    NUM_UAV = args.num_uav

    addr, server, thread = serve()
    print(f"start listening \t{addr._host}:{addr._port}")
