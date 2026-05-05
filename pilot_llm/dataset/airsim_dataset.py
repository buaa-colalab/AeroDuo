import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from .orthography import Orthophoto
from .vis_data import visualize_waypoints
from .vision_process import process_vision_info


class AirSimDataset(torch.utils.data.Dataset):
    img_size = 784

    def __init__(
        self,
        data_list_json_paths=[],
        video_frame_num=5,
        target_interval=30,
        visualize=False,
        sigma=20,
        visual_prompt=False,
    ):
        data_list = []
        for file in data_list_json_paths:
            with open(file, "r") as f:
                data = json.load(f)
            data_list += data
        self.data = data_list

        self.video_frame_num = video_frame_num
        self.target_interval = target_interval
        self.visualize = visualize
        self.visual_prompt = visual_prompt
        self.sigma = sigma

        self.ortho_processor = Orthophoto(granularity=0.3)

        print("dataset has {} samples load from file {}".format(
            len(self.data),
            data_list_json_paths,
        ))

    def __len__(self):
        return len(self.data)

    def preprocess(self, image, pad_color=(0, 0, 0)):
        h, w = image.shape[:2]
        scale = self.img_size * 1.0 / max(h, w)
        new_h, new_w = h * scale, w * scale
        new_w = int(new_w + 0.5)
        new_h = int(new_h + 0.5)
        resized_image = cv2.resize(image, (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)
        resized_hw = (new_h, new_w)

        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        padded_image = cv2.copyMakeBorder(resized_image,
                                          0,
                                          pad_h,
                                          0,
                                          pad_w,
                                          cv2.BORDER_CONSTANT,
                                          value=pad_color)
        return padded_image, resized_hw

    def generate_prob_message_v2(self, pil_image, description):
        if isinstance(pil_image, np.ndarray):
            pil_image = Image.fromarray(pil_image)
        if isinstance(description, list):
            description = description[0]
        text_parts = description.split(
            "The description of the target and its surrounding is shown below."
        )
        direction = text_parts[0].strip().split(
            "Compass north corresponds to the top of the bird's-eye-view image."
        )[-1]
        direction = direction.strip()
        object_description = text_parts[-1].strip()

        prob_message = [{
            'role':
            'user',
            'content': [{
                'type': 'image',
                'image': pil_image
            }, {
                'type':
                'text',
                'text':
                "Task: Predict the probability distribution of the drone's future flight locations to search for the target."
                "Input Image: The image is an orthophoto map generated from the drone's past flight trajectory."
            }, {
                'type':
                'text',
                'text':
                "The green dots indicate past drone positions."
                if self.visual_prompt else " "
            }, {
                'type':
                'text',
                'text':
                "The top of the image corresponds to the north in the world coordinate system.\n"
                f"Target Information: {direction}"
                f"The description of the target and its surrounding is shown below: {object_description}\n"
                "Objective: "
                "Analyze the provided orthophoto map and target information."
                "Predict the next flight location for the drone that maximize the probability of finding the target."
                "Output a probability map, indicating the likelihood of different regions in the orthophoto map being the optimal next flight destinations."
            }]
        }, {
            'role': 'assistant',
            'content': [{
                'type': 'image',
                'image': pil_image
            }]
        }]
        return prob_message

    def gaussian(self, target, img_size):
        h, w = img_size
        prob_map = np.zeros((h, w), dtype=np.float32)
        i, j = target
        i, j = min(round(i), h - 1), min(round(j), w - 1)
        prob_map[i, j] = 1
        i_1 = min(h - 1, i + 1)
        j_1 = min(w - 1, j + 1)
        i_2 = max(0, i - 1)
        j_2 = max(0, j - 1)
        prob_map[i, j_1] = 1
        prob_map[i, j_2] = 1
        prob_map[i_1, j] = 1
        prob_map[i_2, j] = 1

        sigma = max(h, w) // 25 if self.sigma is None else self.sigma
        prob_map = gaussian_filter(prob_map, sigma=sigma)
        return prob_map

    def get_prob_map(self,
                     ortho,
                     coor_map,
                     end,
                     ortho_depth=None,
                     delta_height=None):
        h, w = ortho.shape[:2]
        if ortho_depth is not None:
            depth_mask = (ortho_depth > delta_height).reshape(h, w)
        i, j = self.ortho_processor.world_to_pixel(end, coor_map=coor_map)
        i, j = min(max(round(i), 0), h - 1), min(max(round(j), 0), w - 1)
        prob_map_0 = np.zeros((h, w))
        prob_map_0[i, j] = 1

        yy, xx = np.ogrid[:h, :w]
        distances = np.sqrt((yy - i)**2 + (xx - j)**2)
        max_dist = np.sqrt(2 * (h - 1)**2)
        prob_map_0 = 1 - distances / max_dist
        prob_map_0[~depth_mask] = 0

        new_i, new_j = np.unravel_index(np.argmax(prob_map_0), (h, w))
        prob_map = self.gaussian((new_i, new_j), (h, w))
        # prob_map[~depth_mask] = 0
        prob_map = prob_map / (np.max(prob_map) + 1e-6)

        return prob_map, depth_mask

    def get_batch(self, idx):
        data_info = self.data[idx]
        traj_dir = data_info["traj_folder_path"]
        depth_dir = os.path.join(traj_dir, "bevcamera_depth")
        image_dir = os.path.join(traj_dir, "bevcamera")
        log_dir = os.path.join(traj_dir, "log")
        image_path = data_info["image_path"]

        high_uav_pos_now = data_info["high_uav_pos_now"]
        end_pos = data_info["end_pos"]
        int_time = data_info["int_time"]
        target_time = int_time + self.target_interval

        description_path = os.path.join(traj_dir,
                                        "object_description_with_help.json")
        with open(description_path, 'r') as f:
            description = json.load(f)
        description = description[0]

        image_files = sorted([f for f in os.listdir(image_dir)])
        image_numbers = sorted([int(f.split('.')[0]) for f in image_files])
        available_images = [t for t in image_numbers if t <= int_time]
        available_num = len(available_images)
        if available_num > self.video_frame_num:
            indices = [
                round(i * (available_num - 1) / (self.video_frame_num - 1))
                for i in range(self.video_frame_num)
            ]
            available_images = [available_images[i] for i in indices]
        names = [f"{t:06d}" for t in available_images]

        # historial orthography
        frame_paths = [os.path.join(image_dir, f"{idx}.png") for idx in names]
        log_paths = [os.path.join(log_dir, f"{idx}.json") for idx in names]
        depth_paths = [os.path.join(depth_dir, f"{idx}.png") for idx in names]
        positions = np.array([
            json.load(open(log_path, "r"))["sensors"]["state"]["position"]
            for log_path in log_paths
        ])
        frames = np.array(
            [cv2.imread(frame_path) for frame_path in frame_paths])
        depths = np.array([
            cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            for depth_path in depth_paths
        ])

        coord_3d_clouds = self.ortho_processor.project_images_to_3d(
            depths, positions)
        merged_ortho, coor_map, merged_depth = self.ortho_processor.orthorectify(
            frames, coord_3d_clouds, depths)
        merged_ortho = cv2.cvtColor(merged_ortho, cv2.COLOR_BGR2RGB)

        # get prob_map gt
        with open(os.path.join(traj_dir, "gt_waypoints.json"), "r") as f:
            gt_waypoints = json.load(f)
        if len(gt_waypoints) > len(image_numbers):
            indices = [
                round(i * (len(gt_waypoints) - 1) / (len(image_numbers) - 1))
                for i in range(len(image_numbers))
            ]
        else:
            indices = [i for i in range(len(gt_waypoints))]
        index = image_numbers.index(int_time)
        time_now = indices[index]
        time_target = time_now + self.target_interval
        waypoint_now = gt_waypoints[time_now]
        if time_target < len(gt_waypoints):
            waypoint_target = gt_waypoints[time_target]
        else:
            waypoint_target = end_pos
        if self.visual_prompt:
            time_indexs = indices[:index]
            vis_waypoints = [gt_waypoints[n] for n in time_indexs]
            merged_ortho_prob = visualize_waypoints(vis_waypoints, coor_map,
                                                    merged_ortho)
        else:
            merged_ortho_prob = merged_ortho
        prob_map, depth_mask = self.get_prob_map(merged_ortho,
                                                 coor_map,
                                                 waypoint_target,
                                                 ortho_depth=merged_depth,
                                                 delta_height=waypoint_now[2] -
                                                 high_uav_pos_now[2])
        ortho_resize_pad, resized_hw = self.preprocess(merged_ortho_prob)
        prob_map, _ = self.preprocess(prob_map)
        depth_mask = depth_mask.astype(np.uint8)
        depth_mask, _ = self.preprocess(depth_mask)
        prob_message = self.generate_prob_message_v2(ortho_resize_pad,
                                                     description)

        frame = np.array(
            [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)])
        depth = np.array([
            cv2.imread(image_path.replace("bevcamera", "bevcamera_depth"),
                       cv2.IMREAD_UNCHANGED)
        ])
        coord_3d_cloud = self.ortho_processor.project_images_to_3d(
            depth, np.array([high_uav_pos_now]))
        ortho_now, _, _ = self.ortho_processor.orthorectify(
            frame, coord_3d_cloud, depth)

        prob_map = torch.from_numpy(prob_map)
        prob_map = prob_map.unsqueeze(0)

        return {
            "prob_message": prob_message,
            "prob_map": prob_map.float(),
            "target_time": target_time,
            "traj_dir": traj_dir,
        }

    def __getitem__(self, idx):
        n = self.__len__()
        cur = idx
        for _ in range(min(n, 50)):
            try:
                return self.get_batch(cur)
            except Exception as e:
                print(f"[AirSimDataset] skip idx={cur}: {e}")
                cur = int(np.random.randint(0, n))
        raise RuntimeError("AirSimDataset: too many bad samples")


def find_assistant_content_sublist_indexes(l):
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 1):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i + 1] == 77091 and l[i + 2] == 198:
            start_indexes.append(i + 3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i + 3, len(l) - 1):
                if l[j] == 151645 and l[j + 1] == 198:
                    end_indexes.append(j + 2)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


def collate_fn(batch, processor=None, mission="random"):

    masks_list = []
    messages_list = []
    traj_folders = []
    target_times = []

    for data in batch:
        masks_list.append(data["prob_map"])
        message = data["prob_message"]
        messages_list.append(message)
        traj_folders.append(data["traj_dir"])
        target_times.append(data["target_time"])

    texts = [
        processor.apply_chat_template(msg,
                                      tokenize=False,
                                      add_generation_prompt=False)
        for msg in messages_list
    ]
    image_inputs, video_inputs = process_vision_info(messages_list)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages_list) == len(input_ids_lists)

    labels = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(
            ids_list)  # -100 is the ignore index in loss function
        for begin_end_indexs in find_assistant_content_sublist_indexes(
                ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[
                begin_end_indexs[0]:begin_end_indexs[1]]
        labels.append(label_ids)
    labels = torch.tensor(labels, dtype=torch.int64)

    return inputs, {
        "masks_list": masks_list,
        "labels": labels,
        "mission": mission,
        "image_inputs": image_inputs,
        "traj_folders": traj_folders,
        "pred_time": target_times,
    }
