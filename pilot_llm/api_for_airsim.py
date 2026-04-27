import math
import os

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor

from .A_star import A_star
from .models.pilot_llm import PilotLLM
from .orthography import Orthophoto
from .vision_process import process_vision_info

BASE_MODEL_PATH = "pilot_llm/weights/Qwen2-VL-2B-Instruct"
IMAGE_SIZE = 784


class CallPilot:

    def __init__(self,
                 checkpoint_path="pilot_llm/weights/AeroDuo-PilotLLM",
                 device=None,
                 use_a_star=True):
        # super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available(
        ) else "cpu") if device is None else device
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH,
                                                       padding_side="right")
        model = PilotLLM.from_pretrained(BASE_MODEL_PATH, num_token=784)
        model = PeftModel.from_pretrained(model, checkpoint_path)
        weights = torch.load(
            os.path.join(checkpoint_path,
                         "pytorch_model/mp_rank_00_model_states.pt"))
        model.load_state_dict(weights['module'], strict=False)

        self.model = model.to(device)
        self.model.eval()

        self.ortho_processor = Orthophoto()
        self.warm_up = False

        self.use_a_star = use_a_star
        self.a_star_searcher = A_star()
        self.save_index = 0

        self.img_size = 784

    def reset_ortho_granularity(self, granularity=0.6):
        self.ortho_processor = Orthophoto(granularity=granularity)

    def get_ortho_indices(self, cur_idx, max_interval=25):
        if cur_idx < 5:
            return np.arange(0, cur_idx)
        else:
            interval = min(max_interval, cur_idx // 4)
            return np.arange(cur_idx - 4 * interval - 1, cur_idx, interval)

    def get_orthophoto(self, bev_image, point_cloud, bev_depth):
        cur_idx = len(bev_image)
        indices = self.get_ortho_indices(cur_idx)
        selected_bev_image = np.array(bev_image)[indices]
        selected_point_cloud = np.array(point_cloud)[indices]
        selected_bev_depth = np.array(bev_depth)[indices]

        merged_ortho, coor_map, merged_depth = self.ortho_processor.orthorectify(
            selected_bev_image, selected_point_cloud, selected_bev_depth)
        return merged_ortho, coor_map, merged_depth

    def generate_prob_message(self, pil_image, description):
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

    def update_traj_info(self, traj_name):
        self.traj_name = traj_name
        self.visualize_cnt = 0

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

    @torch.no_grad()
    def __call__(self,
                 bev_images,
                 bev_depths,
                 point_clouds,
                 high_uav_locations,
                 low_uav_locations,
                 description,
                 history_waypoints,
                 history_waypoints2,
                 target_positions,
                 num_keypoints=5):
        """
        Args:
            bev_iamges: numpy.ndarray, shape (batch_size, 1024, 1024, 3) 注意：RGB格式，如果是cv2读取的图片需要转换
            point_clouds: numpy.ndarray, shape (batch_size, 1024, 1024, 3)
        Returns:
            pred_masks: torch.Tensor, shape (batch_size, 1, 784, 784)
            没写可视化函数, 得到的pred_masks是一个tensor, 需要sigmoid激活再后作为参考概率图
        """
        print(
            f"drone1_start: {low_uav_locations} | drone2_start: {high_uav_locations}"
        )

        history_orthophotos = []
        ortho_depths = []
        coor_maps = []
        resized_record = []
        original_shapes = []
        for bev_image, bev_depth, point_cloud in zip(bev_images, bev_depths,
                                                     point_clouds):
            ortho, coor_map, merged_depth = self.ortho_processor.orthorectify(
                np.array(bev_image), np.array(point_cloud),
                np.array(bev_depth))
            original_shapes.append(ortho.shape[:2])
            ortho_pad_resize, resized_hw = self.preprocess(ortho)
            history_orthophotos.append(ortho_pad_resize)
            ortho_depths.append(merged_depth)
            coor_maps.append(coor_map)
            resized_record.append(resized_hw)

        messages = []
        for ortho, desc in zip(history_orthophotos, description):
            ortho = cv2.cvtColor(ortho, cv2.COLOR_BGR2RGB)
            message = self.generate_prob_message(ortho, desc)
            # ortho = cv2.cvtColor(ortho, cv2.COLOR_RGB2BGR)
            messages.append(message)

        texts = [
            self.processor.apply_chat_template(msg,
                                               tokenize=False,
                                               add_generation_prompt=False)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids_lists = inputs['input_ids'].tolist()
        assert len(messages) == len(input_ids_lists)
        labels = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(
                ids_list)  # -100 is the ignore index in loss function
            for begin_end_indexs in find_assistant_content_sublist_indexes(
                    ids_list):
                label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[
                    begin_end_indexs[0]:begin_end_indexs[1]]
            labels.append(label_ids)
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)

        # TODO: check why the first output is wrong
        if not self.warm_up:
            self.model(**inputs,
                       labels=labels,
                       is_inference=True,
                       mission="prob")
            self.warm_up = True

        pred_masks = self.model(**inputs,
                                labels=labels,
                                is_inference=True,
                                mission="prob")
        pred_waypoints, pred_idx, prob_map, occupancy_map = self.postprocess_llm_output(
            pred_masks, low_uav_locations, high_uav_locations, coor_maps,
            ortho_depths, resized_record, original_shapes)
        pred_waypoints = np.array(pred_waypoints, dtype=np.float32).tolist()

        # a_star search
        if self.use_a_star:
            a_star_inputs = {
                "global_depth":
                ortho_depths[0],
                "global_coor_map":
                coor_maps[0],
                "start_world":
                low_uav_locations[0],
                "goal_world":
                pred_waypoints[0],
                "delta_altitude":
                low_uav_locations[0][2] - high_uav_locations[0][2],
            }
            waypoint_list = self.a_star_searcher.search(**a_star_inputs)[1:]
            # choose N waypoints from list
            len_waypoint_list = len(waypoint_list)
            if len_waypoint_list > num_keypoints + 1:
                interval = (len_waypoint_list - 1) // num_keypoints
                pred_idxs = [
                    waypoint_list[len_waypoint_list - 1 - i * interval]
                    for i in range(num_keypoints, -1, -1)
                ]
            else:
                pred_idxs = waypoint_list

            waypoint_list = self.a_star_searcher.from_pixel_to_3d(
                pred_idxs, low_uav_locations[0][-1], coor_maps[0])
        else:
            waypoint_list = pred_waypoints

        return waypoint_list, None

    def postprocess_llm_output(self, drone1_prob_map, low_uav_locations,
                               high_uav_locations, coor_maps, merged_depths,
                               resized_record, original_shapes):
        prob_maps = np.array(drone1_prob_map.squeeze(dim=1).cpu())
        drone1_waypoints = []
        for i, prob_map in enumerate(prob_maps):
            resized_hw = resized_record[i]
            prob_map = prob_map[:resized_hw[0], :resized_hw[1]]
            origin_shape = original_shapes[i]
            prob_map = cv2.resize(prob_map, (origin_shape[1], origin_shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            depth = merged_depths[i]
            delta_height = low_uav_locations[i][2] - high_uav_locations[i][2]
            # first obtain top 200000 of prob_map
            h, w = prob_map.shape
            prob_map = torch.sigmoid(torch.tensor(prob_map)).flatten()
            prob_map_top_k = torch.topk(prob_map, k=20000)
            new_prob_map = torch.zeros_like(prob_map)
            new_prob_map[prob_map_top_k.indices] = prob_map_top_k.values
            prob_map = new_prob_map.reshape(h, w).cpu().numpy()
            # apply the occupancy mask
            occupancy_map = (depth >= delta_height).astype(np.uint8).squeeze()
            if np.all(prob_map * occupancy_map == 0):
                print(f"Warning: all zero prob_map")
                prob_map = prob_map
            else:
                prob_map = prob_map * occupancy_map
            # use center of gravity to find the target point
            row, col = prob_map.shape
            i_indice, j_indice = np.indices((row, col))
            sum_total = np.sum(prob_map)
            c_i = int(np.sum(prob_map * i_indice) / sum_total)
            c_j = int(np.sum(prob_map * j_indice) / sum_total)
            # deal with the corner case
            c_i = min(c_i, row - 1)
            c_j = min(c_j, col - 1)
            pred_idx = (c_i, c_j)
            plane_coor = coor_maps[i][pred_idx]
            drone1_waypoint = [
                plane_coor[0], plane_coor[1], low_uav_locations[i][2]
            ]
            drone1_waypoints.append(drone1_waypoint)

        return drone1_waypoints, pred_idx, prob_map, occupancy_map


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
