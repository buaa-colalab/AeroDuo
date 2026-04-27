import json
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vlnce_src.clip_encoder import CLIPVisionTower
from src.vlnce_src.eva_vit import EVAVisionTowerLavis


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower',
                           getattr(vision_tower_cfg, 'vision_tower', None))
    image_processor = getattr(
        vision_tower_cfg, 'image_processor',
        getattr(vision_tower_cfg, 'image_processor',
                "./model_zoo/OpenAI/clip-vit-large-patch14"))
    is_absolute_path_exists = os.path.exists(vision_tower)

    if not is_absolute_path_exists:
        raise ValueError(f'Not find vision tower: {vision_tower}')

    if "openai" in vision_tower.lower() or "laion" in vision_tower.lower():
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
        return EVAVisionTowerLavis(vision_tower,
                                   image_processor,
                                   args=vision_tower_cfg,
                                   **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SelfAttention(nn.Module):

    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1)**
                                                                   0.5)
        attention_weights = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output


class CrossAttention(nn.Module):

    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key_value):
        Q = self.query(query)
        K = self.key(key_value)
        V = self.value(key_value)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1)**
                                                                   0.5)
        attention_weights = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output


class VisionTrajectoryGenerator(nn.Module):

    def __init__(self, config):
        super(VisionTrajectoryGenerator, self).__init__()
        self.config = config
        config.hidden_dim = 2048
        config.feature_dim = 1024
        self.vision_tower = build_vision_tower(config, delay_load=False)
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        self.vision_projector = MLP(1408, config.hidden_dim,
                                    config.feature_dim - 3)
        # self.waypoint_processer = MLP(3, config.hidden_dim, config.feature_dim)
        # self.fusion_module = SelfAttention(config.feature_dim)
        self.waypoint_query = nn.Parameter(torch.randn(7, config.feature_dim))
        # self.waypoint_decoder = CrossAttention(config.feature_dim)

        # self.waypoints_predictor = nn.GRUCell(input_size=3, hidden_size=1024)
        # self.waypoints_output = nn.Linear(1024, 3)
        self.waypoint_predictor = nn.Sequential(nn.Linear(1024, 256), nn.ELU(),
                                                nn.Dropout(0.1),
                                                nn.Linear(256, 21))
        # self.fc_output = MLP(config.feature_dim, config.hidden_dim, 21)
        self.waypoints_loss_func = torch.nn.L1Loss()

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def forward(self, inputs, label=None):
        images = inputs['img'].to(device=self.waypoint_query.device,
                                  dtype=self.waypoint_query.dtype)
        waypoints = inputs['target'].to(device=self.waypoint_query.device,
                                        dtype=self.waypoint_query.dtype)
        if label is not None:
            label = label.to(device=self.waypoint_query.device,
                             dtype=self.waypoint_query.dtype)

        with torch.no_grad():
            vision_features = self.vision_tower(images)

        vision_features = self.vision_projector(vision_features)[:, 1:]
        pooled_vision_features = F.avg_pool1d(vision_features.permute(0, 2, 1),
                                              256, 1).squeeze(-1)
        # waypoint_features = self.waypoint_processer(waypoints)

        combined_features = torch.cat((pooled_vision_features, waypoints),
                                      dim=1)
        # gru
        # output_wp = []
        # output_delta = []
        # waypoints_feature = combined_features
        # x = torch.zeros(size=(label.shape[0], 3), dtype=combined_features.dtype).to(combined_features.device)
        # for _ in range(7):
        #     x_in = x
        #     with torch.autocast(device_type='cuda', dtype=torch.float32):
        #         waypoints_feature = self.waypoints_predictor(x_in, waypoints_feature)
        #     dx = self.waypoints_output(waypoints_feature.to(combined_features.dtype))
        #     x = dx + x
        #     output_wp.append(x)
        #     output_delta.append(dx)
        # pred_trajectory_points = torch.cat(output_wp, dim=1)
        # predicted_deltas = torch.cat(output_delta, dim=1)
        # pred_trajectory_points = pred_trajectory_points.view(-1, 7, 3)
        # predicted_deltas = predicted_deltas.view(-1, 7, 3)

        # fused_features = self.fusion_module(combined_features)
        # decoded_features = self.waypoint_decoder(self.waypoint_query, combined_features)

        pred_trajectory_points = self.waypoint_predictor(
            combined_features).view(-1, 7, 3)
        # pred_trajectory_points = self.fc_output(combined_features)
        # pred_trajectory_points = pred_trajectory_points.view(-1, 7, 3)
        if label is None:
            return pred_trajectory_points
        loss = self.waypoints_loss_func(label, pred_trajectory_points)

        return loss, pred_trajectory_points
