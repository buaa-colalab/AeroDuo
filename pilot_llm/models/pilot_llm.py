from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch
from .modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        

class MaskHead(nn.Module):
    def __init__(self, hidden_size, upsample_mode='bilinear'):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.hidden_size = hidden_size
        self.upsample_mode = upsample_mode

    def forward(self, mask_hidden_states, thw):
        bs, image_token_num, hidden_size = mask_hidden_states.shape
        t, patch_h, patch_w = thw
        h = w = image_token_num
        mask_hidden_states = self.mlp(mask_hidden_states)
        mask_hidden_states = mask_hidden_states.view(bs, 1, patch_h//2, patch_w//2)
        mask_hidden_states = F.interpolate(mask_hidden_states, size=(patch_h, patch_w), 
                                           mode=self.upsample_mode, align_corners=False)
        pred_masks = F.interpolate(mask_hidden_states, size=(h, w), 
                                   mode=self.upsample_mode, align_corners=False)
        return pred_masks


class PilotLLM(Qwen2VLForConditionalGeneration):

    def __init__(self, config, **kwargs):
        super(PilotLLM, self).__init__(config)

        self.num_image_token = kwargs.get("num_image_token", 784)
        self.seg_head = MaskHead(config.hidden_size)
        self.depth_head = MaskHead(config.hidden_size)
        self.seg_query = nn.Parameter(
            torch.zeros([self.num_image_token , self.config.hidden_size]))
        self.ce_loss = nn.CrossEntropyLoss() # 自带sigmoid激活函数
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        masks_list: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_inference: bool = False,
        **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bs = input_ids.shape[0]
        labels = labels.to(input_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)


            special_mask = (input_ids == self.config.image_token_id) & (labels == self.config.image_token_id)
            for i in range(bs):
                mask_indices = special_mask[i].nonzero(as_tuple=True)[0]
                inputs_embeds[i, mask_indices] = self.seg_query
                
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore.
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state

        seg_query = hidden_states[special_mask].view(bs, self.num_image_token, self.config.hidden_size)
        preds = self.seg_head(seg_query, image_grid_thw[-1])

        if is_inference:
            return preds
        
        bce_loss = 0
        num_masks = 0
        for pred_mask, gt_mask in zip(preds, masks_list):
            pred_mask = pred_mask[:, : gt_mask.shape[1], : gt_mask.shape[2]]
            gt_mask = gt_mask.to(pred_mask.device)
            assert (
                gt_mask.shape == pred_mask.shape
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            bce_loss += self.bce_loss(pred_mask, gt_mask)
            num_masks += gt_mask.shape[0]
        
        loss = bce_loss / (num_masks + 1e-8)

        return preds, loss