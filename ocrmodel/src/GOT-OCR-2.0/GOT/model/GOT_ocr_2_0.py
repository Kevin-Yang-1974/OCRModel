from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from GOT.utils.constants import *
from GOT.model.vision_encoder.vary_b import build_vary_vit_b
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.model.layout_query import (
    VLQAOutput,
    VisualLayoutQueryAdapter,
    VisualLayoutQueryLoss,
)
from GOT.model.layout_prompt_decoder import PromptedVariableLayoutAdapter
from GOT.model.generic_adapter import GenericVisualTransformerAdapter


@dataclass
class GOTBaseModelOutputWithPast(BaseModelOutputWithPast):
    layout_loss: Optional[torch.FloatTensor] = None
    layout_object_loss: Optional[torch.FloatTensor] = None
    layout_bbox_l1_loss: Optional[torch.FloatTensor] = None
    layout_bbox_giou_loss: Optional[torch.FloatTensor] = None
    layout_direction_loss: Optional[torch.FloatTensor] = None
    layout_object_accuracy: Optional[torch.FloatTensor] = None
    layout_bbox_mean_iou: Optional[torch.FloatTensor] = None
    layout_direction_accuracy: Optional[torch.FloatTensor] = None
    layout_object_logit_abs_max: Optional[torch.FloatTensor] = None
    layout_direction_logit_abs_max: Optional[torch.FloatTensor] = None
    layout_bbox_pred_min: Optional[torch.FloatTensor] = None
    layout_bbox_pred_max: Optional[torch.FloatTensor] = None
    layout_query_abs_max: Optional[torch.FloatTensor] = None
    layout_prediction_query_abs_max: Optional[torch.FloatTensor] = None
    layout_bbox_logit_abs_max: Optional[torch.FloatTensor] = None
    layout_object_logits: Optional[torch.FloatTensor] = None
    layout_bbox_xyxy: Optional[torch.FloatTensor] = None
    layout_direction_logits: Optional[torch.FloatTensor] = None
    layout_sequence_loss: Optional[torch.FloatTensor] = None
    layout_type_loss: Optional[torch.FloatTensor] = None
    layout_count_loss: Optional[torch.FloatTensor] = None
    layout_eos_accuracy: Optional[torch.FloatTensor] = None
    layout_region_count_mae: Optional[torch.FloatTensor] = None
    layout_generated_ids: Optional[torch.LongTensor] = None
    layout_generated_eos: Optional[torch.BoolTensor] = None
    layout_truncated: Optional[torch.BoolTensor] = None
    layout_truncated_by_max_layout_tokens: Optional[torch.BoolTensor] = None
    layout_stopped_by_max_layout_records: Optional[torch.BoolTensor] = None
    layout_num_generated_regions: Optional[torch.LongTensor] = None
    layout_num_layout_tokens: Optional[torch.LongTensor] = None
    layout_region_token_probabilities: Optional[torch.FloatTensor] = None
    layout_sequence_log_probability: Optional[torch.FloatTensor] = None
    layout_coverage_summary: Optional[torch.FloatTensor] = None
    layout_coverage_region_counts: Optional[torch.LongTensor] = None
    layout_record_mask: Optional[torch.BoolTensor] = None
    layout_type_logits: Optional[torch.FloatTensor] = None


@dataclass
class GOTCausalLMOutputWithPast(CausalLMOutputWithPast):
    ocr_loss: Optional[torch.FloatTensor] = None
    layout_loss: Optional[torch.FloatTensor] = None
    layout_object_loss: Optional[torch.FloatTensor] = None
    layout_bbox_l1_loss: Optional[torch.FloatTensor] = None
    layout_bbox_giou_loss: Optional[torch.FloatTensor] = None
    layout_direction_loss: Optional[torch.FloatTensor] = None
    layout_object_accuracy: Optional[torch.FloatTensor] = None
    layout_bbox_mean_iou: Optional[torch.FloatTensor] = None
    layout_direction_accuracy: Optional[torch.FloatTensor] = None
    layout_object_logit_abs_max: Optional[torch.FloatTensor] = None
    layout_direction_logit_abs_max: Optional[torch.FloatTensor] = None
    layout_bbox_pred_min: Optional[torch.FloatTensor] = None
    layout_bbox_pred_max: Optional[torch.FloatTensor] = None
    layout_query_abs_max: Optional[torch.FloatTensor] = None
    layout_prediction_query_abs_max: Optional[torch.FloatTensor] = None
    layout_bbox_logit_abs_max: Optional[torch.FloatTensor] = None
    layout_object_logits: Optional[torch.FloatTensor] = None
    layout_bbox_xyxy: Optional[torch.FloatTensor] = None
    layout_direction_logits: Optional[torch.FloatTensor] = None
    layout_sequence_loss: Optional[torch.FloatTensor] = None
    layout_type_loss: Optional[torch.FloatTensor] = None
    layout_count_loss: Optional[torch.FloatTensor] = None
    layout_eos_accuracy: Optional[torch.FloatTensor] = None
    layout_region_count_mae: Optional[torch.FloatTensor] = None
    layout_generated_ids: Optional[torch.LongTensor] = None
    layout_generated_eos: Optional[torch.BoolTensor] = None
    layout_truncated: Optional[torch.BoolTensor] = None
    layout_truncated_by_max_layout_tokens: Optional[torch.BoolTensor] = None
    layout_stopped_by_max_layout_records: Optional[torch.BoolTensor] = None
    layout_num_generated_regions: Optional[torch.LongTensor] = None
    layout_num_layout_tokens: Optional[torch.LongTensor] = None
    layout_region_token_probabilities: Optional[torch.FloatTensor] = None
    layout_sequence_log_probability: Optional[torch.FloatTensor] = None
    layout_coverage_summary: Optional[torch.FloatTensor] = None
    layout_coverage_region_counts: Optional[torch.LongTensor] = None
    layout_record_mask: Optional[torch.BoolTensor] = None
    layout_type_logits: Optional[torch.FloatTensor] = None

class GOTConfig(Qwen2Config):
    model_type = "GOT"


class GOTQwenModel(Qwen2Model):
    config_class = GOTConfig

    def __init__(self, config: Qwen2Config):
        super(GOTQwenModel, self).__init__(config)

        self.vision_tower_high = build_vary_vit_b()

        self.mm_projector_vary =  nn.Linear(1024, 1024)

        self.layout_adapter = None
        self.variable_layout_adapter = None
        self.generic_adapter = None
        self.layout_criterion = None
        enabled_paths = sum(bool(getattr(config, name, False)) for name in (
            "use_vlqa", "use_generic_adapter", "variable_layout_enabled"
        ))
        if enabled_paths > 1:
            raise ValueError("Fixed-Slot VLQA, PVLD, and generic adapter are mutually exclusive.")
        if getattr(config, "use_generic_adapter", False):
            config.generic_adapter_dim = int(getattr(config, "generic_adapter_dim", 256))
            config.generic_adapter_num_heads = int(
                getattr(config, "generic_adapter_num_heads", 8)
            )
            config.generic_adapter_ffn_expansion = int(
                getattr(config, "generic_adapter_ffn_expansion", 8)
            )
            config.generic_adapter_dropout = float(
                getattr(config, "generic_adapter_dropout", 0.0)
            )
            self.generic_adapter = GenericVisualTransformerAdapter(
                visual_dim=1024,
                adapter_dim=config.generic_adapter_dim,
                num_heads=config.generic_adapter_num_heads,
                ffn_expansion=config.generic_adapter_ffn_expansion,
                dropout=config.generic_adapter_dropout,
            )
        if getattr(config, "use_vlqa", False):
            config.vlqa_num_queries = int(getattr(config, "vlqa_num_queries", 16))
            config.vlqa_adapter_dim = int(getattr(config, "vlqa_adapter_dim", 256))
            config.vlqa_num_heads = int(getattr(config, "vlqa_num_heads", 8))
            config.vlqa_ffn_expansion = int(getattr(config, "vlqa_ffn_expansion", 4))
            config.vlqa_dropout = float(getattr(config, "vlqa_dropout", 0.0))
            config.layout_writeback_mode = str(
                getattr(config, "layout_writeback_mode", "layout_value")
            )
            config.layout_writeback_source = str(
                getattr(config, "layout_writeback_source", "layout_evidence")
            )
            config.layout_writeback_num_heads = int(
                getattr(config, "layout_writeback_num_heads", config.vlqa_num_heads)
            )
            config.layout_writeback_dropout = float(
                getattr(config, "layout_writeback_dropout", config.vlqa_dropout)
            )
            config.layout_writeback_gate_init = float(
                getattr(config, "layout_writeback_gate_init", 0.0)
            )
            config.vlqa_num_direction_classes = int(
                getattr(config, "vlqa_num_direction_classes", 5)
            )
            config.vlqa_layout_input_dim = int(
                getattr(config, "vlqa_layout_input_dim", 1024)
            )
            config.vlqa_object_weight = float(
                getattr(config, "vlqa_object_weight", 1.0)
            )
            config.vlqa_bbox_l1_weight = float(
                getattr(config, "vlqa_bbox_l1_weight", 5.0)
            )
            config.vlqa_bbox_giou_weight = float(
                getattr(config, "vlqa_bbox_giou_weight", 2.0)
            )
            config.vlqa_direction_weight = float(
                getattr(config, "vlqa_direction_weight", 1.0)
            )
            config.ocr_loss_weight = float(getattr(config, "ocr_loss_weight", 1.0))
            config.layout_loss_weight = float(
                getattr(config, "layout_loss_weight", 1.0)
            )
            self.layout_adapter = VisualLayoutQueryAdapter(
                visual_dim=1024,
                layout_input_dim=config.vlqa_layout_input_dim,
                adapter_dim=config.vlqa_adapter_dim,
                num_queries=config.vlqa_num_queries,
                num_heads=config.vlqa_num_heads,
                ffn_expansion=config.vlqa_ffn_expansion,
                num_direction_classes=config.vlqa_num_direction_classes,
                dropout=config.vlqa_dropout,
                writeback_mode=config.layout_writeback_mode,
                writeback_num_heads=config.layout_writeback_num_heads,
                writeback_dropout=config.layout_writeback_dropout,
                writeback_gate_init=config.layout_writeback_gate_init,
            )
            self.layout_criterion = VisualLayoutQueryLoss(
                object_weight=config.vlqa_object_weight,
                bbox_l1_weight=config.vlqa_bbox_l1_weight,
                bbox_giou_weight=config.vlqa_bbox_giou_weight,
                direction_weight=config.vlqa_direction_weight,
            )
        if getattr(config, "variable_layout_enabled", False):
            config.pvld_decoder_version = "causal_transformer_fsm_previous_region_v1"
            config.pvld_decoder_memory = "layout_evidence_only"
            config.pvld_coverage_detach = False
            config.num_layout_prompt_queries = int(
                getattr(config, "num_layout_prompt_queries", 32)
            )
            config.max_layout_tokens = int(getattr(config, "max_layout_tokens", 2048))
            config.max_layout_records = int(getattr(config, "max_layout_records", 512))
            config.layout_decoder_layers = int(getattr(config, "layout_decoder_layers", 2))
            config.layout_decoder_hidden_size = int(
                getattr(config, "layout_decoder_hidden_size", 256)
            )
            config.layout_decoder_num_heads = int(
                getattr(config, "layout_decoder_num_heads", 8)
            )
            config.layout_writeback_mode = "visual_value_layout_routing"
            config.layout_writeback_source = "layout_evidence"
            self.variable_layout_adapter = PromptedVariableLayoutAdapter(
                visual_dim=1024,
                high_resolution_dim=1024,
                hidden_size=config.layout_decoder_hidden_size,
                num_prompt_queries=config.num_layout_prompt_queries,
                decoder_layers=config.layout_decoder_layers,
                num_heads=config.layout_decoder_num_heads,
                max_layout_tokens=config.max_layout_tokens,
                max_layout_records=config.max_layout_records,
                num_directions=int(getattr(config, "vlqa_num_direction_classes", 5)),
                dropout=float(getattr(config, "layout_writeback_dropout", 0.0)),
                gate_init=float(getattr(config, "layout_writeback_gate_init", 0.0)),
                bbox_weight=float(getattr(config, "layout_bbox_loss_weight", 5.0)),
                bbox_giou_weight=float(
                    getattr(config, "layout_bbox_giou_loss_weight", 2.0)
                ),
                type_weight=float(getattr(config, "layout_type_loss_weight", 1.0)),
                direction_weight=float(getattr(config, "layout_direction_loss_weight", 1.0)),
                count_weight=float(getattr(config, "layout_count_loss_weight", 0.1)),
                prompt_diversity_weight=float(
                    getattr(config, "layout_prompt_diversity_loss_weight", 0.0)
                ),
            )

    @staticmethod
    def _concatenate_layout_outputs(outputs: List[VLQAOutput]) -> VLQAOutput:
        if not outputs:
            raise ValueError("Cannot concatenate an empty VLQA output list.")
        return VLQAOutput(
            visual_tokens=torch.cat([output.visual_tokens for output in outputs], dim=0),
            layout_queries=torch.cat([output.layout_queries for output in outputs], dim=0),
            prediction_queries=torch.cat(
                [output.prediction_queries for output in outputs], dim=0
            ),
            object_logits=torch.cat([output.object_logits for output in outputs], dim=0),
            bbox_logits=torch.cat([output.bbox_logits for output in outputs], dim=0),
            bbox_cxcywh=torch.cat([output.bbox_cxcywh for output in outputs], dim=0),
            bbox_xyxy=torch.cat([output.bbox_xyxy for output in outputs], dim=0),
            direction_logits=torch.cat(
                [output.direction_logits for output in outputs], dim=0
            ),
            layout_residual=torch.cat(
                [output.layout_residual for output in outputs], dim=0
            ),
        )


    def initialize_vision_modules(
        self, 
        vision_tower,
        pretrained_stage1_model=None,
        freeze_vision_tower=False,
        use_im_start_end=False,
        vision_select_layer=-1,
        dtype=torch.float16,
        device="cuda"
    ):

        # Vary old codes, not use in GOT
        image_processor = BlipImageEvalProcessor(image_size=1024)
        # 1024*1024

        image_processor_high = BlipImageEvalProcessor(image_size=1024)


      
        self.vision_tower_high = self.vision_tower_high.to(dtype=dtype, device=device)

        self.mm_projector_vary = self.mm_projector_vary.to(dtype=dtype, device=device)


        image_token_len = 256

        self.config.vision_tower = vision_tower
        self.config.image_token_len = image_token_len
        # self.config.use_im_start_end = use_im_start_end
        self.config.use_im_start_end = True

        self.config.vision_select_layer = vision_select_layer
        self.config.freeze_vision_tower = freeze_vision_tower
        
        return dict(
            image_processor=image_processor,
            image_processor_high=image_processor_high,
            image_token_len=image_token_len,
        )
         
    # def get_input_embeddings(self, x):
    #     return self.wte(x)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        layout_bbox_targets: Optional[torch.FloatTensor] = None,
        layout_bbox_mask: Optional[torch.BoolTensor] = None,
        layout_object_targets: Optional[torch.FloatTensor] = None,
        layout_object_mask: Optional[torch.BoolTensor] = None,
        layout_direction_targets: Optional[torch.LongTensor] = None,
        layout_input_ids: Optional[torch.LongTensor] = None,
        layout_attention_mask: Optional[torch.BoolTensor] = None,
        layout_region_positions: Optional[torch.LongTensor] = None,
        layout_record_mask: Optional[torch.BoolTensor] = None,
        layout_type_targets: Optional[torch.LongTensor] = None,
        layout_count_targets: Optional[torch.FloatTensor] = None,
        generate_variable_layout: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        if orig_embeds_params is not None:
            with torch.no_grad():
                self.get_input_embeddings().weight[:-self.num_new_tokens] = orig_embeds_params[:-self.num_new_tokens].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        fixed_layout_targets = (
            layout_bbox_targets,
            layout_bbox_mask,
            layout_object_targets,
            layout_object_mask,
            layout_direction_targets,
        )
        if self.layout_adapter is not None and any(target is not None for target in fixed_layout_targets) and not all(
            target is not None for target in fixed_layout_targets
        ):
            raise ValueError("All VLQA target tensors must be supplied together.")
        has_layout_targets = self.layout_adapter is not None and all(
            target is not None for target in fixed_layout_targets
        )
        if has_layout_targets and self.layout_adapter is None:
            raise ValueError("Layout targets were supplied but config.use_vlqa is disabled.")
        if has_layout_targets and not return_dict:
            raise ValueError("VLQA supervision requires return_dict=True.")
        variable_targets = (
            layout_input_ids,
            layout_attention_mask,
            layout_region_positions,
            layout_record_mask,
            layout_bbox_targets,
            layout_type_targets,
            layout_direction_targets,
            layout_count_targets,
        )
        has_variable_targets = self.variable_layout_adapter is not None and all(
            target is not None for target in variable_targets
        )

        vision_tower_high = getattr(self, 'vision_tower_high', None)
        layout_outputs = []
        variable_layout_outputs = []


        if vision_tower_high is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
        # if True:
            # assert type(images) is list, ValueError("To fit both interleave and conversation, images must be list of batches of images")
            # print(im)
            use_im_start_end = getattr(self.config, "use_im_start_end", -1)

            vision_select_layer = getattr(self.config, "vision_select_layer", -1)
            im_patch_token = getattr(self.config, "im_patch_token", -1)
            im_start_token = getattr(self.config, "im_start_token", -1)
            im_end_token = getattr(self.config, "im_end_token", -1)
            freeze_vision_tower = getattr(self.config, "freeze_vision_tower", False)

            im_patch_token = 151859

            im_start_token = 151857

            im_end_token = 151858
            


            image_features = []
            

            for image_index, image in enumerate(images):
                P, C, H, W = image[1].shape
                # with torch.set_grad_enabled(True):
                #     # print(image[1].shape)
                #     cnn_feature = vision_tower_high(image[1])
                #     cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1) # 256  1024
                #     # image_features.append(cnn_feature)
                # image_features_2.append(cnn_feature)
                if P == 1:
                    with torch.set_grad_enabled(False):
                        # print(image[1].shape)
                        cnn_feature = vision_tower_high(image[1])
                        cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1) # 256*1024
                        # image_features.append(cnn_feature)
                    # image_features_2.append(cnn_feature)
                    image_feature = self.mm_projector_vary(cnn_feature)
                    if self.generic_adapter is not None:
                        image_feature = self.generic_adapter(image_feature)
                    if self.layout_adapter is not None:
                        layout_output = self.layout_adapter(
                            image_feature,
                            layout_memory=cnn_feature,
                            memory_grid_size=(16, 16),
                        )
                        image_feature = layout_output.visual_tokens
                        layout_outputs.append(layout_output)
                    elif self.variable_layout_adapter is not None:
                        variable_output = self.variable_layout_adapter(
                            image_feature,
                            cnn_feature,
                            layout_input_ids=(
                                layout_input_ids[image_index : image_index + 1]
                                if has_variable_targets else None
                            ),
                            layout_attention_mask=(
                                layout_attention_mask[image_index : image_index + 1]
                                if has_variable_targets else None
                            ),
                            layout_region_positions=(
                                layout_region_positions[image_index : image_index + 1]
                                if has_variable_targets else None
                            ),
                            layout_record_mask=(
                                layout_record_mask[image_index : image_index + 1]
                                if has_variable_targets else None
                            ),
                            layout_bbox_targets=(
                                layout_bbox_targets[image_index : image_index + 1]
                                if has_variable_targets else None
                            ),
                            layout_type_targets=(
                                layout_type_targets[image_index : image_index + 1]
                                if has_variable_targets else None
                            ),
                            layout_direction_targets=(
                                layout_direction_targets[image_index : image_index + 1]
                                if has_variable_targets else None
                            ),
                            layout_count_targets=(
                                layout_count_targets[image_index : image_index + 1]
                                if has_variable_targets else None
                            ),
                            generate_layout=generate_variable_layout,
                        )
                        image_feature = variable_output.visual_tokens
                        variable_layout_outputs.append(variable_output)
                    image_features.append(image_feature)

                else:
                    if (self.layout_adapter is not None or self.variable_layout_adapter is not None
                            or self.generic_adapter is not None):
                        raise ValueError(
                            "Formal visual-adapter training accepts one whole-page image per sample; "
                            f"received {P} image patches."
                        )
                    image_patches = torch.unbind(image[1])
                    image_patches_features = []
                    for image_patch in image_patches:
                        image_p = torch.stack([image_patch])
                        with torch.set_grad_enabled(False):
                            cnn_feature_p = vision_tower_high(image_p)
                            cnn_feature_p = cnn_feature_p.flatten(2).permute(0, 2, 1)
                        image_feature_p = self.mm_projector_vary(cnn_feature_p)
                        image_patches_features.append(image_feature_p)
                    image_feature = torch.cat(image_patches_features, dim=1)
                    # print(P)
                    # print(image_feature.shape)
                    # exit()
                    image_features.append(image_feature)



            dummy_image_features_2 = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            # dummy_image_features_2 = self.mm_projector_vary(dummy_image_features_2)
            dummy_image_features = dummy_image_features_2
            use_im_start_end = True
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
                if (cur_input_ids == im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue

                if use_im_start_end:
                    if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    
                    image_start_tokens = torch.where(cur_input_ids == im_start_token)[0]
                    for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                        per_cur_image_features = per_cur_image_features.to(device=cur_input_embeds.device)
                        num_patches = per_cur_image_features.shape[0]

                        if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        
                        cur_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:image_start_token_pos+1], 
                                per_cur_image_features, 
                                cur_input_embeds[image_start_token_pos + num_patches + 1:]
                            ), 
                            dim=0
                        )


                    new_input_embeds.append(cur_input_embeds)
                else:
                    raise NotImplementedError

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        combined_layout_output = (
            self._concatenate_layout_outputs(layout_outputs) if layout_outputs else None
        )
        layout_losses = None
        if has_layout_targets:
            if len(layout_outputs) != layout_bbox_targets.shape[0]:
                raise ValueError(
                    "VLQA output/target batch mismatch: "
                    f"outputs={len(layout_outputs)}, targets={layout_bbox_targets.shape[0]}."
                )
            layout_losses = self.layout_criterion(
                output=combined_layout_output,
                bbox_targets_xyxy=layout_bbox_targets,
                bbox_mask=layout_bbox_mask,
                object_targets=layout_object_targets,
                object_mask=layout_object_mask,
                direction_targets=layout_direction_targets,
            )
        variable_losses = [
            output.losses for output in variable_layout_outputs if output.losses is not None
        ]
        if has_variable_targets and len(variable_losses) != layout_input_ids.shape[0]:
            raise ValueError("PVLD output/target batch mismatch.")

        def mean_variable(name):
            values = [getattr(loss, name) for loss in variable_losses]
            return torch.stack(values).mean() if values else None

        generated_outputs = [
            output for output in variable_layout_outputs if output.decoder_output is not None
        ]
        generated_ids = None
        generated_eos = None
        generated_truncated = None
        generated_token_truncated = None
        stopped_by_records = None
        generated_region_counts = None
        generated_token_counts = None
        region_token_probabilities = None
        sequence_log_probability = None
        coverage_summary = None
        coverage_region_counts = None
        record_mask = None
        record_bbox = None
        record_type_logits = None
        record_direction_logits = None
        if generate_variable_layout and generated_outputs:
            def pad_records(value, width):
                if value.shape[1] == width:
                    return value
                padding_shape = list(value.shape)
                padding_shape[1] = width - value.shape[1]
                return torch.cat((value, value.new_zeros(padding_shape)), dim=1)

            record_width = max(output.record_mask.shape[1] for output in generated_outputs)
            generated_width = max(
                output.decoder_output.generated_ids.shape[1] for output in generated_outputs
            )
            generated_ids = torch.cat(
                [
                    pad_records(output.decoder_output.generated_ids, generated_width)
                    for output in generated_outputs
                ],
                dim=0,
            )
            generated_eos = torch.cat(
                [output.decoder_output.generated_eos for output in generated_outputs], dim=0
            )
            generated_truncated = torch.cat(
                [output.decoder_output.truncated for output in generated_outputs], dim=0
            )
            generated_token_truncated = torch.cat(
                [output.decoder_output.truncated_by_max_layout_tokens for output in generated_outputs], dim=0
            )
            stopped_by_records = torch.cat(
                [output.decoder_output.stopped_by_max_layout_records for output in generated_outputs], dim=0
            )
            generated_region_counts = torch.cat(
                [output.decoder_output.num_generated_regions for output in generated_outputs], dim=0
            )
            generated_token_counts = torch.cat(
                [output.decoder_output.num_layout_tokens for output in generated_outputs], dim=0
            )
            region_token_probabilities = torch.cat(
                [
                    pad_records(output.decoder_output.region_token_probabilities, record_width)
                    for output in generated_outputs
                ],
                dim=0,
            )
            sequence_log_probability = torch.cat(
                [output.decoder_output.sequence_log_probability for output in generated_outputs], dim=0
            )
            coverage_summary = torch.cat(
                [output.decoder_output.coverage_summary for output in generated_outputs], dim=0
            )
            coverage_region_counts = torch.cat(
                [output.decoder_output.coverage_region_counts for output in generated_outputs], dim=0
            )
            record_mask = torch.cat(
                [pad_records(output.record_mask, record_width) for output in generated_outputs],
                dim=0,
            )
            record_bbox = torch.cat(
                [pad_records(output.record_output.bbox, record_width) for output in generated_outputs],
                dim=0,
            )
            record_type_logits = torch.cat(
                [pad_records(output.record_output.type_logits, record_width) for output in generated_outputs],
                dim=0,
            )
            record_direction_logits = torch.cat(
                [
                    pad_records(output.record_output.direction_logits, record_width)
                    for output in generated_outputs
                ],
                dim=0,
            )

        base_output = super(GOTQwenModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids = position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if not return_dict:
            return base_output
        return GOTBaseModelOutputWithPast(
            last_hidden_state=base_output.last_hidden_state,
            past_key_values=base_output.past_key_values,
            hidden_states=base_output.hidden_states,
            attentions=base_output.attentions,
            layout_loss=(
                layout_losses.loss if layout_losses is not None else mean_variable("loss")
            ),
            layout_object_loss=(
                layout_losses.object_loss if layout_losses is not None else None
            ),
            layout_bbox_l1_loss=(
                layout_losses.bbox_l1_loss if layout_losses is not None
                else mean_variable("bbox_l1_loss")
            ),
            layout_bbox_giou_loss=(
                layout_losses.bbox_giou_loss if layout_losses is not None
                else mean_variable("bbox_giou_loss")
            ),
            layout_direction_loss=(
                layout_losses.direction_loss if layout_losses is not None
                else mean_variable("direction_loss")
            ),
            layout_object_accuracy=(
                layout_losses.object_accuracy if layout_losses is not None else None
            ),
            layout_bbox_mean_iou=(
                layout_losses.bbox_mean_iou if layout_losses is not None else None
            ),
            layout_direction_accuracy=(
                layout_losses.direction_accuracy if layout_losses is not None else None
            ),
            layout_object_logit_abs_max=(
                layout_losses.object_logit_abs_max if layout_losses is not None else None
            ),
            layout_direction_logit_abs_max=(
                layout_losses.direction_logit_abs_max
                if layout_losses is not None
                else None
            ),
            layout_bbox_pred_min=(
                layout_losses.bbox_pred_min if layout_losses is not None else None
            ),
            layout_bbox_pred_max=(
                layout_losses.bbox_pred_max if layout_losses is not None else None
            ),
            layout_query_abs_max=(
                layout_losses.query_abs_max if layout_losses is not None else None
            ),
            layout_prediction_query_abs_max=(
                layout_losses.prediction_query_abs_max
                if layout_losses is not None
                else None
            ),
            layout_bbox_logit_abs_max=(
                layout_losses.bbox_logit_abs_max if layout_losses is not None else None
            ),
            layout_object_logits=(
                combined_layout_output.object_logits
                if combined_layout_output is not None
                else None
            ),
            layout_bbox_xyxy=(
                combined_layout_output.bbox_xyxy
                if combined_layout_output is not None
                else record_bbox
            ),
            layout_direction_logits=(
                combined_layout_output.direction_logits
                if combined_layout_output is not None
                else record_direction_logits
            ),
            layout_sequence_loss=mean_variable("sequence_loss"),
            layout_type_loss=mean_variable("type_loss"),
            layout_count_loss=mean_variable("count_loss"),
            layout_eos_accuracy=mean_variable("eos_accuracy"),
            layout_region_count_mae=mean_variable("region_count_mae"),
            layout_generated_ids=generated_ids,
            layout_generated_eos=generated_eos,
            layout_truncated=generated_truncated,
            layout_truncated_by_max_layout_tokens=generated_token_truncated,
            layout_stopped_by_max_layout_records=stopped_by_records,
            layout_num_generated_regions=generated_region_counts,
            layout_num_layout_tokens=generated_token_counts,
            layout_region_token_probabilities=region_token_probabilities,
            layout_sequence_log_probability=sequence_log_probability,
            layout_coverage_summary=coverage_summary,
            layout_coverage_region_counts=coverage_region_counts,
            layout_record_mask=record_mask,
            layout_type_logits=record_type_logits,
        )



class GOTQwenForCausalLM(Qwen2ForCausalLM):
    config_class = GOTConfig
    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = GOTQwenModel(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, GOTQwenModel):
    #         module.gradient_checkpointing = value
    # @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        layout_bbox_targets: Optional[torch.FloatTensor] = None,
        layout_bbox_mask: Optional[torch.BoolTensor] = None,
        layout_object_targets: Optional[torch.FloatTensor] = None,
        layout_object_mask: Optional[torch.BoolTensor] = None,
        layout_direction_targets: Optional[torch.LongTensor] = None,
        layout_input_ids: Optional[torch.LongTensor] = None,
        layout_attention_mask: Optional[torch.BoolTensor] = None,
        layout_region_positions: Optional[torch.LongTensor] = None,
        layout_record_mask: Optional[torch.BoolTensor] = None,
        layout_type_targets: Optional[torch.LongTensor] = None,
        layout_count_targets: Optional[torch.FloatTensor] = None,
        generate_variable_layout: bool = False,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print(input_ids)
        # print(len(images))

        # print(inputs_embeds)

        outputs  = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            layout_bbox_targets=layout_bbox_targets,
            layout_bbox_mask=layout_bbox_mask,
            layout_object_targets=layout_object_targets,
            layout_object_mask=layout_object_mask,
            layout_direction_targets=layout_direction_targets,
            layout_input_ids=layout_input_ids,
            layout_attention_mask=layout_attention_mask,
            layout_region_positions=layout_region_positions,
            layout_record_mask=layout_record_mask,
            layout_type_targets=layout_type_targets,
            layout_count_targets=layout_count_targets,
            generate_variable_layout=generate_variable_layout,
            return_dict=return_dict
            
        )


        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # logits

        ocr_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            if shift_labels.ne(IGNORE_INDEX).any():
                loss_fct = CrossEntropyLoss()
                ocr_loss = loss_fct(shift_logits, shift_labels)
            else:
                ocr_loss = shift_logits.sum() * 0.0

        layout_loss = getattr(outputs, "layout_loss", None) if return_dict else None
        loss = None
        if ocr_loss is not None:
            loss = float(getattr(self.config, "ocr_loss_weight", 1.0)) * ocr_loss
        if layout_loss is not None:
            weighted_layout_loss = (
                float(getattr(self.config, "layout_loss_weight", 1.0)) * layout_loss
            )
            loss = weighted_layout_loss if loss is None else loss + weighted_layout_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return GOTCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            ocr_loss=ocr_loss,
            layout_loss=layout_loss,
            layout_object_loss=outputs.layout_object_loss,
            layout_bbox_l1_loss=outputs.layout_bbox_l1_loss,
            layout_bbox_giou_loss=outputs.layout_bbox_giou_loss,
            layout_direction_loss=outputs.layout_direction_loss,
            layout_object_accuracy=outputs.layout_object_accuracy,
            layout_bbox_mean_iou=outputs.layout_bbox_mean_iou,
            layout_direction_accuracy=outputs.layout_direction_accuracy,
            layout_object_logit_abs_max=outputs.layout_object_logit_abs_max,
            layout_direction_logit_abs_max=outputs.layout_direction_logit_abs_max,
            layout_bbox_pred_min=outputs.layout_bbox_pred_min,
            layout_bbox_pred_max=outputs.layout_bbox_pred_max,
            layout_query_abs_max=outputs.layout_query_abs_max,
            layout_prediction_query_abs_max=outputs.layout_prediction_query_abs_max,
            layout_bbox_logit_abs_max=outputs.layout_bbox_logit_abs_max,
            layout_object_logits=outputs.layout_object_logits,
            layout_bbox_xyxy=outputs.layout_bbox_xyxy,
            layout_direction_logits=outputs.layout_direction_logits,
            layout_sequence_loss=outputs.layout_sequence_loss,
            layout_type_loss=outputs.layout_type_loss,
            layout_count_loss=outputs.layout_count_loss,
            layout_eos_accuracy=outputs.layout_eos_accuracy,
            layout_region_count_mae=outputs.layout_region_count_mae,
            layout_generated_ids=outputs.layout_generated_ids,
            layout_generated_eos=outputs.layout_generated_eos,
            layout_truncated=outputs.layout_truncated,
            layout_truncated_by_max_layout_tokens=outputs.layout_truncated_by_max_layout_tokens,
            layout_stopped_by_max_layout_records=outputs.layout_stopped_by_max_layout_records,
            layout_num_generated_regions=outputs.layout_num_generated_regions,
            layout_num_layout_tokens=outputs.layout_num_layout_tokens,
            layout_region_token_probabilities=outputs.layout_region_token_probabilities,
            layout_sequence_log_probability=outputs.layout_sequence_log_probability,
            layout_coverage_summary=outputs.layout_coverage_summary,
            layout_coverage_region_counts=outputs.layout_coverage_region_counts,
            layout_record_mask=outputs.layout_record_mask,
            layout_type_logits=outputs.layout_type_logits,
        )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "generate_variable_layout": kwargs.get("generate_variable_layout", False),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(
        self, 
        tokenizer, 
        freeze_lm_model=False, 
        pretrained_stage1_model=None,
        device="cuda"
    ):
        config = self.get_model().config

        # add image patch token <image>
        # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        # config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

        config.im_patch_token = 151859

        config.use_im_start_end = True

        # add image start token <im_start> and end token <im_end>
        if config.use_im_start_end:
            # num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            # config.im_start_token, config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            config.im_start_token, config.im_end_token = 151857, 151858


AutoConfig.register("GOT", GOTConfig)
AutoModelForCausalLM.register(GOTConfig, GOTQwenForCausalLM)
