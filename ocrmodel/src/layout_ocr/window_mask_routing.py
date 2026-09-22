"""line100 routing with 3--5-character windows and a first-layer mask head.

The original AttentionRouting hook owns decode-only, all-layer, once-per-step
bias injection. Its synced pointer is used ONLY for GT diagnostics. Deployment
uses the previous forward's predicted window; neither text nor boxes enter it.
"""

from dataclasses import asdict, dataclass

import torch

from .attention_routing import AttentionRouting
from .decoder_mask_model import DecoderMaskRuntime
from .decoder_mask_router import DecoderMaskConfig, DecoderMaskRouter
from .prefix_injection import _find_text_model


@dataclass(frozen=True)
class WindowRoutingProfile:
    anchor: str = "line100"
    historical_cer: float = 0.124694
    bias: float = 1.0
    mask_layer: int = 0
    mask_threshold: float = 0.5
    window_min: int = 3
    window_max: int = 5
    max_pixels: int = 4000000
    max_new_tokens: int = 1536
    seed: int = 42
    acceptance_cer: float = 0.13
    validation_pages: int = 149
    validation_sha256: str = "36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348"


class WindowRouting(AttentionRouting):
    def __init__(self, head_runtime, tokenizer, profile):
        super().__init__(
            head_runtime,
            profile.bias,
            head_runtime.image_token_id,
            tokenizer=tokenizer,
            pointer="step",
        )
        self.profile = profile
        self.source = "predicted"
        self.gt_masks = None
        self.char_to_token = []
        self.applied_mask = None

    def _mask_for(self, step, kv_length, device, dtype):
        if self.source == "gt":
            token = (
                self.char_to_token[self.position]
                if self.position < len(self.char_to_token)
                else None
            )
            mask = None if token is None else self.gt_masks[:, token : token + 1]
        else:
            mask = self.bridge.last_mask
        if mask is None:
            self.applied_mask = None
            self.missing += 1
            return None
        inside = mask[0, -1].detach() >= self.profile.mask_threshold
        self.applied_mask = inside
        self.biased += 1
        self.boxes_hit += int(inside.sum())
        bias = torch.zeros(1, 1, 1, kv_length, device=device, dtype=dtype)
        bias[0, 0, 0, self.visual_start : self.visual_start + self.visual_count] = (
            inside.to(dtype) * self.bias
        )
        return bias


class FirstLayerWindowRuntime:
    """Frozen backbone + learned recurrent head at layer 0 output.

    Prediction made at query q is used at q+1 (not q). In training its target
    must therefore be the window for label[q+2]. Prefill is unbiased, matching
    line100; the first decode consumes the window produced at prefill's last row.
    """

    def __init__(self, model, tokenizer, config=None, profile=None):
        profile = profile or WindowRoutingProfile()
        self.profile = profile
        text = _find_text_model(model)
        hidden = model.get_input_embeddings().weight.shape[1]
        config = config or DecoderMaskConfig(
            target_mode="window",
            bias_max=profile.bias,
            split_layer=0,
            mask_feedback_noise=0,
            input_noise=0,
        )
        if config.visual_source != "merged" or config.head != "mlp":
            raise ValueError("first-layer window routing uses the merged-grid MLP head")
        config = DecoderMaskConfig(
            **{
                **config.__dict__,
                "hidden_size": hidden,
                "visual_hidden_size": hidden,
                "split_layer": 0,
            }
        )
        self.head = DecoderMaskRouter(config).to(device=next(model.parameters()).device)
        text.add_module("decoder_mask_router", self.head)
        image_id = getattr(model.config, "image_token_id", None)
        if image_id is None:
            image_id = model.config.text_config.image_token_id
        self.runtime = DecoderMaskRuntime(
            self.head, config, image_id, int(model.model.visual.spatial_merge_size)
        )
        self.route = WindowRouting(self.runtime, tokenizer, profile)
        self.visual_features = None
        self.handles = [
            model.register_forward_pre_hook(self.route.observe_inputs, with_kwargs=True),
            text.register_forward_pre_hook(self.capture_visual, with_kwargs=True),
            *[
                layer.register_forward_pre_hook(self.route.hook, with_kwargs=True)
                for layer in text.layers
            ],
            text.layers[profile.mask_layer].register_forward_hook(self.capture),
        ]

    def capture_visual(self, module, args, kwargs):
        if self.runtime.keys is None:
            embeddings = kwargs.get("inputs_embeds")
            if embeddings is None and args:
                embeddings = args[0]
            self.visual_features = embeddings[:, self.runtime.image_positions].detach()
        self.runtime.capture_visual(module, args, kwargs)

    def capture(self, module, args, output):
        if self.route.source == "gt":
            return
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        mask, stop, logits = self.runtime._run_head(hidden.detach())
        self.runtime.last_mask = mask
        self.runtime.last_stop = stop
        self.runtime.last_logits = logits

    def set_page(self, inputs, page_id="", *, gt_targets=None, reference=None):
        self.runtime.clear_page()
        self.visual_features = None
        length = inputs["input_ids"].shape[1]
        self.runtime.set_page(inputs["image_grid_thw"], inputs["input_ids"], length, None)
        route = self.route
        route.source = "gt" if gt_targets is not None else "predicted"
        route.pointer = "synced" if gt_targets is not None else "step"
        route.gt_masks = None
        route.char_to_token = []
        route.applied_mask = None
        if gt_targets is not None:
            if reference is None:
                raise ValueError("GT synced diagnostics require a reference")
            route.gt_masks = gt_targets.mask
            route.char_to_token = [None] * len(reference)
            for token, span in enumerate(gt_targets.char_spans):
                if span is not None:
                    for char in range(max(0, span[0]), min(len(reference), span[1])):
                        if route.char_to_token[char] is None:
                            route.char_to_token[char] = token
        elif reference is not None:
            raise ValueError("predicted-mask inference does not accept reference text")
        # Empty characters is an explicit GT-pointer marker; no boxes are read by
        # our mask provider. None disables the inherited reference observer.
        route.set_page(
            page_id,
            [] if gt_targets is not None else None,
            length,
            inputs["input_ids"],
            reference=reference,
        )

    def detach_recurrence(self):
        if self.runtime.prev_mask is not None:
            self.runtime.prev_mask = self.runtime.prev_mask.detach()
        if self.visual_features is not None:
            # Rebuild the projection graph at each TBPTT boundary. Detaching keys
            # permanently would train the visual projection on only the first chunk.
            self.runtime.keys, _, _ = self.head.project_visual(
                self.visual_features, self.runtime.grid_thw, self.runtime.spatial_merge_size
            )

    def report(self):
        return {
            "profile": asdict(self.profile),
            "mask_source": self.route.source,
            "head_layer": self.profile.mask_layer,
            "predicted_mask_lag": 1,
            "injection_layers": "all",
            "prefill_bias": False,
            "reads_ground_truth": self.route.source == "gt",
            "routing": self.route.report(),
        }

    def remove(self):
        for handle in self.handles:
            handle.remove()
