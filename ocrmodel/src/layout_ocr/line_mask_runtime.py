"""Last-layer line predictor; its output drives the NEXT cached decode.

Prefill stays unbiased, as in line100. Position q predicts the region for
label[q+2], used during forward q+1. The first LM token is not mask-controlled.
"""
from types import SimpleNamespace

import torch

from .decoder_mask_router import _normalized_grid_xywh
from .prefix_injection import _find_text_model
from .window_mask_routing import WindowRouting, WindowRoutingProfile


class LineMaskRuntime:
    def __init__(self, model, tokenizer, head, *, bias=None, layer_scope='all'):
        self.head = head
        self.text = _find_text_model(model)
        layers = list(self.text.layers)
        if not layers:
            raise ValueError('line-mask runtime requires at least one decoder layer')
        if layer_scope not in ('all', 'latter_half'):
            raise ValueError("layer_scope must be 'all' or 'latter_half'")
        first_layer = 0 if layer_scope == 'all' else len(layers) // 2
        self.layer_scope = layer_scope
        self.injection_layers = tuple(range(first_layer, len(layers)))
        self.bias = float(head.config.bias if bias is None else bias)
        if self.bias < 0:
            raise ValueError('line-mask bias must be non-negative')
        image_id = getattr(model.config, 'image_token_id', None)
        if image_id is None:
            image_id = model.config.text_config.image_token_id
        self.bridge = SimpleNamespace(image_token_id=image_id, last_mask=None)
        profile = WindowRoutingProfile(bias=self.bias, mask_threshold=head.config.threshold)
        self.route = WindowRouting(self.bridge, tokenizer, profile)
        self.merge = int(model.model.visual.spatial_merge_size)
        self.enabled = True
        self.features = None
        self.hidden = None
        self.keys = None
        self.previous = None
        self.capture_trace = False
        self.trace_steps = []
        self.prompt_prediction = None
        self._pending_trace = None
        self._generated_input_position = 0
        self._current_input_length = None
        self.handles = [model.register_forward_pre_hook(self.capture_input_shape, with_kwargs=True),
                        model.register_forward_pre_hook(self.route.observe_inputs, with_kwargs=True),
                        self.text.register_forward_pre_hook(self.capture_visual, with_kwargs=True),
                        self.text.layers[-1].register_forward_hook(self.capture_hidden)]
        self.handles += [layers[index].register_forward_pre_hook(self.route.hook, with_kwargs=True)
                         for index in self.injection_layers]

    def capture_input_shape(self, module, args, kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is None and args:
            input_ids = args[0]
        self._current_input_length = int(input_ids.shape[1]) if input_ids is not None else None

    def set_page(self, inputs, page_id='', query_positions=None, *, capture_trace=False):
        self.xywh, self.shape = _normalized_grid_xywh(inputs['image_grid_thw'], self.merge)
        self.positions = (inputs['input_ids'][0] == self.bridge.image_token_id).nonzero().flatten()
        if len(self.positions) != self.xywh.shape[1]:
            raise ValueError('visual grid and image token count disagree')
        self.query_positions = query_positions
        self.features = self.hidden = self.keys = self.previous = None
        self.bridge.last_mask = None
        self.route.applied_mask = None
        self.capture_trace = bool(capture_trace)
        self.trace_steps = []
        self.prompt_prediction = None
        self._pending_trace = None
        self._generated_input_position = 0
        self._current_input_length = None
        self.route.set_page(page_id, None, inputs['input_ids'].shape[1], inputs['input_ids'])

    def capture_visual(self, module, args, kwargs):
        if self.features is None:
            embeddings = kwargs.get('inputs_embeds')
            if embeddings is None:
                embeddings = args[0]
            self.features = embeddings[:, self.positions].detach()

    def capture_hidden(self, module, args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        self.hidden = hidden[:, self.query_positions].detach() if self.query_positions is not None else hidden[:, -1:].detach()
        if not self.enabled:
            return
        with torch.no_grad():
            if self.keys is None:
                self.keys = self.head.encode(self.features, self.xywh, self.shape)
                self.previous = self.keys.new_zeros(self.keys.shape[:2])
            is_prefill = self._current_input_length is not None and self._current_input_length > 1
            if self.capture_trace and not is_prefill:
                if self._current_input_length != 1:
                    raise RuntimeError('diagnostic cached decode must process exactly one input token')
                self.trace_steps.append(self._trace_applied_mask())
            prior = self.previous
            mask, _, update_logits, _ = self.head.step(self.hidden[:, -1], self.keys, prior)
            delta = (mask - prior).abs().mean(-1)[0]
            raw_update = update_logits.sigmoid()[0, 0]
            effective_update = 1.0 if bool(prior.sum(-1)[0] < 1e-6) else float(raw_update)
            prediction = {
                'source_generation_position': -1 if is_prefill else self._generated_input_position,
                'update_gate': effective_update,
                'update_gate_raw': float(raw_update),
                'mask_change_mean_abs': float(delta),
                'mask': mask[0].detach().to(torch.float16).clone() if self.capture_trace else None,
            }
            self.previous = mask
            self.bridge.last_mask = mask[:, None]
            self._pending_trace = prediction
            if self.capture_trace and is_prefill:
                self.prompt_prediction = prediction
            if self.capture_trace and not is_prefill:
                self._generated_input_position += 1

    def _trace_applied_mask(self):
        """Capture the mask used for the token emitted by this cached forward."""
        pending = self._pending_trace or {}
        binary = self.route.applied_mask
        return {
            'generation_position': self._generated_input_position + 1,
            'mask_source_generation_position': pending.get('source_generation_position'),
            'mask_applied': binary is not None,
            'update_gate': pending.get('update_gate'),
            'update_gate_raw': pending.get('update_gate_raw'),
            'mask_change_mean_abs': pending.get('mask_change_mean_abs'),
            'continuous_mask': pending.get('mask'),
            'binary_area': int(binary.sum()) if binary is not None else 0,
            'bias': self.bias,
            'layer_scope': self.layer_scope,
            'injection_layers': list(self.injection_layers),
        }

    def remove(self):
        for handle in self.handles:
            handle.remove()
