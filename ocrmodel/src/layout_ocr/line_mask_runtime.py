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
    def __init__(self, model, tokenizer, head):
        self.head = head
        self.text = _find_text_model(model)
        image_id = getattr(model.config, 'image_token_id', None)
        if image_id is None:
            image_id = model.config.text_config.image_token_id
        self.bridge = SimpleNamespace(image_token_id=image_id, last_mask=None)
        self.route = WindowRouting(self.bridge, tokenizer, WindowRoutingProfile(bias=head.config.bias, mask_threshold=head.config.threshold))
        self.merge = int(model.model.visual.spatial_merge_size)
        self.enabled = True
        self.features = None
        self.hidden = None
        self.keys = None
        self.previous = None
        self.handles = [model.register_forward_pre_hook(self.route.observe_inputs, with_kwargs=True),
                        self.text.register_forward_pre_hook(self.capture_visual, with_kwargs=True),
                        self.text.layers[-1].register_forward_hook(self.capture_hidden)]
        self.handles += [layer.register_forward_pre_hook(self.route.hook, with_kwargs=True) for layer in self.text.layers]

    def set_page(self, inputs, page_id='', query_positions=None):
        self.xywh, self.shape = _normalized_grid_xywh(inputs['image_grid_thw'], self.merge)
        self.positions = (inputs['input_ids'][0] == self.bridge.image_token_id).nonzero().flatten()
        if len(self.positions) != self.xywh.shape[1]:
            raise ValueError('visual grid and image token count disagree')
        self.query_positions = query_positions
        self.features = self.hidden = self.keys = self.previous = None
        self.bridge.last_mask = None
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
            self.previous, _, _, _ = self.head.step(self.hidden[:, -1], self.keys, self.previous)
            self.bridge.last_mask = self.previous[:, None]

    def remove(self):
        for handle in self.handles:
            handle.remove()
