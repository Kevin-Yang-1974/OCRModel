from types import SimpleNamespace

import torch
from torch import nn

from layout_ocr.line_mask_head import LineMaskConfig, LineMaskHead
from layout_ocr.line_mask_runtime import LineMaskRuntime
from layout_ocr.line_mask_v3_diagnostics import (
    align_nonspace,
    fingerprint_decoder_lora,
    fingerprint_safetensors,
    stratified_sample,
)


def sample_records():
    return [
        {
            "page_id": f"p{index:02d}",
            "page_text": ("\u7532\u4e59\u4e19\u4e01" * ((index % 4) + 1)),
            "source_group_id": f"book-{index % 4}",
            "regions": [{"id": region} for region in range(index % 3)],
        }
        for index in range(24)
    ]


def test_stratified_sample_is_repeatable_unique_and_source_aware():
    records = sample_records()
    first, report = stratified_sample(records, count=12, seed=42)
    second, second_report = stratified_sample(records, count=12, seed=42)

    assert [row["page_id"] for row in first] == [row["page_id"] for row in second]
    assert report == second_report
    assert len(first) == len({row["page_id"] for row in first}) == 12
    assert {row["source_group_id"] for row in first} == {
        row["source_group_id"] for row in records
    }
    assert len(report["full_mask_trace_page_ids"]) == 3
    assert report["layout_density_quartile_coverage_complete"] is True
    assert set(report["selected_layout_density_quartile_counts"]) == {"0", "1", "2", "3"}
    assert report["selected_layout_region_count_range"] == [0, 2]


def test_alignment_marks_insertions_substitutions_and_repeated_text_ambiguity():
    alignment = align_nonspace("\u7532\u4e59", "\u7532\u4e19\u4e01")
    assert alignment["edit_distance"] == 2
    assert alignment["operations"] == ["M", "I", "S"]
    assert alignment["prediction_to_reference"] == [0, None, None]
    assert alignment["prediction_aligned_reference_position"] == [0, None, 1]

    repeated = align_nonspace("\u7532\u7532", "\u7532")
    assert repeated["candidate_reference_positions"] == [[0, 1]]
    assert repeated["ambiguous"] == [True]


def test_alignment_ignores_whitespace_for_character_mapping():
    alignment = align_nonspace("\u7532 \u4e59", "\u7532\u4e59")
    assert alignment["edit_distance"] == 0
    assert alignment["reference_positions"] == [0, 2]
    assert alignment["prediction_positions"] == [0, 1]
    assert alignment["prediction_to_reference"] == [0, 1]
    assert [alignment["reference_positions"][index]
            for index in alignment["prediction_to_reference"]] == [0, 2]
    assert alignment["ambiguous"] == [False, False]


def test_raw_model_fingerprint_hashes_shards_and_rejects_adapter_weights(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"raw model shard")
    first = fingerprint_safetensors(tmp_path, "revision-abc")
    second = fingerprint_safetensors(tmp_path, "revision-abc")
    assert first == second
    assert first["decoder_lora_loaded"] is False
    assert len(first["model_weights_sha256"]) == 64

    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")
    try:
        fingerprint_safetensors(tmp_path, "revision-abc")
    except ValueError as error:
        assert "adapter weights" in str(error)
    else:
        raise AssertionError("adapter weights should be rejected from the raw model snapshot")


def test_decoder_lora_fingerprint_records_frozen_training_start(tmp_path):
    weights = tmp_path / "decoder_lora.safetensors"
    weights.write_bytes(b"paired frozen decoder LoRA")

    result = fingerprint_decoder_lora(tmp_path, "revision-abc")

    assert result["model_revision"] == "revision-abc"
    assert result["decoder_lora_loaded"] is True
    assert result["decoder_lora_sha256"]
    assert result["rank"] == 8
    assert result["alpha"] == 8.0
    assert result["dropout"] == 0.0


class DiagnosticTokenizer:
    def decode(self, ids, **kwargs):
        return ""


class DiagnosticLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return hidden_states + 0.01


class DiagnosticText(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DiagnosticLayer(), DiagnosticLayer(), DiagnosticLayer()])

    def forward(self, inputs_embeds, **kwargs):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden_states=hidden, **kwargs)
        return hidden


class DiagnosticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.config = SimpleNamespace(image_token_id=3)
        self.model = nn.Module()
        self.model.language_model = DiagnosticText()
        self.model.visual = SimpleNamespace(spatial_merge_size=2)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **kwargs):
        return self.model.language_model(inputs_embeds=self.embedding(input_ids), **kwargs)


def test_runtime_bias_override_layer_scope_and_cached_decode_trace():
    model = DiagnosticModel()
    head = LineMaskHead(LineMaskConfig(hidden_size=16, dim=8))
    runtime = LineMaskRuntime(
        model, DiagnosticTokenizer(), head, bias=0.5, layer_scope="latter_half"
    )
    assert runtime.injection_layers == (1, 2)
    assert runtime.bias == 0.5

    inputs = {
        "input_ids": torch.tensor([[1, 3, 3, 3, 3, 2]]),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    runtime.set_page(inputs, "page-1", capture_trace=True)
    model(input_ids=inputs["input_ids"], cache_position=torch.arange(6))
    assert runtime.prompt_prediction is not None
    assert runtime.trace_steps == []

    model(input_ids=torch.tensor([[10]]), cache_position=torch.tensor([6]))
    assert len(runtime.trace_steps) == 1
    trace = runtime.trace_steps[0]
    assert trace["generation_position"] == 1
    assert trace["mask_source_generation_position"] == -1
    assert trace["mask_applied"] is True
    assert trace["injection_layers"] == [1, 2]
    runtime.remove()
