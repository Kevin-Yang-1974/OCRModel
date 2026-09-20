"""Tests for how a locked-test evaluation learns a checkpoint's prefix.

Every prefix arm's locked test died inside ``load_prefix_checkpoint``: the
evaluator was built without a prefix, the checkpoint carried one, and the guard
that refuses to score an untrained projection fired.  The evaluator is reached from
launchers that never knew about the prefix, so the configuration has to come from
the artifact.  These tests pin the two directions of that contract -- what
``save_prefix_checkpoint`` writes and what the evaluator reads back.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from layout_ocr.train_screen import (
    PREFIX_CONFIG_NAME,
    declared_prefix_config,
    prefix_model_args,
    save_prefix_checkpoint,
)

DECLARATION = {
    "token_count": 4,
    "reserved_ids": [7, 8, 9, 10],
    "payload_mode": "regions",
    "position": "tail",
}


class _FakeInjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 5)


class _FakeRuntime:
    def __init__(self) -> None:
        self.token_count = DECLARATION["token_count"]
        self.reserved_ids = list(DECLARATION["reserved_ids"])
        self.position = DECLARATION["position"]
        self.injector = _FakeInjector()
        self.injector.payload_mode = DECLARATION["payload_mode"]


class _FakeBridge:
    def __init__(self, runtime) -> None:
        self.prefix_runtime = runtime


def test_a_checkpoint_without_a_prefix_declares_nothing(tmp_path):
    assert declared_prefix_config(tmp_path) is None
    assert prefix_model_args(tmp_path) == {
        "prefix_tokens": 0,
        "prefix_payload": "queries",
        "prefix_position": "front",
    }


def test_what_save_writes_is_what_the_evaluator_reads(tmp_path):
    """The two halves of the contract have to agree on the file and its keys."""

    saved = save_prefix_checkpoint(tmp_path, _FakeBridge(_FakeRuntime()), step=256)

    assert saved is not None
    assert (tmp_path / PREFIX_CONFIG_NAME).is_file()
    assert declared_prefix_config(tmp_path) == DECLARATION
    assert prefix_model_args(tmp_path) == {
        "prefix_tokens": 4,
        "prefix_payload": "regions",
        "prefix_position": "tail",
    }


def test_a_checkpoint_without_a_prefix_writes_no_declaration(tmp_path):
    assert save_prefix_checkpoint(tmp_path, _FakeBridge(None), step=1) is None
    assert not (tmp_path / PREFIX_CONFIG_NAME).exists()


def test_an_incomplete_declaration_is_refused(tmp_path):
    """Half a declaration would install a prefix that does not match the weights."""

    (tmp_path / PREFIX_CONFIG_NAME).write_text(
        json.dumps({"token_count": 4}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="is missing"):
        declared_prefix_config(tmp_path)


def test_a_declaration_that_is_not_an_object_is_refused(tmp_path):
    (tmp_path / PREFIX_CONFIG_NAME).write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        declared_prefix_config(tmp_path)
