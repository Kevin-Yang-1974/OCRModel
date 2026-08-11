from __future__ import annotations

import sys
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "GOT-OCR-2.0"
    / "scripts"
    / "local_tokenizer.py"
)
SPEC = spec_from_file_location("local_tokenizer_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
local_tokenizer = module_from_spec(SPEC)
sys.modules[SPEC.name] = local_tokenizer
SPEC.loader.exec_module(local_tokenizer)


class FakeTokenizer:
    calls: list[str] = []
    failures: dict[str, Exception] = {}

    @classmethod
    def from_pretrained(cls, path: str, **_: object) -> object:
        cls.calls.append(path)
        if path in cls.failures:
            raise cls.failures[path]
        return {"path": path}


class LocalTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeTokenizer.calls = []
        FakeTokenizer.failures = {}

    def test_candidates_are_ordered_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            primary = Path(temporary) / "primary"
            fallback = Path(temporary) / "fallback"
            self.assertEqual(
                local_tokenizer.tokenizer_candidates(primary, primary),
                (primary.resolve(),),
            )
            self.assertEqual(
                local_tokenizer.tokenizer_candidates(primary, fallback),
                (primary.resolve(), fallback.resolve()),
            )

    def test_loader_uses_transformers_for_nonstandard_qwen_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "qwen"
            directory.mkdir()
            (directory / "qwen.tiktoken").write_text("placeholder", encoding="ascii")
            loaded, used = local_tokenizer.load_local_tokenizer(
                FakeTokenizer,
                [directory],
                trust_remote_code=True,
                local_files_only=True,
            )
            self.assertEqual(loaded, {"path": str(directory.resolve())})
            self.assertEqual(used, directory.resolve())
            self.assertEqual(FakeTokenizer.calls, [str(directory.resolve())])

    def test_loader_continues_after_a_candidate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            first.mkdir()
            second.mkdir()
            FakeTokenizer.failures[str(first.resolve())] = ValueError("bad tokenizer")
            loaded, used = local_tokenizer.load_local_tokenizer(
                FakeTokenizer,
                [first, second],
            )
            self.assertEqual(loaded, {"path": str(second.resolve())})
            self.assertEqual(used, second.resolve())
            self.assertEqual(
                FakeTokenizer.calls,
                [str(first.resolve()), str(second.resolve())],
            )

    def test_loader_error_is_bounded_and_lists_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "broken"
            directory.mkdir()
            FakeTokenizer.failures[str(directory.resolve())] = ValueError("x" * 1000)
            with self.assertRaisesRegex(
                RuntimeError,
                r"Unable to load.*broken.*ValueError: x{237}\.\.\.",
            ):
                local_tokenizer.load_local_tokenizer(FakeTokenizer, [directory])


if __name__ == "__main__":
    unittest.main()
