"""Model adapters with dependency-gated execution.

Adapters never silently replace one model with another. Missing official
dependencies or an unsupported output contract become a controlled failure in
the prediction JSONL.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .registry import ModelSpec
from .schema import normalize_text


class AdapterError(RuntimeError):
    pass


def _json_safe(value: Any, *, _path: tuple[str, ...] = ()) -> Any:
    """Convert official provider containers to the prediction JSON schema."""
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, _path=(*_path, str(key)))
            for key, item in value.items()
            if not (str(key) == "img" and "blocks" in _path)
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, _path=_path) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), _path=_path)
    if isinstance(value, Path):
        return str(value)
    return value


class BaseAdapter:
    def __init__(self, spec: ModelSpec, model_root: Path, *, device: str = "cuda", dtype: str = "bf16") -> None:
        self.spec = spec
        self.model_root = model_root
        self.device = device
        self.dtype = dtype
        self.model: Any = None
        self.processor: Any = None
        self._model_loader: Any = None
        self.runtime: dict[str, Any] = {"device": device, "dtype": dtype, "quantization": None}

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, image_path: Path, prompt: str) -> tuple[Any, str, dict[str, Any]]:
        raise NotImplementedError

    def parameter_stats(self) -> dict[str, int | None]:
        total = None
        if self.model is not None and hasattr(self.model, "parameters"):
            total = sum(int(p.numel()) for p in self.model.parameters())
        return {"total_parameters": total, "trainable_parameters": None}


class TransformersImageTextAdapter(BaseAdapter):
    """Generic official Transformers path used by PaddleOCR-VL and GLM-OCR."""

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError:
                AutoModelForImageTextToText = None
            from transformers import AutoModelForCausalLM
        except Exception as exc:  # pragma: no cover - exercised on server
            raise AdapterError(f"official_transformers_dependency_missing: {exc}") from exc
        if not self.model_root.exists():
            raise AdapterError(f"deployed_model_missing: {self.model_root}")
        try:
            if self.spec.adapter == "paddleocr_vl":
                # PaddleOCR-VL ships its processor implementation alongside
                # the checkpoint. Transformers' generic registry raises a
                # KeyError for this provider, so use the verified official
                # class directly.
                import importlib.util
                import sys
                import types
                from transformers import modeling_rope_utils
                if "default" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
                    def _paddle_default_rope(config: Any = None, device: Any = None, seq_len: Any = None, layer_type: Any = None) -> tuple[Any, float]:
                        theta = float(getattr(config, "rope_theta", 10000.0))
                        head_dim = int(getattr(config, "head_dim", 0) or (int(config.hidden_size) // int(config.num_attention_heads)))
                        frequencies = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                        return 1.0 / (theta ** (frequencies / head_dim)), 1.0
                    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _paddle_default_rope
                package_name = "_paddleocr_official"
                package = types.ModuleType(package_name)
                package.__path__ = [str(self.model_root)]
                sys.modules[package_name] = package
                def load_official_module(module_name: str, file_name: str) -> Any:
                    qualified = f"{package_name}.{module_name}"
                    module_spec = importlib.util.spec_from_file_location(qualified, self.model_root / file_name)
                    if module_spec is None or module_spec.loader is None:
                        raise AdapterError(f"official_paddle_module_import_failed: {file_name}")
                    module = importlib.util.module_from_spec(module_spec)
                    sys.modules[qualified] = module
                    module_spec.loader.exec_module(module)
                    return module
                load_official_module("configuration_paddleocr_vl", "configuration_paddleocr_vl.py")
                processor_file = self.model_root / "processing_paddleocr_vl.py"
                if not processor_file.is_file():
                    raise AdapterError(f"official_paddle_processor_missing: {processor_file}")
                processor_module = load_official_module("processing_paddleocr_vl", "processing_paddleocr_vl.py")
                self.processor = processor_module.PaddleOCRVLProcessor.from_pretrained(str(self.model_root), trust_remote_code=True)
            else:
                self.processor = AutoProcessor.from_pretrained(str(self.model_root), trust_remote_code=True)
            # Keep the adapter independent of accelerate/device_map. The
            # bounded smoke loads one model at a time and moves it explicitly.
            kwargs: dict[str, Any] = {"trust_remote_code": True}
            if self.dtype == "bf16" and torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.bfloat16
            elif self.dtype == "fp16" and torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.float16
            if self.spec.adapter == "paddleocr_vl":
                model_module = load_official_module("modeling_paddleocr_vl", "modeling_paddleocr_vl.py")
                # Transformers 5.x calls this method while initializing custom
                # RotaryEmbedding modules. PaddleOCR-VL's official remote
                # code predates that method and already exposes the same
                # default-RoPE calculation through ``rope_init_fn``.
                def _compute_default_rope_parameters(config: Any, device: Any = None, **_: Any) -> tuple[Any, float]:
                    theta = float(getattr(config, "rope_theta", 10000.0))
                    head_dim = int(getattr(config, "head_dim", 0) or (int(config.hidden_size) // int(config.num_attention_heads)))
                    frequencies = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                    return 1.0 / (theta ** (frequencies / head_dim)), 1.0
                for _name in ("RotaryEmbedding", "Ernie4_5RotaryEmbedding"):
                    _cls = getattr(model_module, _name, None)
                    if _cls is not None and not hasattr(_cls, "compute_default_rope_parameters"):
                        _cls.compute_default_rope_parameters = staticmethod(_compute_default_rope_parameters)
                # The checkpoint's remote code passes ``cache_position`` to
                # create_causal_mask, but the installed Transformers overlay
                # still exposes the pre-5.x signature. Keep the official
                # mask implementation and discard only this newer keyword.
                import inspect
                _mask_fn = getattr(model_module, "create_causal_mask", None)
                if _mask_fn is not None and "cache_position" not in inspect.signature(_mask_fn).parameters:
                    def _compat_create_causal_mask(*args: Any, cache_position: Any = None, **kwargs: Any) -> Any:
                        return _mask_fn(*args, **kwargs)
                    model_module.create_causal_mask = _compat_create_causal_mask
                model_loader = model_module.PaddleOCRVLForConditionalGeneration
            else:
                model_loader = AutoModelForImageTextToText if AutoModelForImageTextToText is not None else AutoModelForCausalLM
            self.model = model_loader.from_pretrained(str(self.model_root), **kwargs)
            self._model_loader = model_loader
            if self.device != "cpu" and hasattr(self.model, "to"):
                self.model.to(self.device)
            self.model.eval()
            self.runtime.update(self.parameter_stats())
        except Exception as exc:  # pragma: no cover - provider-specific server dependency
            details = traceback.format_exc()[-2000:]
            raise AdapterError(f"official_model_load_failed: {type(exc).__name__}: {exc}\n{details}") from exc

    def _messages(self, image_path: Path, prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": prompt}]}]

    def predict(self, image_path: Path, prompt: str) -> tuple[Any, str, dict[str, Any]]:
        if self.model is None or self.processor is None:
            raise AdapterError("adapter_not_loaded")
        try:
            import torch
            from PIL import Image
            if self.spec.adapter == "paddleocr_vl":
                paddle_image = Image.open(image_path).convert("RGB")
                messages = [{"role": "user", "content": [{"type": "image", "image": paddle_image}, {"type": "text", "text": prompt}]}]
            else:
                messages = self._messages(image_path, prompt)
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_value: Any = str(image_path)
                if self.spec.adapter == "paddleocr_vl":
                    # PaddleOCR-VL's custom image processor bypasses the
                    # generic path loader and requires decoded PIL images.
                    image_value = paddle_image
                inputs = self.processor(text=[text], images=[image_value], return_tensors="pt")
            else:
                image_value = str(image_path)
                if self.spec.adapter == "paddleocr_vl":
                    image_value = paddle_image
                inputs = self.processor(images=[image_value], text=[prompt], return_tensors="pt")
            inputs = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            start = time.perf_counter()
            with torch.inference_mode():
                generation_kwargs: dict[str, Any] = {"max_new_tokens": 512, "do_sample": False}
                if self.spec.adapter == "paddleocr_vl" and inputs.get("input_ids") is not None:
                    # The official remote model's generation helper still
                    # expects the prefill cache_position explicitly, while
                    # Transformers 5.x no longer supplies it by default.
                    seq_len = inputs["input_ids"].shape[-1]
                    generation_kwargs["cache_position"] = torch.arange(seq_len, device=self.model.device)
                generated = self.model.generate(**inputs, **generation_kwargs)
            elapsed = time.perf_counter() - start
            input_length = inputs.get("input_ids").shape[-1] if inputs.get("input_ids") is not None else 0
            decoded = self.processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)[0]
            stats = {"latency_seconds": elapsed, "pages_per_second": 1.0 / elapsed if elapsed else None}
            if torch.cuda.is_available():
                stats["peak_memory_mib"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            return {"text": decoded}, normalize_text(decoded), stats
        except Exception as exc:
            details = traceback.format_exc()[-2000:]
            raise AdapterError(f"official_inference_failed: {type(exc).__name__}: {exc}\n{details}") from exc

    def train_one_step(self, image_path: Path, target_text: str, output_dir: Path) -> dict[str, Any]:
        """Run a bounded teacher-forcing step through the official Transformers model.

        This is only an engineering smoke. It uses one train page and never
        reads validation or test; formal GLM-OCR fine-tuning remains the
        official LLaMA-Factory route recorded in the registry.
        """
        if self.model is None or self.processor is None:
            raise AdapterError("adapter_not_loaded")
        try:
            import torch
            from PIL import Image
            if self.spec.adapter == "paddleocr_vl":
                train_image = Image.open(image_path).convert("RGB")
                messages = [{"role": "user", "content": [{"type": "image", "image": train_image}, {"type": "text", "text": target_text[:4096]}]}]
            else:
                train_image = None
                messages = self._messages(image_path, target_text[:4096])
            image_value = train_image if train_image is not None else str(image_path)
            if hasattr(self.processor, "apply_chat_template"):
                prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                inputs = self.processor(text=[prompt], images=[image_value], return_tensors="pt")
            else:
                inputs = self.processor(images=[image_value], text=[target_text[:4096]], return_tensors="pt")
            inputs = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            if "input_ids" not in inputs:
                raise AdapterError("official_processor_did_not_return_input_ids")
            # The generic smoke does not have PaddleOCR-VL's official
            # multimodal collator. Ignore visual placeholder tokens and use
            # one final text token for a finite teacher-forcing objective;
            # this is only a bounded engineering check, not a training loss
            # definition for formal fine-tuning.
            inputs["labels"] = inputs["input_ids"].clone()
            if inputs["labels"].shape[-1] > 1:
                inputs["labels"][:, :-1] = -100
            trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=1e-7)
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(**inputs)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise AdapterError("non_finite_loss")
            loss.backward()
            gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise AdapterError("non_finite_gradient")
            optimizer.step()
            checkpoint_dir = output_dir / "checkpoint-step-1"
            tied_keys = getattr(self.model, "_tied_weights_keys", None)
            if isinstance(tied_keys, list):
                # Transformers 5.x expects a mapping here, while the
                # official Paddle remote class still declares a list.
                self.model._tied_weights_keys = {}
            try:
                self.model.save_pretrained(checkpoint_dir)
            finally:
                if isinstance(tied_keys, list):
                    self.model._tied_weights_keys = tied_keys
            if hasattr(self.processor, "save_pretrained"):
                self.processor.save_pretrained(checkpoint_dir)
            from transformers import AutoModelForCausalLM
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError:
                AutoModelForImageTextToText = None
            if self.spec.adapter == "paddleocr_vl" and self._model_loader is not None:
                reload_loader = self._model_loader
            else:
                reload_loader = AutoModelForImageTextToText if self.spec.adapter == "glm_ocr" and AutoModelForImageTextToText is not None else AutoModelForCausalLM
            reload_loader.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
            self.model.eval()
            return {"loss": float(loss.detach().cpu()), "loss_finite": True, "gradient_finite": True, "checkpoint_reload_ok": True, "trainable_parameters": sum(int(p.numel()) for p in trainable), "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None}
        except AdapterError:
            raise
        except Exception as exc:
            details = traceback.format_exc()[-3000:]
            raise AdapterError(f"official_one_step_failed: {type(exc).__name__}: {exc}\n{details}") from exc


class OpenDocAdapter(BaseAdapter):
    """OpenOCR/OpenDoc official CLI adapter; it preserves JSON and Markdown."""

    def load(self) -> None:
        if not self.model_root.exists():
            raise AdapterError(f"deployed_model_missing: {self.model_root}")
        if os.environ.get("OPENOCR_ONNX", "0") == "1":
            try:
                configured_openocr = os.environ.get("OPENOCR_PYTHONPATH", "")
                for candidate in configured_openocr.split(os.pathsep):
                    openocr_root = Path(candidate) / "openocr"
                    if openocr_root.is_dir():
                        # OpenOCR's internal absolute ``tools.*`` imports are
                        # isolated to this import; keep it after this
                        # project's source path so ``tools.sota`` wins.
                        if str(candidate) not in sys.path:
                            sys.path.insert(0, str(candidate))
                        project_tools = sys.modules.get("tools")
                        if project_tools is not None and hasattr(project_tools, "__path__"):
                            tools_path = str(openocr_root / "tools")
                            if tools_path not in project_tools.__path__:
                                project_tools.__path__.append(tools_path)
                        break
                from openocr.tools.infer_doc_onnx import OpenDocONNX
                cache_root = Path(os.environ["OPENOCR_ONNX_CACHE"])
                unirec_root = cache_root / ".cache" / "openocr" / "unirec_0_1b_onnx"
                required = [
                    unirec_root / "unirec_encoder.onnx",
                    unirec_root / "unirec_decoder.onnx",
                    unirec_root / "unirec_tokenizer_mapping.json",
                ]
                if not all(path.is_file() for path in required):
                    raise AdapterError("official_opendoc_onnx_assets_missing")
                self._onnx_pipeline = OpenDocONNX(
                    unirec_encoder_path=str(required[0]),
                    unirec_decoder_path=str(required[1]),
                    tokenizer_mapping_path=str(required[2]),
                    use_gpu=self.device != "cpu",
                    use_layout_detection=False,
                    auto_download=False,
                )
                self.runtime["provider"] = "official_openocr_onnx"
                self.runtime["layout_detection"] = False
                self.runtime["onnx_cache"] = str(cache_root)
                return
            except AdapterError:
                raise
            except Exception as exc:
                details = traceback.format_exc()[-2000:]
                raise AdapterError(f"official_opendoc_onnx_load_failed: {type(exc).__name__}: {exc}\n{details}") from exc
        configured_pythonpath = os.environ.get("OPENOCR_PYTHONPATH")
        # OPENOCR_PYTHONPATH may be a platform path list; the first existing
        # directory is the provider package root used by the module runner.
        candidates = configured_pythonpath.split(os.pathsep) if configured_pythonpath else []
        self._pythonpath = next((Path(item) for item in candidates if Path(item).is_dir()), None)
        if self._pythonpath is not None and self._pythonpath.is_dir():
            self.runtime["pythonpath"] = str(self._pythonpath)
            return
        try:
            completed = subprocess.run(["openocr", "--version"], capture_output=True, text=True, timeout=20)
        except FileNotFoundError as exc:
            raise AdapterError("official_openocr_dependency_missing: openocr executable not found") from exc
        if completed.returncode != 0:
            raise AdapterError(f"official_openocr_dependency_failed: {completed.stderr[-500:]}")
        self.runtime["openocr_version"] = completed.stdout.strip()

    def predict(self, image_path: Path, prompt: str) -> tuple[Any, str, dict[str, Any]]:
        if getattr(self, "_onnx_pipeline", None) is not None:
            start = time.perf_counter()
            try:
                raw = self._onnx_pipeline(img_path=str(image_path), max_length=512, merge_layout_blocks=True)
                elapsed = time.perf_counter() - start
                text = normalize_text(raw)
                return _json_safe(raw), text, {"latency_seconds": elapsed, "pages_per_second": 1.0 / elapsed if elapsed else None}
            except Exception as exc:
                details = traceback.format_exc()[-2000:]
                raise AdapterError(f"official_opendoc_onnx_inference_failed: {type(exc).__name__}: {exc}\n{details}") from exc
        output_dir = self.model_root / ".sota_smoke_output"
        output_dir.mkdir(exist_ok=True)
        if getattr(self, "_pythonpath", None):
            command = [sys.executable, "-m", "openocr.tools.infer_doc", "--input_path", str(image_path), "--output_path", str(output_dir), "--gpus", "0", "--is_save_json", "--is_save_markdown", "--pretty"]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(self._pythonpath) + os.pathsep + environment.get("PYTHONPATH", "")
        else:
            command = ["openocr", "--task", "doc", "--input_path", str(image_path), "--use_layout_detection", "--save_json", "--save_markdown", "--output_path", str(output_dir)]
            environment = None
        start = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300, env=environment)
        elapsed = time.perf_counter() - start
        if completed.returncode != 0:
            raise AdapterError(f"official_openocr_failed: {completed.stderr[-800:]}")
        json_candidates = sorted(output_dir.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        markdown_candidates = sorted(output_dir.rglob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
        raw: Any = {"stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}
        if json_candidates:
            raw["json"] = json.loads(json_candidates[0].read_text(encoding="utf-8"))
        if markdown_candidates:
            raw["markdown"] = markdown_candidates[0].read_text(encoding="utf-8")
        text = normalize_text(raw)
        return raw, text, {"latency_seconds": elapsed, "pages_per_second": 1.0 / elapsed if elapsed else None}


def build_adapter(spec: ModelSpec, model_root: Path, *, device: str = "cuda", dtype: str = "bf16") -> BaseAdapter:
    if spec.adapter in {"paddleocr_vl", "glm_ocr"}:
        return TransformersImageTextAdapter(spec, model_root, device=device, dtype=dtype)
    if spec.adapter == "opendoc":
        return OpenDocAdapter(spec, model_root, device=device, dtype=dtype)
    if spec.adapter == "mineru":
        raise AdapterError("official_mineru_adapter_requires_provider_runtime: use MinerU official CLI/API contract")
    raise AdapterError(f"no_adapter_registered: {spec.adapter}")
