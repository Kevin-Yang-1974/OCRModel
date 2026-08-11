from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


GOT_EXPECTED = {
    "GOT": "0.1.0",
    "torch": "2.0.1+cu118",
    "torchvision": "0.15.2+cu118",
    "transformers": "4.37.2",
    "deepspeed": "0.12.3",
    "accelerate": "0.28.0",
}
ANANDASKY_EXPECTED = {
    "torch": "2.5.1+cu121",
    "transformers": "4.55.4",
    "accelerate": "1.10.1",
    "flash_attn": "2.7.4.post1",
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    root = Path(os.environ.get("OCRMODEL_ROOT", default_root))
    workspace = Path(os.environ.get("OCR_WORKSPACE", root.parent))
    parser = argparse.ArgumentParser(description="Emit a compact, read-only A100 environment check.")
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--ocrmodel-root", type=Path, default=root)
    parser.add_argument("--include-anandasky", action="store_true")
    return parser.parse_args()


def bounded_text(value: str, limit: int = 1200) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[:limit] + "...[truncated]"


def environment_report(
    python: Path,
    expected_python: str,
    expected_packages: dict[str, str],
    expected_source: Path | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {"python": str(python), "present": python.is_file()}
    if not python.is_file():
        result["ok"] = False
        return result

    probe = """
import importlib.metadata as metadata
import json
import os
import platform

packages = json.loads(os.environ['OCRMODEL_PACKAGES'])
payload = {'python_version': platform.python_version(), 'packages': {}, 'direct_url': None}
for name in packages:
    try:
        distribution = metadata.distribution(name)
        payload['packages'][name] = distribution.version
        if name == 'GOT':
            direct_url = distribution.read_text('direct_url.json')
            payload['direct_url'] = json.loads(direct_url) if direct_url else None
    except metadata.PackageNotFoundError:
        payload['packages'][name] = None
print(json.dumps(payload, ensure_ascii=True, separators=(',', ':')))
"""
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OCRMODEL_PACKAGES"] = json.dumps(sorted(expected_packages))
    try:
        completed = subprocess.run(
            [str(python), "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        result.update({"ok": False, "probe_error": f"{type(exc).__name__}: {exc}"})
        return result

    package_versions = payload.get("packages", {})
    mismatches = {
        name: {"expected": expected, "actual": package_versions.get(name)}
        for name, expected in expected_packages.items()
        if package_versions.get(name) != expected
    }
    result.update(
        {
            "python_version": payload.get("python_version"),
            "expected_python": expected_python,
            "package_mismatches": mismatches,
            "direct_url": payload.get("direct_url"),
        }
    )

    source_ok = True
    if expected_source is not None:
        direct_url = payload.get("direct_url") or {}
        url = direct_url.get("url") if isinstance(direct_url, dict) else None
        parsed = unquote(urlparse(url).path) if isinstance(url, str) else ""
        source_ok = Path(parsed).resolve() == expected_source.resolve() if parsed else False
        result["editable_source_ok"] = source_ok

    try:
        pip_check = subprocess.run(
            [str(python), "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        pip_check_ok = pip_check.returncode == 0
        result["pip_check"] = bounded_text(pip_check.stdout or pip_check.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pip_check_ok = False
        result["pip_check"] = f"{type(exc).__name__}: {exc}"

    result["ok"] = (
        payload.get("python_version") == expected_python
        and not mismatches
        and source_ok
        and pip_check_ok
    )
    return result


def main() -> int:
    args = parse_args()
    workspace = args.workspace
    root = args.ocrmodel_root
    got_root = Path(os.environ.get("GOT_PROJECT_ROOT", root / "src" / "GOT-OCR-2.0"))
    source_model = Path(
        os.environ.get(
            "GOT_SOURCE_MODEL",
            workspace / "models" / "GOT-OCR2_0",
        )
    )
    ancientdoc_root = Path(
        os.environ.get("ANCIENTDOC_ROOT", workspace / "datasets" / "AncientDoc")
    )

    paths = {
        "config_present": (root / "config" / "paths.env").is_file(),
        "got_source_present": (got_root / "pyproject.toml").is_file(),
        "got_source": str(got_root),
        "got_model_present": (source_model / "model.safetensors").is_file(),
        "ancientdoc_present": (ancientdoc_root / "label_for_got_split5.json").is_file(),
    }
    report: dict[str, object] = {
        "workspace": str(workspace),
        "ocrmodel_root": str(root),
        "paths": paths,
        "got": environment_report(
            workspace / "envs" / "got2" / "bin" / "python",
            "3.10.20",
            GOT_EXPECTED,
            got_root,
        ),
    }
    ok = bool(paths["got_source_present"]) and bool(paths["got_model_present"])
    ok = ok and bool(report["got"]["ok"])
    if args.include_anandasky:
        ananda_model = Path(os.environ.get("ANANDASKY_MODEL", workspace / "models" / "AnandaSky"))
        paths["anandasky_model_present"] = (ananda_model / "model.safetensors").is_file()
        report["anandasky"] = environment_report(
            workspace / "envs" / "anandasky" / "bin" / "python",
            "3.11.15",
            ANANDASKY_EXPECTED,
        )
        ok = ok and bool(paths["anandasky_model_present"]) and bool(report["anandasky"]["ok"])
    report["ok"] = ok
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
