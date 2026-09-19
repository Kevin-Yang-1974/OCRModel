"""Shell launchers must reach the Linux runner with LF line endings.

``set -Eeuo pipefail`` is the first real statement of every launcher here.  With
CRLF endings the remote bash reads the last word as ``pipefail\\r`` and refuses:
``set: pipefail: invalid option name``.  The launcher then exits 2 at line 20,
before any of its own validation, and a driver that only echoes exit codes looks
like a scheduler problem rather than a byte-level one.

This is not hypothetical.  On 2026-09-19 an edit made through a tool that writes
with the platform's default newline put CRLF into two launchers, and the training
comparison died instantly on both arms with no useful message.

The check is on the raw bytes rather than on a text read, because Python's
universal-newline handling hides the difference.
"""

from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
SHELL_GLOBS = ("tools/**/*.sh", "config/**/*.sh", "*.sh")


def _shell_scripts() -> list[Path]:
    found: set[Path] = set()
    for pattern in SHELL_GLOBS:
        found.update(REPOSITORY.glob(pattern))
    return sorted(path for path in found if path.is_file())


def test_shell_scripts_use_lf_line_endings() -> None:
    scripts = _shell_scripts()
    assert scripts, "no shell scripts were found; the glob is wrong"
    offenders = [
        str(path.relative_to(REPOSITORY))
        for path in scripts
        if b"\r\n" in path.read_bytes()
    ]
    assert not offenders, (
        "shell scripts contain CRLF line endings, which makes bash read "
        f"'pipefail\\r' as an option name: {offenders}"
    )


def test_shell_scripts_start_with_a_bash_shebang() -> None:
    """A ``sh`` shebang would reject ``pipefail`` even with LF endings."""

    offenders = []
    for path in _shell_scripts():
        first = path.read_bytes().split(b"\n", 1)[0]
        if not first.startswith(b"#!/") or b"bash" not in first:
            offenders.append(str(path.relative_to(REPOSITORY)))
    assert not offenders, f"shell scripts without a bash shebang: {offenders}"
