"""Repeat the historical v3 audit using its checksum-verified frozen source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    output = root / args.output
    if output.exists():
        raise ValueError("Historical reports are preserved; choose a new output")
    archive_path = root / "data/reports/completion-v3-source.zip"
    with tempfile.TemporaryDirectory(prefix="practica-v3-") as temporary:
        snapshot = pathlib.Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("snapshot-manifest.json"))
            if manifest["code_version"] != "5062061eb904a32cda814083dc931a9b4fb9bd9a6b8b5a5e1f479dcc501bc309":
                raise ValueError("Unexpected historical code version")
            if (
                hashlib.sha256(json.dumps(manifest["source_sha256"], sort_keys=True).encode()).hexdigest()
                != manifest["code_version"]
            ):
                raise ValueError("Invalid snapshot version")
            for filename, expected in manifest["source_sha256"].items():
                path = pathlib.Path(filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Invalid snapshot path")
                content = archive.read(filename)
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"Snapshot changed: {filename}")
                destination = snapshot / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
        (snapshot / "data").symlink_to(root / "data", target_is_directory=True)
        for original in root.glob("Постановка*.docx"):
            (snapshot / original.name).symlink_to(original)
        helper = snapshot / "docs/ml/evaluate_completion.py"
        helper.parent.mkdir(parents=True)
        shutil.copyfile(root / "docs/ml/evaluate_completion.py", helper)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(snapshot / "src"), str(snapshot)))
        result = subprocess.run(
            [sys.executable, str(helper), "--output", str(output)], cwd=snapshot, env=environment, check=False
        )
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
