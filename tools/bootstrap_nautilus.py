#!/usr/bin/env python3
"""Rebuild a wiped Nautilus JupyterLab workspace.

Run this after a Nautilus/Jupyter reset to reinstall Python tools, restore the
PCB dataset, verify the GPU stack, and optionally start a detector training run.
Secrets are read from environment variables or ~/.kaggle files and are never
printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_DEPS = [
    "kaggle",
    "ultralytics==8.4.51",
    "albumentations==2.0.8",
    "ensemble-boxes",
    "pycocotools",
    "onnx",
    "onnxruntime",
    "tqdm",
    "pyyaml",
]


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    local_bin = str(Path.home() / ".local" / "bin")
    env["PATH"] = local_bin + os.pathsep + env.get("PATH", "")
    return env


def quote_cmd(cmd: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run(cmd: list[str | Path], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {quote_cmd(cmd)}", flush=True)
    subprocess.run([str(part) for part in cmd], cwd=cwd, env=env or base_env(), check=True)


def ensure_path_on_shell_startup() -> None:
    line = 'export PATH="$HOME/.local/bin:$PATH"'
    for rc_name in [".bashrc", ".zshrc"]:
        rc_path = Path.home() / rc_name
        current = rc_path.read_text(encoding="utf-8") if rc_path.exists() else ""
        if line not in current:
            with rc_path.open("a", encoding="utf-8") as handle:
                if current and not current.endswith("\n"):
                    handle.write("\n")
                handle.write(f"{line}\n")


def kaggle_bin() -> str:
    found = shutil.which("kaggle", path=base_env().get("PATH"))
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "kaggle"
    return str(fallback)


def configure_kaggle_auth(token_env: str) -> bool:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(mode=0o700, exist_ok=True)
    access_token = kaggle_dir / "access_token"
    kaggle_json = kaggle_dir / "kaggle.json"

    token = os.environ.get(token_env, "").strip()
    if token:
        access_token.write_text(token + "\n", encoding="utf-8")
        access_token.chmod(0o600)
        print(f"Kaggle token loaded from ${token_env} into ~/.kaggle/access_token.")
        return True

    if access_token.exists() and access_token.stat().st_size > 0:
        access_token.chmod(0o600)
        print("Kaggle auth found at ~/.kaggle/access_token.")
        return True

    if kaggle_json.exists() and kaggle_json.stat().st_size > 0:
        kaggle_json.chmod(0o600)
        print("Kaggle auth found at ~/.kaggle/kaggle.json.")
        return True

    print(
        "Kaggle auth was not found. Add ~/.kaggle/access_token or export "
        f"{token_env} before downloading Kaggle data."
    )
    return False


def current_pcb_exists(data_root: Path) -> bool:
    candidates = [
        data_root / "current_pcb" / "PCB-DATASET-master",
        data_root / "PCB-DATASET-master",
    ]
    for root in candidates:
        if (root / "Annotations").is_dir() and (root / "images").is_dir():
            return True
    return False


def install_dependencies() -> None:
    ensure_path_on_shell_startup()
    run([sys.executable, "-m", "pip", "install", "--user", *DEFAULT_DEPS])


def verify_environment() -> None:
    code = (
        "import torch, ultralytics, albumentations\n"
        "print('CUDA:', torch.cuda.is_available())\n"
        "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\n"
        "print('Ultralytics:', ultralytics.__version__)\n"
        "print('Albumentations:', albumentations.__version__)\n"
    )
    run([sys.executable, "-c", code])


def download_dataset(args: argparse.Namespace, data_root: Path) -> None:
    dataset_dir = data_root / "current_pcb"
    if current_pcb_exists(data_root) and not args.force_dataset:
        print(f"Current PCB dataset already exists under {data_root}; skipping download.")
        return
    if not configure_kaggle_auth(args.kaggle_token_env):
        raise RuntimeError("Cannot download the dataset until Kaggle auth is configured.")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        kaggle_bin(),
        "datasets",
        "download",
        "-d",
        args.dataset_slug,
        "--unzip",
        "-p",
        str(dataset_dir),
    ]
    if args.force_dataset:
        cmd.append("--force")
    run(cmd)


def run_smoke(repo_dir: Path, args: argparse.Namespace, data_root: Path, output_dir: Path) -> None:
    runner = repo_dir / "tools" / "run_nautilus_experiments.py"
    run(
        [
            sys.executable,
            runner,
            "--experiment",
            "smoke",
            "--data-root",
            data_root,
            "--output-dir",
            output_dir,
            "--file-mode",
            args.file_mode,
        ],
        cwd=repo_dir,
    )


def download_kaggle_outputs(repo_dir: Path, args: argparse.Namespace) -> None:
    if not configure_kaggle_auth(args.kaggle_token_env):
        raise RuntimeError("Cannot download Kaggle outputs until Kaggle auth is configured.")
    target = expand_path(args.kaggle_output_dir)
    target.mkdir(parents=True, exist_ok=True)
    run([kaggle_bin(), "kernels", "status", args.kernel_slug], cwd=repo_dir)
    run([kaggle_bin(), "kernels", "output", args.kernel_slug, "-p", target], cwd=repo_dir)


def start_detector_train(repo_dir: Path, args: argparse.Namespace, data_root: Path, output_dir: Path) -> int | None:
    logs_dir = expand_path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{args.run_name}.log"
    runner = repo_dir / "tools" / "run_nautilus_experiments.py"
    cmd = [
        sys.executable,
        str(runner),
        "--experiment",
        "detector_train",
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--yolo-model",
        args.yolo_model,
        "--run-name",
        args.run_name,
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--epochs",
        str(args.epochs),
        "--workers",
        str(args.workers),
    ]
    if args.background_train:
        print(f"\nStarting detector training in the background. Log: {log_path}")
        print(f"$ {quote_cmd(cmd)} > {shlex.quote(str(log_path))} 2>&1 &")
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=repo_dir,
            env=base_env(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"Background PID: {process.pid}")
        print(f"Monitor with: tail -f {log_path}")
        return process.pid

    run(cmd, cwd=repo_dir)
    return None


def write_status(output_dir: Path, status: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "bootstrap_status.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"\nBootstrap status written to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default=None, help="Repository path. Defaults to this script's repo.")
    parser.add_argument("--data-root", default="~/data")
    parser.add_argument("--output-dir", default="~/outputs/nautilus")
    parser.add_argument("--logs-dir", default="~/logs")
    parser.add_argument("--file-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--dataset-slug", default="aditya2402/pcb-dataset")
    parser.add_argument("--kernel-slug", default="aditya2402/project")
    parser.add_argument("--kaggle-output-dir", default="~/Capstone_project/kaggle_cli_output/current_artifacts")
    parser.add_argument("--kaggle-token-env", default="KAGGLE_API_TOKEN")
    parser.add_argument("--install-deps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-kaggle-output", action="store_true")
    parser.add_argument("--start-detector-train", action="store_true")
    parser.add_argument("--background-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yolo-model", default="yolo11m.pt")
    parser.add_argument("--run-name", default="yolo11m_publication")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_dir = expand_path(args.repo_dir) if args.repo_dir else Path(__file__).resolve().parents[1]
    data_root = expand_path(args.data_root)
    output_dir = expand_path(args.output_dir)
    status: dict[str, Any] = {
        "started": timestamp(),
        "repo_dir": str(repo_dir),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "steps": [],
    }

    def step(name: str, fn: Any) -> Any:
        started = timestamp()
        print(f"\n=== {name} ===")
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            status["steps"].append({"step": name, "status": "failed", "started": started, "error": repr(exc)})
            status["finished"] = timestamp()
            write_status(output_dir, status)
            raise
        status["steps"].append({"step": name, "status": "completed", "started": started, "finished": timestamp()})
        return result

    data_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.install_deps:
        step("install_dependencies", install_dependencies)
    step("verify_environment", verify_environment)
    if args.download_dataset:
        step("download_dataset", lambda: download_dataset(args, data_root))
    if args.smoke:
        step("smoke_check", lambda: run_smoke(repo_dir, args, data_root, output_dir))
    if args.download_kaggle_output:
        step("download_kaggle_outputs", lambda: download_kaggle_outputs(repo_dir, args))
    if args.start_detector_train:
        pid = step("start_detector_train", lambda: start_detector_train(repo_dir, args, data_root, output_dir))
        status["detector_train_pid"] = pid

    status["finished"] = timestamp()
    write_status(output_dir, status)
    print("\nDone. After a reset, this script is the repeatable recovery path.")


if __name__ == "__main__":
    main()
