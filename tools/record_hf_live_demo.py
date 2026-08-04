#!/usr/bin/env python3
"""Record a short live Hugging Face Space demo for the ESCS presentation."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
URL = "https://adiivd-pcb-defect-detection.hf.space"
OUT_DIR = ROOT / "reports" / "presentation" / "demo_video"
VIDEO_DIR = OUT_DIR / "playwright_video"
EXAMPLE_IMAGE = ROOT / "online_deployment" / "examples" / "clean_open_circuit.jpg"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def wait_soft(page, ms: int) -> None:
    page.wait_for_timeout(ms)


def scroll_to(page, y: int) -> None:
    page.evaluate("(y) => window.scrollTo({ top: y, behavior: 'smooth' })", y)
    wait_soft(page, 900)


def save_screenshot(page, name: str) -> None:
    page.screenshot(path=str(OUT_DIR / name), full_page=False)


def wait_for_space_ready(page) -> None:
    page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
    try:
        page.wait_for_load_state("networkidle", timeout=45_000)
    except PlaywrightTimeoutError:
        # Gradio often keeps a websocket open; visible UI is enough for recording.
        pass
    page.wait_for_selector("text=Automated PCB Defect Detection", timeout=120_000)
    wait_soft(page, 2_500)


def upload_demo_image(page) -> None:
    upload_inputs = page.locator("input[type='file']")
    upload_inputs.first.set_input_files(str(EXAMPLE_IMAGE))
    wait_soft(page, 2_500)


def run_detection(page) -> None:
    button = page.get_by_role("button", name=re.compile("Run detection", re.I))
    if button.count() == 0:
        button = page.locator("button:has-text('Run detection')")
    button.first.click(timeout=30_000)

    for _ in range(48):
        text = page.locator("body").inner_text(timeout=10_000)
        has_final_summary = (
            "candidate(s)" in text
            or "Top classes:" in text
            or "No detections above threshold" in text
        )
        if has_final_summary and "processing |" not in text:
            return
        wait_soft(page, 2_500)
    raise RuntimeError("Timed out waiting for detection output.")


def convert_webm_to_mp4(webm_path: Path, mp4_path: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(webm_path),
            "-vf",
            "scale=1280:-2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ],
        check=True,
    )


def main() -> None:
    if not EXAMPLE_IMAGE.exists():
        raise FileNotFoundError(EXAMPLE_IMAGE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if VIDEO_DIR.exists():
        shutil.rmtree(VIDEO_DIR)
    VIDEO_DIR.mkdir(parents=True)

    with sync_playwright() as playwright:
        launch_kwargs = {
            "headless": True,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
        }
        if CHROME.exists():
            launch_kwargs["executable_path"] = str(CHROME)

        browser = playwright.chromium.launch(**launch_kwargs)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()

        wait_for_space_ready(page)
        save_screenshot(page, "01_loaded.png")

        scroll_to(page, 320)
        upload_demo_image(page)
        save_screenshot(page, "02_uploaded.png")

        wait_soft(page, 900)
        run_detection(page)
        wait_soft(page, 1_500)
        save_screenshot(page, "03_detection_result.png")

        scroll_to(page, 650)
        save_screenshot(page, "04_result_details.png")

        scroll_to(page, 980)
        save_screenshot(page, "05_downloads_and_samples.png")
        wait_soft(page, 2_000)

        context.close()
        browser.close()

    video_files = sorted(VIDEO_DIR.glob("*.webm"), key=lambda path: path.stat().st_mtime)
    if not video_files:
        raise RuntimeError("Playwright did not create a recorded video.")

    webm_path = OUT_DIR / "hf_live_demo.webm"
    mp4_path = OUT_DIR / "hf_live_demo.mp4"
    shutil.copyfile(video_files[-1], webm_path)
    convert_webm_to_mp4(webm_path, mp4_path)

    print(mp4_path)


if __name__ == "__main__":
    main()
