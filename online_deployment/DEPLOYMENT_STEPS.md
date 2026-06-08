# Online Deployment Steps

## 1. Confirm the Model Artifact

The deployment folder includes:

```text
online_deployment/models/best.pt
online_deployment/models/best.onnx
```

The app loads `models/best.pt` first. The ONNX file is included as deployment evidence and can
be used by setting `MODEL_PATH=models/best.onnx`.

Optional Space environment variables:

- `MODEL_LABEL` (default: `YOLO11l`)
- `DEFAULT_IMGSZ` (default: `1280`)
- `BENCHMARK_MS` (default: `30.7`)
- `STUDENT_NAME`, `INSTITUTION`, `CONTACT_EMAIL`

## 2. Test Locally

From the project root:

```bash
cd online_deployment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL, upload a clean PCB image, and verify that boxes are drawn.
Avoid YOLO validation mosaics, screenshots, or images that already contain filenames/class labels;
those overlays can cause false positives around text.

## 3. Deploy to Hugging Face Spaces

Option A: deploy through the Hugging Face web UI.

1. Create a new Hugging Face Space.
2. Choose **Gradio** as the SDK.
3. Upload everything inside `online_deployment/`.
4. Make sure `models/best.pt` and `models/best.onnx` are included under `models/`.
5. Wait for the Space build to finish.
6. Copy the public Space URL into the final presentation/report.

Option B: deploy with the `hf` CLI.

```bash
hf auth login
hf repo create USERNAME/pcb-defect-detection --type space --space-sdk gradio --public --exist-ok
hf upload USERNAME/pcb-defect-detection online_deployment . --repo-type space
```

Replace `USERNAME` with your Hugging Face username.

## 4. What to Claim

Use this wording:

```text
The trained PCB defect detector was deployed as an online Gradio web demo. The demo accepts
uploaded PCB images and returns annotated defect predictions with confidence scores.
```

Do not claim:

```text
The model was fully deployed and benchmarked on Jetson/TensorRT.
```

unless that benchmark is actually completed.
