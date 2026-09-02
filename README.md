# SatQuery AI — SENTRY

SatQuery AI is an evidence-first, agentic vision-language assistant for remote-sensing imagery. It accepts a single optical/SAR image, a co-registered optical–SAR pair, or a bi-temporal pair and routes a natural-language question to the appropriate specialist workflow.

## Start here

The complete project explanation and operator guide is here:

[`docs/SatQuery_AI_Operator_and_Implementation_Handbook.docx`](docs/SatQuery_AI_Operator_and_Implementation_Handbook.docx)

It covers:

- What the problem statement asks for
- What SatQuery AI does and what is innovative about it
- The architecture and agentic orchestration flow
- Every UI control and expected result
- Local run and API testing instructions
- Dataset and benchmark mapping
- Current prototype boundaries
- Deployment and GitHub workflows
- Production implementation roadmap
- Evaluation criteria and troubleshooting

## Run locally

```powershell
Set-Location D:\satquery-ai
npm install
npm run dev
```

Open <http://localhost:3000>.

Validation commands:

```powershell
npm run lint
npm run build
```

## GPU inference and training

The Python service is now implemented under `inference/` and uses the CUDA device when available. Keep datasets and checkpoints outside Git:

```powershell
Set-Location D:\satquery-ai
& .\.venv\Scripts\Activate.ps1
uvicorn inference.api:app --host 127.0.0.1 --port 8000
```

Set `SATQUERY_INFERENCE_URL=http://127.0.0.1:8000` for the Next API proxy. Check `/health` before sending a request; `runtime=real` means a checkpoint is loaded and uploaded pixels are processed.

### Local Qwen2.5-VL summaries

If Ollama is installed, SatQuery uses `qwen2.5vl:3b` as a second-stage vision-language report writer. Install the model with `ollama pull qwen2.5vl:3b`. Qwen receives preview images and the measured evidence JSON, but it does not control masks, percentages, modality decisions, confidence, or the final evidence contract. If Ollama is unavailable or times out, SatQuery uses its grounded template summary. Override `SATQUERY_OLLAMA_MODEL` or `SATQUERY_OLLAMA_URL` when needed; `/health` reports `ollama_ready`.

### Observation distinction

Each uploaded file has an `AUTO DETECT` / `OPTICAL` / `SAR` selector and the console accepts up to four observations. Auto detection uses sensor tokens in the filename and band structure. Use the selector for generic one- or two-band GeoTIFFs; the service rejects ambiguous analytic pairs and rejects fusion unless there is at least one optical and one SAR observation. Fusion also applies SAR log preprocessing and reports modality classification in the execution trace. Multi-observation uploads are catalogued unless the query explicitly asks for change or sensor fusion.

The first checkpoint can be reproduced with:

```powershell
& .\.venv\Scripts\python.exe -m training.train_pixel_scene --epochs 3 --samples 128 --size 128 --batch-size 4
```

That command is a GPU smoke baseline. For actual BigEarthNet.txt adaptation, download a small public annotation slice, place the corresponding Sentinel-1/Sentinel-2 patches under `D:\satquery-data`, prepare a JSONL manifest, validate it, then add the dataset-backed training stage. The public annotation viewer reports 464,044 co-registered S1/S2 scenes and about 9.6M text annotations; it is not safe or necessary to download the complete corpus for the first workstation test.

For a capped real-data adaptation that fits this workstation, use EuroSAT. It downloads a small RGB remote-sensing archive, weakly maps scene labels into water/built-up/other masks, and writes the same checkpoint consumed by inference:

```powershell
& .\.venv\Scripts\python.exe -m training.train_eurosat --limit 4000 --epochs 2 --batch-size 8
```

This is a genuine open-data adaptation step, but it is not a substitute for the paired Sentinel-1/Sentinel-2 BigEarthNet.txt experiment; its weak-label limitation is recorded in the checkpoint version and should be disclosed in evaluation.

### Bounded paired BigEarthNet S1/S2 subset

The paired Hugging Face export can also be sampled without downloading its full archive. In this workstation run, a strictly capped 1 GB byte prefix produced 76 complete paired validation samples after extraction, containing real 2-band Sentinel-1 VV/VH and 12-band Sentinel-2 TIFFs. BigEarthNet uses a multi-page TIFF layout, so SatQuery includes a `tifffile` fallback that reads these as `C x H x W` instead of misreading them as `120 x 2` or `120 x 12` images.

To reproduce the bounded fine-tune after obtaining a capped prefix:

```powershell
& .\.venv\Scripts\python.exe -m training.extract_bigearthnet_prefix D:\satquery-data\bigearthnet-s1s2-prefix-1gb-capped.bin --output D:\satquery-data\bigearthnet-subset-1gb
& .\.venv\Scripts\python.exe -m training.train_bigearthnet --limit 76 --epochs 4 --batch-size 8
```

The paired candidate is saved separately as `D:\satquery-checkpoints\satquery-pixel-bigearthnet-s1s2.pt`; it is not automatically promoted over the EuroSAT checkpoint unless its held-out evaluation is stronger. This prevents a tiny, split-limited sample from reducing general upload quality.

Run the local regression tests with:

```powershell
$env:PATH = "D:\satquery-ai\.venv\Lib\site-packages\torch\lib;$env:PATH"
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The data utilities are:

```powershell
& .\.venv\Scripts\python.exe -m training.download_bigearthnet_metadata --length 1000
& .\.venv\Scripts\python.exe -m training.prepare_manifest --annotations D:\satquery-data\manifests\bigearthnet_annotations.jsonl --image-root D:\satquery-data\bigearthnet --output D:\satquery-data\manifests\bigearthnet.jsonl --limit 1000
& .\.venv\Scripts\python.exe -m training.validate_dataset D:\satquery-data\manifests\bigearthnet.jsonl
& .\.venv\Scripts\python.exe -m training.train_image_text D:\satquery-data\manifests\bigearthnet.jsonl --limit 1000
```

The image-text command requires local image paths in the prepared manifest; it refuses to train on annotation-only rows.

## Current demo

The public shareable demo is deployed at <https://satquery-ai-public.vercel.app>.

The owner-only Sites deployment is also available at <https://satquery-sentry.shrutikaverma22.chatgpt.site> for internal review.

The UI uses a black-and-white brutalist control system and a full-colour satellite-style evidence basemap. With `SATQUERY_INFERENCE_URL` configured, uploads are analyzed by the local CUDA service, which returns model-backed scene context, sensor-specific evidence masks, modality checks, confidence, and a grounded plain-language summary. Without that variable, the Next route remains an explicit deterministic demo fallback.

## Main implementation files

- `app/satquery-console.tsx` — interactive console, cases, modes, upload handling and report download.
- `app/api/analyse/route.ts` — multipart validation, TIFF/PNG/JPEG inspection and task routing.
- `app/globals.css` — interface styling and evidence map layers.
- `public/evidence-map.png` — satellite-style evidence basemap.
- `docs/SatQuery_AI_Operator_and_Implementation_Handbook.docx` — full documentation.
