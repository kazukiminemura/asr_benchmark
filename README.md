# asr_benchmark

CLI benchmark for microphone ASR inference with Whisper OpenVINO IR models.

## Setup

```powershell
uv sync
```

All examples below use `uv run`, so the CLI can run from the local project
without installing `asr-bench` onto your global `PATH`.

If you want to install the command into the active Python environment:

```powershell
pip install -e .
```

## Prepare Models

Exports `openai/whisper-tiny`, `openai/whisper-base`, and `openai/whisper-small` to Optimum Intel OpenVINO IR directories.

```powershell
uv run asr-bench prepare
```

You can export selected sizes:

```powershell
uv run asr-bench prepare --size tiny --size base
```

If you only prepared `tiny` and `base`, pass those model directories explicitly
when recording. The default `record` command expects `tiny`, `base`, and `small`.

## List Microphones

```powershell
uv run asr-bench devices
```

## Run Benchmark

Records the microphone once into memory, then runs the same audio through each model.

```powershell
uv run asr-bench record --seconds 5
```

By default, `record` uses:

- models: `./models/whisper-tiny`, `./models/whisper-base`, `./models/whisper-small`
- OpenVINO device: `GPU`
- language: `ja`
- task: `transcribe`
- audio: 16 kHz, mono, float32, in memory

Example with explicit models:

```powershell
uv run asr-bench record --seconds 5 `
  --model ./models/whisper-tiny `
  --model ./models/whisper-base
```

Example output:

```text
model=whisper-tiny device=GPU language=ja task=transcribe record=5.012s load=2.301s warmup=0.000s preprocess=0.042s infer=0.821s decode=0.004s total=8.180s text="こんにちは"
```

JSON Lines output:

```powershell
uv run asr-bench record --seconds 5 --jsonl
```

Use another OpenVINO device:

```powershell
uv run asr-bench record --seconds 5 --device CPU
```

Use a specific microphone:

```powershell
uv run asr-bench record --seconds 5 --input-device 1
```
