from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console


app = typer.Typer(help="Benchmark microphone ASR inference with Whisper OpenVINO IR models.")
console = Console()

DEFAULT_MODEL_ROOT = Path("models")
DEFAULT_SIZES = ("tiny", "base", "small")
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DEVICE = "GPU"
DEFAULT_LANGUAGE = "ja"
DEFAULT_TASK = "transcribe"


@dataclass(frozen=True)
class Recording:
    samples: object
    sample_rate: int
    seconds: float
    elapsed: float


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    device: str
    language: str
    task: str
    record: float
    load: float
    warmup: float
    preprocess: float
    infer: float
    decode: float
    total: float
    text: str


def _format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def _model_name(path: Path) -> str:
    return path.name or str(path)


def _default_model_paths() -> list[Path]:
    return [DEFAULT_MODEL_ROOT / f"whisper-{size}" for size in DEFAULT_SIZES]


def _require_model_dirs(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return

    for path in missing:
        console.print(f"[red]missing model:[/red] {path}")
        name = path.name.removeprefix("whisper-")
        if name in DEFAULT_SIZES:
            console.print(f"hint: run `asr-bench prepare --size {name}`")
    raise typer.Exit(code=2)


def _import_dependency(module: str, package: str | None = None) -> object:
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module.split(".")[0]:
            raise
        install_name = package or module
        console.print(f"[red]missing dependency:[/red] {install_name}")
        console.print("Install dependencies first, for example: `uv sync` or `pip install -e .`")
        raise typer.Exit(code=2) from exc


def _parse_input_device(input_device: str | None) -> int | str | None:
    if input_device is None:
        return None
    try:
        return int(input_device)
    except ValueError:
        return input_device


def _record_microphone(seconds: float, sample_rate: int, input_device: int | str | None) -> Recording:
    np = _import_dependency("numpy")
    sd = _import_dependency("sounddevice")

    frames = int(round(seconds * sample_rate))
    if frames <= 0:
        raise typer.BadParameter("--seconds must be greater than 0")

    started = time.perf_counter()
    audio = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=input_device,
    )
    sd.wait()
    elapsed = time.perf_counter() - started
    return Recording(
        samples=np.asarray(audio, dtype=np.float32).reshape(-1),
        sample_rate=sample_rate,
        seconds=seconds,
        elapsed=elapsed,
    )


def _legacy_generate_kwargs(processor: object, language: str, task: str) -> dict[str, object]:
    get_decoder_prompt_ids = getattr(processor, "get_decoder_prompt_ids", None)
    if callable(get_decoder_prompt_ids):
        try:
            return {"forced_decoder_ids": get_decoder_prompt_ids(language=language, task=task)}
        except TypeError:
            pass
    return {}


def _generate(model: object, input_features: object, processor: object, language: str, task: str) -> object:
    try:
        return model.generate(input_features, language=language, task=task)
    except TypeError:
        legacy_kwargs = _legacy_generate_kwargs(processor, language, task)
        return model.generate(input_features, **legacy_kwargs)


def _run_model(
    model_dir: Path,
    recording: Recording,
    device: str,
    language: str,
    task: str,
    warmup: int,
) -> BenchmarkResult:
    openvino_module = _import_dependency("optimum.intel.openvino", "optimum-intel[openvino]")
    transformers_module = _import_dependency("transformers")
    OVModelForSpeechSeq2Seq = openvino_module.OVModelForSpeechSeq2Seq
    AutoProcessor = transformers_module.AutoProcessor

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_dir)
    model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, device=device)
    load_elapsed = time.perf_counter() - load_started

    preprocess_started = time.perf_counter()
    inputs = processor(
        recording.samples,
        sampling_rate=recording.sample_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features
    preprocess_elapsed = time.perf_counter() - preprocess_started

    warmup_elapsed = 0.0
    for _ in range(warmup):
        warmup_started = time.perf_counter()
        _generate(model, input_features, processor, language, task)
        warmup_elapsed += time.perf_counter() - warmup_started

    infer_started = time.perf_counter()
    predicted_ids = _generate(model, input_features, processor, language, task)
    infer_elapsed = time.perf_counter() - infer_started

    decode_started = time.perf_counter()
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    decode_elapsed = time.perf_counter() - decode_started

    total = (
        recording.elapsed
        + load_elapsed
        + warmup_elapsed
        + preprocess_elapsed
        + infer_elapsed
        + decode_elapsed
    )
    return BenchmarkResult(
        model=_model_name(model_dir),
        device=device,
        language=language,
        task=task,
        record=recording.elapsed,
        load=load_elapsed,
        warmup=warmup_elapsed,
        preprocess=preprocess_elapsed,
        infer=infer_elapsed,
        decode=decode_elapsed,
        total=total,
        text=text,
    )


def _print_result(result: BenchmarkResult, jsonl: bool) -> None:
    if jsonl:
        console.print(json.dumps(asdict(result), ensure_ascii=False))
        return

    console.print(
        " ".join(
            [
                f"model={result.model}",
                f"device={result.device}",
                f"language={result.language}",
                f"task={result.task}",
                f"record={_format_seconds(result.record)}",
                f"load={_format_seconds(result.load)}",
                f"warmup={_format_seconds(result.warmup)}",
                f"preprocess={_format_seconds(result.preprocess)}",
                f"infer={_format_seconds(result.infer)}",
                f"decode={_format_seconds(result.decode)}",
                f"total={_format_seconds(result.total)}",
                f'text="{result.text}"',
            ]
        )
    )


@app.command()
def prepare(
    size: Annotated[
        list[str] | None,
        typer.Option("--size", "-s", help="Whisper model size to export. Can be repeated."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory where OpenVINO IR models are written."),
    ] = DEFAULT_MODEL_ROOT,
) -> None:
    """Export Whisper models to Optimum Intel OpenVINO IR directories."""

    sizes = size or list(DEFAULT_SIZES)
    optimum_cli = shutil.which("optimum-cli")
    if optimum_cli is None:
        console.print("[red]optimum-cli was not found.[/red]")
        console.print("Install dependencies first, for example: `uv sync` or `pip install -e .`")
        raise typer.Exit(code=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    for model_size in sizes:
        model_id = f"openai/whisper-{model_size}"
        target = output_dir / f"whisper-{model_size}"
        console.print(f"exporting {model_id} -> {target}")
        subprocess.run(
            [
                optimum_cli,
                "export",
                "openvino",
                "--model",
                model_id,
                str(target),
            ],
            check=True,
        )


@app.command()
def devices() -> None:
    """List available audio input devices."""

    sd = _import_dependency("sounddevice")

    for index, device in enumerate(sd.query_devices()):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        console.print(f"{index}: {device['name']}")


@app.command()
def record(
    seconds: Annotated[
        float,
        typer.Option("--seconds", help="Seconds to record from the microphone."),
    ] = 5.0,
    model: Annotated[
        list[Path] | None,
        typer.Option("--model", "-m", help="Optimum Intel OpenVINO IR model directory. Can be repeated."),
    ] = None,
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="OpenVINO device, for example GPU, CPU, AUTO, or NPU."),
    ] = DEFAULT_DEVICE,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Whisper language code."),
    ] = DEFAULT_LANGUAGE,
    task: Annotated[
        str,
        typer.Option("--task", help="Whisper task."),
    ] = DEFAULT_TASK,
    input_device: Annotated[
        str | None,
        typer.Option("--input-device", help="Audio input device index or name."),
    ] = None,
    sample_rate: Annotated[
        int,
        typer.Option("--sample-rate", help="Microphone sample rate."),
    ] = DEFAULT_SAMPLE_RATE,
    warmup: Annotated[
        int,
        typer.Option("--warmup", help="Warmup generations per model. Included in total."),
    ] = 0,
    jsonl: Annotated[
        bool,
        typer.Option("--jsonl", help="Print one JSON object per model."),
    ] = False,
) -> None:
    """Record microphone audio once and benchmark it against one or more models."""

    if warmup < 0:
        raise typer.BadParameter("--warmup must be 0 or greater")

    model_paths = model or _default_model_paths()
    _require_model_dirs(model_paths)

    recording = _record_microphone(seconds, sample_rate, _parse_input_device(input_device))
    for model_dir in model_paths:
        result = _run_model(
            model_dir=model_dir,
            recording=recording,
            device=device,
            language=language,
            task=task,
            warmup=warmup,
        )
        _print_result(result, jsonl=jsonl)


if __name__ == "__main__":
    app()
