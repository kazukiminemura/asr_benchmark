from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Annotated
from urllib.request import urlretrieve

import typer
from rich.console import Console


app = typer.Typer(help="Benchmark microphone ASR inference with Whisper OpenVINO IR models.")
console = Console()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MODEL_ROOT = Path("models")
DEFAULT_SIZES = ("tiny", "base", "small")
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DEVICE = "GPU"
DEFAULT_LANGUAGE = "ja"
DEFAULT_TASK = "transcribe"
DEFAULT_PRECISION = "int8"
QWEN_HELPER_URL = (
    "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/"
    "latest/notebooks/qwen3-asr/qwen_3_asr_helper.py"
)
QWEN_ASR_MODEL_IDS = (
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
)
QWEN_ASR_PRECISIONS = ("fp16", "int8", "int4")
KOTOBA_FASTER_MODEL_ID = "RoachLin/kotoba-whisper-v2.2-faster"
KOTOBA_SOURCE_MODEL_ID = "kotoba-tech/kotoba-whisper-v2.2"
KOTOBA_PRECISIONS = ("fp32", "fp16")
DEFAULT_MODEL_IDS = (
    "OpenVINO/whisper-tiny-int8-ov",
    "OpenVINO/whisper-base-int8-ov",
    "OpenVINO/whisper-small-int8-ov",
    "OpenVINO/whisper-large-v3-turbo-int4-ov",
    "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "OpenVINO/whisper-large-v3-turbo-fp16-ov",
    "OpenVINO/whisper-large-v3-int4-ov",
    "OpenVINO/whisper-large-v3-int8-ov",
    "OpenVINO/whisper-large-v3-fp16-ov",
)
COLLECTION_WHISPER_MODEL_IDS = (
    "OpenVINO/whisper-large-v3-turbo-int4-ov",
    "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "OpenVINO/whisper-large-v3-turbo-fp16-ov",
    "OpenVINO/whisper-large-v3-int4-ov",
    "OpenVINO/whisper-large-v3-int8-ov",
    "OpenVINO/whisper-large-v3-fp16-ov",
    "OpenVINO/whisper-medium-int8-ov",
    "OpenVINO/whisper-medium-int4-ov",
    "OpenVINO/whisper-medium-fp16-ov",
    "OpenVINO/whisper-tiny-int4-ov",
    "OpenVINO/whisper-tiny-int8-ov",
    "OpenVINO/whisper-tiny-fp16-ov",
    "OpenVINO/whisper-base-int4-ov",
    "OpenVINO/whisper-base-int8-ov",
    "OpenVINO/whisper-base-fp16-ov",
    "OpenVINO/whisper-small-int4-ov",
    "OpenVINO/whisper-small-int8-ov",
    "OpenVINO/whisper-small-fp16-ov",
    "OpenVINO/whisper-medium.en-int4-ov",
    "OpenVINO/whisper-medium.en-int8-ov",
    "OpenVINO/whisper-medium.en-fp16-ov",
    "OpenVINO/whisper-tiny.en-int4-ov",
    "OpenVINO/whisper-tiny.en-int8-ov",
    "OpenVINO/whisper-tiny.en-fp16-ov",
    "OpenVINO/whisper-base.en-int4-ov",
    "OpenVINO/whisper-base.en-int8-ov",
    "OpenVINO/whisper-base.en-fp16-ov",
    "OpenVINO/whisper-small.en-int4-ov",
    "OpenVINO/whisper-small.en-int8-ov",
    "OpenVINO/whisper-small.en-fp16-ov",
)


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
    total_scope: str
    record: float
    load: float
    warmup: float
    preprocess: float
    infer: float
    decode: float
    total: float
    text: str


def _qwen_model_dir_name(model_id: str, precision: str) -> str:
    return f"{model_id.split('/')[-1].lower()}-{precision}-ov"


def _kotoba_model_dir_name(model_id: str, precision: str) -> str:
    return f"{model_id.split('/')[-1]}-{precision}-ov"


def _format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def _model_name(path: Path) -> str:
    return path.name or str(path)


def _model_id_to_dir_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def _model_id_for_size(size: str, precision: str) -> str:
    return f"OpenVINO/whisper-{size}-{precision}-ov"


def _default_model_paths() -> list[Path]:
    return [DEFAULT_MODEL_ROOT / _model_id_to_dir_name(model_id) for model_id in DEFAULT_MODEL_IDS]


def _require_model_dirs(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    for path in missing:
        console.print(f"[red]missing model:[/red] {path}")
        console.print("hint: run `asr-bench prepare` or pass existing model directories with `--model`")

    incomplete = []
    for path in paths:
        if not path.exists():
            continue
        for xml_path in path.glob("*.xml"):
            bin_path = xml_path.with_suffix(".bin")
            if not bin_path.exists():
                incomplete.append((path, bin_path.name))

    for path, file_name in incomplete:
        console.print(f"[red]incomplete model:[/red] {path} is missing {file_name}")
        console.print(f"hint: run `asr-bench prepare --model-id OpenVINO/{path.name}`")

    if missing or incomplete:
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


def _load_qwen_helper() -> ModuleType:
    helper_dir = Path(".cache") / "asr_benchmark"
    helper_path = helper_dir / "qwen_3_asr_helper.py"
    if not helper_path.exists():
        helper_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"downloading Qwen3-ASR OpenVINO helper -> {helper_path}")
        urlretrieve(QWEN_HELPER_URL, helper_path)

    spec = spec_from_file_location("asr_benchmark_qwen_3_asr_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Qwen helper from {helper_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qwen_language(language: str | None) -> str | None:
    if language is None or language.lower() in {"auto", "none"}:
        return None
    return {
        "ja": "Japanese",
        "jp": "Japanese",
        "japanese": "Japanese",
        "en": "English",
        "english": "English",
        "zh": "Chinese",
        "cn": "Chinese",
        "chinese": "Chinese",
    }.get(language.lower(), language)


def _clean_qwen_text(text: str) -> str:
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return text.replace("</asr_text>", "").strip()


def _qwen_quantization_config(precision: str) -> dict[str, object] | None:
    if precision == "fp16":
        return None

    nncf = _import_dependency("nncf")
    modes = {
        "int8": nncf.CompressWeightsMode.INT8_SYM,
        "int4": nncf.CompressWeightsMode.INT4_SYM,
    }
    if precision not in modes:
        raise typer.BadParameter(f"Unsupported Qwen precision: {precision}")
    return {"mode": modes[precision]}


def _resolve_kotoba_source_model_id(faster_model_id: str, fallback_model_id: str) -> str:
    huggingface_hub = _import_dependency("huggingface_hub", "huggingface-hub")
    try:
        card_data = huggingface_hub.model_info(faster_model_id).card_data
    except Exception as exc:
        console.print(f"[yellow]warning:[/yellow] could not read {faster_model_id} model card: {exc}")
        return fallback_model_id

    base_model = getattr(card_data, "base_model", None)
    if isinstance(base_model, str):
        return base_model
    if isinstance(base_model, list) and base_model:
        return str(base_model[0])
    return fallback_model_id


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
    exclude_record_load: bool,
    exclude_warmup: bool,
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

    measured_total = preprocess_elapsed + infer_elapsed + decode_elapsed
    if not exclude_warmup:
        measured_total += warmup_elapsed

    excluded = []
    if exclude_record_load:
        excluded.extend(["record", "load"])
    if exclude_warmup:
        excluded.append("warmup")
    total_scope = "all" if not excluded else f"exclude-{','.join(excluded)}"
    total = measured_total if exclude_record_load else recording.elapsed + load_elapsed + measured_total
    return BenchmarkResult(
        model=_model_name(model_dir),
        device=device,
        language=language,
        task=task,
        total_scope=total_scope,
        record=recording.elapsed,
        load=load_elapsed,
        warmup=warmup_elapsed,
        preprocess=preprocess_elapsed,
        infer=infer_elapsed,
        decode=decode_elapsed,
        total=total,
        text=text,
    )


def _run_qwen_model(
    model_dir: Path,
    recording: Recording,
    device: str,
    language: str | None,
    warmup: int,
    max_new_tokens: int,
    max_inference_batch_size: int,
    exclude_record_load: bool,
    exclude_warmup: bool,
) -> BenchmarkResult:
    helper = _load_qwen_helper()

    load_started = time.perf_counter()
    model = helper.OVQwen3ASRModel.from_pretrained(
        model_dir=str(model_dir),
        device=device,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
    )
    load_elapsed = time.perf_counter() - load_started

    audio = (recording.samples, recording.sample_rate)
    qwen_language = _qwen_language(language)

    warmup_elapsed = 0.0
    for _ in range(warmup):
        warmup_started = time.perf_counter()
        model.transcribe(audio=audio, language=qwen_language)
        warmup_elapsed += time.perf_counter() - warmup_started

    infer_started = time.perf_counter()
    results = model.transcribe(audio=audio, language=qwen_language)
    infer_elapsed = time.perf_counter() - infer_started

    result = results[0]
    text = _clean_qwen_text(getattr(result, "text", str(result)))
    detected_language = getattr(result, "language", None)
    output_language = detected_language or qwen_language or "auto"

    measured_total = infer_elapsed
    if not exclude_warmup:
        measured_total += warmup_elapsed

    excluded = []
    if exclude_record_load:
        excluded.extend(["record", "load"])
    if exclude_warmup:
        excluded.append("warmup")
    total_scope = "all" if not excluded else f"exclude-{','.join(excluded)}"
    total = measured_total if exclude_record_load else recording.elapsed + load_elapsed + measured_total

    return BenchmarkResult(
        model=_model_name(model_dir),
        device=device,
        language=output_language,
        task="qwen-transcribe",
        total_scope=total_scope,
        record=recording.elapsed,
        load=load_elapsed,
        warmup=warmup_elapsed,
        preprocess=0.0,
        infer=infer_elapsed,
        decode=0.0,
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
                f"total_scope={result.total_scope}",
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
        typer.Option("--size", "-s", help="Whisper model size to download. Can be repeated."),
    ] = None,
    precision: Annotated[
        str,
        typer.Option("--precision", "-p", help="OpenVINO model precision for size-based downloads."),
    ] = DEFAULT_PRECISION,
    model_id: Annotated[
        list[str] | None,
        typer.Option("--model-id", help="Exact Hugging Face model id to download. Can be repeated."),
    ] = None,
    all_whisper: Annotated[
        bool,
        typer.Option("--all-whisper", help="Download every OpenVINO/whisper-* model from the collection."),
    ] = False,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory where OpenVINO IR models are downloaded."),
    ] = DEFAULT_MODEL_ROOT,
) -> None:
    """Download OpenVINO Whisper IR models from Hugging Face."""

    huggingface_hub = _import_dependency("huggingface_hub", "huggingface-hub")
    snapshot_download = huggingface_hub.snapshot_download

    if all_whisper:
        model_ids = list(COLLECTION_WHISPER_MODEL_IDS)
    elif model_id:
        model_ids = model_id
    elif size is None:
        model_ids = list(DEFAULT_MODEL_IDS)
    else:
        model_ids = [_model_id_for_size(model_size, precision) for model_size in size]

    output_dir.mkdir(parents=True, exist_ok=True)
    for hf_model_id in model_ids:
        target = output_dir / _model_id_to_dir_name(hf_model_id)
        console.print(f"downloading {hf_model_id} -> {target}")
        snapshot_download(repo_id=hf_model_id, local_dir=target)


@app.command()
def prepare_qwen(
    model_id: Annotated[
        list[str] | None,
        typer.Option("--model-id", help="Qwen3-ASR Hugging Face model id. Can be repeated."),
    ] = None,
    precision: Annotated[
        list[str] | None,
        typer.Option("--precision", "-p", help="Precision to export: fp16, int8, or int4. Can be repeated."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory where Qwen OpenVINO IR models are written."),
    ] = DEFAULT_MODEL_ROOT,
) -> None:
    """Convert Qwen3-ASR models to OpenVINO IR for fp16/int8/int4 testing."""

    model_ids = model_id or list(QWEN_ASR_MODEL_IDS)
    precisions = precision or list(QWEN_ASR_PRECISIONS)
    helper = _load_qwen_helper()
    output_dir.mkdir(parents=True, exist_ok=True)

    for hf_model_id in model_ids:
        for fmt in precisions:
            if fmt not in QWEN_ASR_PRECISIONS:
                raise typer.BadParameter(f"Unsupported Qwen precision: {fmt}")
            target = output_dir / _qwen_model_dir_name(hf_model_id, fmt)
            console.print(f"converting {hf_model_id} ({fmt}) -> {target}")
            helper.convert_qwen3_asr_model(
                model_id=hf_model_id,
                output_dir=target,
                quantization_config=_qwen_quantization_config(fmt),
            )


@app.command()
def prepare_kotoba(
    faster_model_id: Annotated[
        str,
        typer.Option(
            "--faster-model-id",
            help="CTranslate2/faster-whisper model id whose base_model metadata is used.",
        ),
    ] = KOTOBA_FASTER_MODEL_ID,
    source_model_id: Annotated[
        str | None,
        typer.Option(
            "--source-model-id",
            help="Transformers Whisper model id to export. Defaults to the faster model's base_model.",
        ),
    ] = None,
    precision: Annotated[
        list[str] | None,
        typer.Option("--precision", "-p", help="OpenVINO IR precision: fp32 or fp16. Can be repeated."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory where Kotoba OpenVINO IR models are written."),
    ] = DEFAULT_MODEL_ROOT,
    include_source_alias: Annotated[
        bool,
        typer.Option(
            "--include-source-alias/--no-source-alias",
            help="Also save the same IR under the source model name, for example kotoba-whisper-v2.2-fp16-ov.",
        ),
    ] = True,
    trust_remote_code: Annotated[
        bool,
        typer.Option("--trust-remote-code/--no-trust-remote-code", help="Allow custom code from the source repo."),
    ] = True,
) -> None:
    """Convert RoachLin/kotoba-whisper-v2.2-faster's source Whisper model to OpenVINO IR."""

    precisions = precision or ["fp16"]
    for fmt in precisions:
        if fmt not in KOTOBA_PRECISIONS:
            raise typer.BadParameter(f"Unsupported Kotoba precision: {fmt}")

    source_id = source_model_id or _resolve_kotoba_source_model_id(
        faster_model_id=faster_model_id,
        fallback_model_id=KOTOBA_SOURCE_MODEL_ID,
    )
    optimum_export = _import_dependency("optimum.exporters.openvino.__main__", "optimum-intel[openvino]")
    openvino_config = _import_dependency("optimum.intel.openvino", "optimum-intel[openvino]")
    main_export = optimum_export.main_export
    OVConfig = openvino_config.OVConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in precisions:
        target = output_dir / _kotoba_model_dir_name(faster_model_id, fmt)
        console.print(f"converting {source_id} ({fmt}) -> {target}")
        main_export(
            model_name_or_path=source_id,
            output=target,
            task="automatic-speech-recognition",
            trust_remote_code=trust_remote_code,
            ov_config=OVConfig(dtype=fmt),
        )
        source_alias = output_dir / _kotoba_model_dir_name(source_id, fmt)
        if include_source_alias and source_alias != target:
            shutil = _import_dependency("shutil")
            if source_alias.exists():
                console.print(f"source alias already exists: {source_alias}")
            else:
                console.print(f"copying source alias -> {source_alias}")
                shutil.copytree(target, source_alias)


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
    exclude_record_load: Annotated[
        bool,
        typer.Option(
            "--exclude-record-load",
            help="Exclude microphone recording and model loading from total.",
        ),
    ] = False,
    exclude_warmup: Annotated[
        bool,
        typer.Option("--exclude-warmup", help="Exclude warmup generations from total."),
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
            exclude_record_load=exclude_record_load,
            exclude_warmup=exclude_warmup,
        )
        _print_result(result, jsonl=jsonl)


@app.command()
def record_qwen(
    seconds: Annotated[
        float,
        typer.Option("--seconds", help="Seconds to record from the microphone."),
    ] = 5.0,
    model: Annotated[
        list[Path] | None,
        typer.Option("--model", "-m", help="Qwen3-ASR OpenVINO IR model directory. Can be repeated."),
    ] = None,
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="OpenVINO device, for example GPU, CPU, AUTO, or NPU."),
    ] = DEFAULT_DEVICE,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Language name/code, or auto."),
    ] = DEFAULT_LANGUAGE,
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
        typer.Option("--warmup", help="Warmup transcriptions per model. Included in total."),
    ] = 0,
    max_new_tokens: Annotated[
        int,
        typer.Option("--max-new-tokens", help="Maximum tokens generated by Qwen3-ASR."),
    ] = 256,
    max_inference_batch_size: Annotated[
        int,
        typer.Option("--max-inference-batch-size", help="Qwen3-ASR inference batch limit."),
    ] = 32,
    jsonl: Annotated[
        bool,
        typer.Option("--jsonl", help="Print one JSON object per model."),
    ] = False,
    exclude_record_load: Annotated[
        bool,
        typer.Option(
            "--exclude-record-load",
            help="Exclude microphone recording and model loading from total.",
        ),
    ] = False,
    exclude_warmup: Annotated[
        bool,
        typer.Option("--exclude-warmup", help="Exclude warmup transcriptions from total."),
    ] = False,
) -> None:
    """Record once and benchmark Qwen3-ASR OpenVINO IR models."""

    if warmup < 0:
        raise typer.BadParameter("--warmup must be 0 or greater")

    model_paths = model or [
        DEFAULT_MODEL_ROOT / _qwen_model_dir_name(model_id, precision)
        for model_id in QWEN_ASR_MODEL_IDS
        for precision in QWEN_ASR_PRECISIONS
    ]
    _require_model_dirs(model_paths)

    recording = _record_microphone(seconds, sample_rate, _parse_input_device(input_device))
    for model_dir in model_paths:
        result = _run_qwen_model(
            model_dir=model_dir,
            recording=recording,
            device=device,
            language=language,
            warmup=warmup,
            max_new_tokens=max_new_tokens,
            max_inference_batch_size=max_inference_batch_size,
            exclude_record_load=exclude_record_load,
            exclude_warmup=exclude_warmup,
        )
        _print_result(result, jsonl=jsonl)


if __name__ == "__main__":
    app()
