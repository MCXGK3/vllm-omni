"""Request execution tracing and offline Gantt chart rendering.

The online path only appends compact JSON records.  Rendering imports
matplotlib lazily and is intended to run after the service has stopped.
"""

from __future__ import annotations

import fcntl
import html
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
_TRUE_VALUES = {"1", "true", "yes", "on"}
_PHASE_COLORS = {
    "initial_prefill": "#2878B5",
    "chunked_prefill": "#F28E2B",
    "decode": "#36A657",
    "prefill_decode": "#8E5DB7",
    "wait_gap": "#C8CDD3",
    "unknown": "#7F8C8D",
}
_PHASE_LABELS = {
    "initial_prefill": "首次预填充",
    "chunked_prefill": "分块预填充",
    "decode": "解码",
    "prefill_decode": "预填充+解码",
    "wait_gap": "未执行间隔",
    "unknown": "未知",
}
_PHASE_LABELS_EN = {
    "initial_prefill": "Initial prefill",
    "chunked_prefill": "Chunked prefill",
    "decode": "Decode",
    "prefill_decode": "Prefill + decode",
    "wait_gap": "Not executing",
    "unknown": "Unknown",
}

_BATCH_EDGE_COLORS = (
    "#1B4332",
    "#7F1D1D",
    "#1E3A8A",
    "#713F12",
    "#581C87",
    "#164E63",
    "#831843",
    "#3F6212",
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return cleaned or "unknown"


class RequestGanttTraceWriter:
    """Append forward-batch records to a stage/PID-specific JSONL file."""

    def __init__(
        self,
        *,
        output_dir: str | os.PathLike[str],
        session_id: str,
        stage_id: int,
        stage_name: str,
    ) -> None:
        self.session_id = str(session_id)
        self.stage_id = int(stage_id)
        self.stage_name = str(stage_name)
        self.pid = os.getpid()
        self._sequence = 0
        session_dir = Path(output_dir).expanduser() / _safe_name(self.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"stage_{self.stage_id}_{_safe_name(self.stage_name)}_"
            f"pid_{self.pid}.jsonl"
        )
        self.path = session_dir / filename

    @classmethod
    def from_env(
        cls,
        *,
        stage_id: int,
        stage_name: str,
    ) -> RequestGanttTraceWriter | None:
        if not env_flag("VLLM_OMNI_REQUEST_GANTT"):
            return None
        return cls(
            output_dir=os.getenv(
                "VLLM_OMNI_REQUEST_GANTT_DIR", "/home/request_gantt_trace"
            ),
            session_id=os.getenv("VLLM_OMNI_REQUEST_GANTT_SESSION", "default"),
            stage_id=stage_id,
            stage_name=stage_name,
        )

    def record_forward(self, record: dict[str, Any]) -> str | None:
        """Write one model-forward batch using a cross-process clock."""
        self._sequence += 1
        start_time = float(record.get("start_time", 0.0))
        end_time = float(record.get("end_time", start_time))
        batch_id = (
            f"{_safe_name(self.session_id)}:stage{self.stage_id}:"
            f"pid{self.pid}:batch{self._sequence}"
        )
        enriched = dict(record)
        enriched.update(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "forward_batch",
                "session_id": self.session_id,
                "pid": self.pid,
                "stage_id": self.stage_id,
                "stage_name": self.stage_name,
                "batch_id": batch_id,
                "start_ns": int(start_time * 1_000_000_000),
                "end_ns": int(end_time * 1_000_000_000),
            }
        )
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                stream.flush()
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            # Tracing is optional and must never make model execution fail.
            return None
        return batch_id

    def record_timing_update(self, batch_id: str, duration_ms: float) -> None:
        """Resolve a previously written batch with an asynchronous GPU time."""
        update = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "forward_timing_update",
            "session_id": self.session_id,
            "pid": self.pid,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "batch_id": str(batch_id),
            "cuda_event_duration_ms": float(duration_ms),
        }
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.write(json.dumps(update, ensure_ascii=False) + "\n")
                stream.flush()
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            return


@dataclass(frozen=True)
class GanttSegment:
    request_id: str
    stage_id: int
    stage_name: str
    batch_id: str
    phase: str
    start_ns: int
    end_ns: int
    scheduled_tokens: int
    prefill_tokens: int
    decode_tokens: int
    prompt_tokens: int
    computed_before: int
    total_batch_tokens: int
    total_batch_requests: int
    iterations: int = 1

    @property
    def duration_ms(self) -> float:
        return max(0, self.end_ns - self.start_ns) / 1_000_000.0

    @property
    def row_key(self) -> tuple[str, int, str]:
        return self.request_id, self.stage_id, self.stage_name


def discover_trace_files(inputs: Sequence[str | os.PathLike[str]]) -> list[Path]:
    files: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.update(item for item in path.rglob("*.jsonl") if item.is_file())
        elif path.is_file():
            files.add(path)
    return sorted(files)


def load_trace_records(
    inputs: Sequence[str | os.PathLike[str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    timing_updates: dict[str, float] = {}
    for path in discover_trace_files(inputs):
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_number}: {exc}"
                    ) from exc
                record_type = record.get("record_type")
                if record_type == "forward_batch":
                    records.append(record)
                elif record_type == "forward_timing_update":
                    batch_id = str(record.get("batch_id", ""))
                    if batch_id:
                        timing_updates[batch_id] = float(
                            record.get("cuda_event_duration_ms", 0.0)
                        )
    for record in records:
        duration_ms = timing_updates.get(str(record.get("batch_id", "")))
        if duration_ms is None:
            continue
        record["cuda_event_duration_ms"] = duration_ms
        record["duration_ms"] = duration_ms
        start_ns = int(record.get("start_ns", 0))
        record["end_ns"] = start_ns + int(duration_ms * 1_000_000)
        start_time = float(record.get("start_time", 0.0))
        record["end_time"] = start_time + duration_ms / 1000.0
    return sorted(records, key=lambda item: int(item.get("start_ns", 0)))


def records_to_segments(
    records: Iterable[dict[str, Any]],
    *,
    request_ids: set[str] | None = None,
    include_gaps: bool = False,
) -> list[GanttSegment]:
    segments: list[GanttSegment] = []
    for record in records:
        start_ns = int(record.get("start_ns", 0))
        end_ns = int(record.get("end_ns", start_ns))
        for request in record.get("requests", []):
            request_id = str(
                request.get("global_request_id")
                or request.get("request_id")
                or "unknown"
            )
            if request_ids and request_id not in request_ids:
                continue
            segments.append(
                GanttSegment(
                    request_id=request_id,
                    stage_id=int(record.get("stage_id", -1)),
                    stage_name=str(record.get("stage_name", "unknown")),
                    batch_id=str(record.get("batch_id", "unknown")),
                    phase=str(request.get("phase", "unknown")),
                    start_ns=start_ns,
                    end_ns=end_ns,
                    scheduled_tokens=int(request.get("scheduled_tokens", 0)),
                    prefill_tokens=int(request.get("prefill_tokens", 0)),
                    decode_tokens=int(request.get("decode_tokens", 0)),
                    prompt_tokens=int(request.get("prompt_tokens", 0)),
                    computed_before=int(request.get("computed_tokens_before", 0)),
                    total_batch_tokens=int(record.get("total_scheduled_tokens", 0)),
                    total_batch_requests=int(record.get("total_batch_requests", 0)),
                )
            )
    segments.sort(key=lambda item: (item.start_ns, item.request_id, item.stage_id))
    if include_gaps:
        segments.extend(_gap_segments(segments))
        segments.sort(key=lambda item: (item.start_ns, item.request_id, item.stage_id))
    return segments


def _gap_segments(segments: Sequence[GanttSegment]) -> list[GanttSegment]:
    grouped: dict[tuple[str, int, str], list[GanttSegment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.row_key].append(segment)
    gaps: list[GanttSegment] = []
    for row_segments in grouped.values():
        ordered = sorted(row_segments, key=lambda item: item.start_ns)
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_ns <= previous.end_ns:
                continue
            gaps.append(
                replace(
                    previous,
                    batch_id="gap",
                    phase="wait_gap",
                    start_ns=previous.end_ns,
                    end_ns=current.start_ns,
                    scheduled_tokens=0,
                    prefill_tokens=0,
                    decode_tokens=0,
                    total_batch_tokens=0,
                    total_batch_requests=0,
                )
            )
    return gaps


def merge_decode_segments(
    segments: Sequence[GanttSegment], *, max_gap_ms: float
) -> list[GanttSegment]:
    grouped: dict[tuple[str, int, str], list[GanttSegment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.row_key].append(segment)
    merged: list[GanttSegment] = []
    max_gap_ns = int(max_gap_ms * 1_000_000)
    for row_segments in grouped.values():
        current: GanttSegment | None = None
        for segment in sorted(row_segments, key=lambda item: item.start_ns):
            can_merge = (
                current is not None
                and current.phase == "decode"
                and segment.phase == "decode"
                and segment.start_ns - current.end_ns <= max_gap_ns
            )
            if can_merge:
                current = replace(
                    current,
                    end_ns=max(current.end_ns, segment.end_ns),
                    batch_id=f"{current.batch_id}..{segment.batch_id}",
                    scheduled_tokens=current.scheduled_tokens + segment.scheduled_tokens,
                    decode_tokens=current.decode_tokens + segment.decode_tokens,
                    iterations=current.iterations + segment.iterations,
                )
            else:
                if current is not None:
                    merged.append(current)
                current = segment
        if current is not None:
            merged.append(current)
    return sorted(merged, key=lambda item: (item.start_ns, item.request_id, item.stage_id))


def _ordered_rows(
    segments: Sequence[GanttSegment],
) -> list[tuple[str, int, str]]:
    first_seen: dict[tuple[str, int, str], int] = {}
    request_first_seen: dict[str, int] = {}
    for segment in segments:
        first_seen.setdefault(segment.row_key, segment.start_ns)
        request_first_seen[segment.request_id] = min(
            request_first_seen.get(segment.request_id, segment.start_ns), segment.start_ns
        )
    return sorted(
        first_seen,
        key=lambda row: (request_first_seen[row[0]], row[0], row[1], row[2]),
    )


def _batch_display_info(
    segments: Sequence[GanttSegment],
) -> dict[str, tuple[str, str]]:
    """Assign stable, compact labels and edge colors to forward batches."""
    first_seen: dict[str, tuple[int, int]] = {}
    for segment in segments:
        if segment.phase == "wait_gap" or segment.batch_id == "gap":
            continue
        current = first_seen.get(segment.batch_id)
        candidate = (segment.stage_id, segment.start_ns)
        if current is None or candidate[1] < current[1]:
            first_seen[segment.batch_id] = candidate

    counters: dict[int, int] = defaultdict(int)
    result: dict[str, tuple[str, str]] = {}
    for batch_id, (stage_id, _) in sorted(
        first_seen.items(), key=lambda item: (item[1][1], item[1][0], item[0])
    ):
        counters[stage_id] += 1
        label = f"S{stage_id}-B{counters[stage_id]:02d}"
        color_index = (counters[stage_id] - 1) % len(_BATCH_EDGE_COLORS)
        result[batch_id] = (label, _BATCH_EDGE_COLORS[color_index])
    return result


def trace_summary(segments: Sequence[GanttSegment]) -> dict[str, Any]:
    execution_segments = [item for item in segments if item.phase != "wait_gap"]
    if not execution_segments:
        return {"requests": 0, "batches": 0, "stages": 0, "span_ms": 0.0}
    return {
        "requests": len({item.request_id for item in execution_segments}),
        "batches": len({item.batch_id for item in execution_segments}),
        "stages": len({item.stage_id for item in execution_segments}),
        "span_ms": (
            max(item.end_ns for item in execution_segments)
            - min(item.start_ns for item in execution_segments)
        )
        / 1_000_000.0,
    }


def render_html(
    segments: Sequence[GanttSegment],
    output_path: str | os.PathLike[str],
    *,
    title: str = "vLLM-Omni 请求执行甘特图",
) -> Path:
    if not segments:
        raise ValueError("No request Gantt segments to render")
    rows = _ordered_rows(segments)
    row_index = {row: index for index, row in enumerate(rows)}
    batch_display = _batch_display_info(segments)
    origin_ns = min(item.start_ns for item in segments)
    end_ns = max(item.end_ns for item in segments)
    span_ms = max((end_ns - origin_ns) / 1_000_000.0, 0.001)
    left = 300
    right = 40
    top = 85
    row_height = 30
    bottom = 55
    plot_width = 1200
    width = left + plot_width + right
    height = top + len(rows) * row_height + bottom

    def x_position(timestamp_ns: int) -> float:
        return left + ((timestamp_ns - origin_ns) / 1_000_000.0) / span_ms * plot_width

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="32" font-size="22" font-family="sans-serif">'
        f"{html.escape(title)}</text>",
    ]
    tick_count = 10
    for tick in range(tick_count + 1):
        tick_ms = span_ms * tick / tick_count
        x = left + plot_width * tick / tick_count
        svg.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top - 12}" '
            f'y2="{top + len(rows) * row_height}" stroke="#E6E8EB"/>'
        )
        svg.append(
            f'<text x="{x:.2f}" y="{height - 22}" text-anchor="middle" '
            f'font-size="11" font-family="sans-serif">{tick_ms:.1f} ms</text>'
        )
    for row, index in row_index.items():
        y = top + index * row_height
        label = f"{row[0]} / {row[2]}(stage {row[1]})"
        svg.append(
            f'<text x="{left - 10}" y="{y + 18}" text-anchor="end" '
            f'font-size="12" font-family="sans-serif">{html.escape(label)}</text>'
        )
        svg.append(
            f'<line x1="{left}" x2="{left + plot_width}" y1="{y + row_height}" '
            f'y2="{y + row_height}" stroke="#F0F1F2"/>'
        )
    for segment in segments:
        index = row_index[segment.row_key]
        y = top + index * row_height + 5
        x = x_position(segment.start_ns)
        bar_width = max(1.2, x_position(segment.end_ns) - x)
        color = _PHASE_COLORS.get(segment.phase, _PHASE_COLORS["unknown"])
        batch_label, batch_edge = batch_display.get(
            segment.batch_id, ("", "#34495E")
        )
        tooltip = (
            f"请求: {segment.request_id}\n阶段: {segment.stage_name}({segment.stage_id})\n"
            f"类型: {_PHASE_LABELS.get(segment.phase, segment.phase)}\n"
            f"时长: {segment.duration_ms:.3f} ms\n"
            f"批次编号: {batch_label or '-'}\nBatch: {segment.batch_id}\n"
            f"请求token: {segment.scheduled_tokens} "
            f"(prefill={segment.prefill_tokens}, decode={segment.decode_tokens})\n"
            f"Batch token: {segment.total_batch_tokens}, "
            f"请求数: {segment.total_batch_requests}\n迭代数: {segment.iterations}"
        )
        svg.append(
            f'<rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" height="20" '
            f'rx="2" fill="{color}" stroke="{batch_edge}" '
            f'stroke-width="{1.25 if batch_label else 0.35}">'
            f"<title>{html.escape(tooltip)}</title></rect>"
        )
        if batch_label and bar_width >= 18:
            svg.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{y + 14}" '
                f'text-anchor="middle" font-size="8.5" font-weight="bold" '
                f'font-family="sans-serif" fill="white" stroke="#202124" '
                f'stroke-width="1.8" paint-order="stroke">'
                f"{html.escape(batch_label)}</text>"
            )
    svg.append("</svg>")
    legend = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(_PHASE_LABELS[phase])}</span>'
        for phase, color in _PHASE_COLORS.items()
        if phase != "unknown"
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:Arial,"Noto Sans CJK SC",sans-serif;margin:20px;color:#202124}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0 14px 300px;font-size:13px}}
.legend span{{display:flex;align-items:center;gap:6px}} .legend i{{width:14px;height:10px;display:inline-block}}
.hint{{margin-left:300px;color:#667085;font-size:12px}} svg{{max-width:100%;height:auto}}
</style></head><body><div class="legend">{legend}</div>
<div class="hint">相同的 Sx-Bxx 编号和边框表示同一阶段的同一执行批次；
将鼠标悬停在色块上可查看完整 batch 和 token 详情。</div>
{''.join(svg)}</body></html>"""
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def render_png(
    segments: Sequence[GanttSegment],
    output_path: str | os.PathLike[str],
    *,
    title: str = "vLLM-Omni request execution Gantt",
) -> Path:
    if not segments:
        raise ValueError("No request Gantt segments to render")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch

    available_fonts = {item.name for item in font_manager.fontManager.ttflist}
    cjk_candidates = (
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
    )
    cjk_font = next(
        (name for name in cjk_candidates if name in available_fonts), None
    )
    if cjk_font:
        plt.rcParams["font.sans-serif"] = [cjk_font, "DejaVu Sans"]
        png_title = title
        png_labels = _PHASE_LABELS
    else:
        # Keep PNG readable in minimal containers that only ship DejaVu.
        # The self-contained HTML retains Chinese labels through browser fonts.
        png_title = title if title.isascii() else "vLLM-Omni request execution Gantt"
        png_labels = _PHASE_LABELS_EN

    rows = _ordered_rows(segments)
    row_index = {row: index for index, row in enumerate(rows)}
    batch_display = _batch_display_info(segments)
    origin_ns = min(item.start_ns for item in segments)
    end_ns = max(item.end_ns for item in segments)
    span_ms = max((end_ns - origin_ns) / 1_000_000.0, 0.001)
    figure_height = max(3.0, 0.42 * len(rows) + 1.6)
    fig, axis = plt.subplots(figsize=(15, figure_height))
    for segment in segments:
        start_ms = (segment.start_ns - origin_ns) / 1_000_000.0
        width_ms = max(segment.duration_ms, 0.001)
        y = row_index[segment.row_key]
        batch_label, batch_edge = batch_display.get(
            segment.batch_id, ("", "#34495E")
        )
        axis.barh(
            y,
            width_ms,
            left=start_ms,
            height=0.62,
            color=_PHASE_COLORS.get(segment.phase, _PHASE_COLORS["unknown"]),
            edgecolor=batch_edge,
            linewidth=1.0 if batch_label else 0.35,
        )
        if batch_label and width_ms / span_ms * 1200 >= 18:
            axis.text(
                start_ms + width_ms / 2,
                y,
                batch_label,
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color="white",
                path_effects=None,
            )
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(
        [f"{request_id} / {stage_name}(stage {stage_id})" for request_id, stage_id, stage_name in rows]
    )
    axis.invert_yaxis()
    axis.set_xlabel("Relative time (ms)")
    axis.set_title(png_title)
    axis.grid(axis="x", color="#E6E8EB", linewidth=0.7)
    axis.set_axisbelow(True)
    used_phases = {item.phase for item in segments}
    legend = [
        Patch(color=color, label=png_labels.get(phase, phase))
        for phase, color in _PHASE_COLORS.items()
        if phase in used_phases
    ]
    if legend:
        axis.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5)
    fig.tight_layout()
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
