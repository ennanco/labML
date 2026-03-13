from __future__ import annotations

import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_GIF = REPO_ROOT / "labml-demo.gif"
PROGRESS_GIF = REPO_ROOT / "labml-progress.gif"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text).replace("\r", "")


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    if completed.stderr.strip():
        output = f"{output}\n{completed.stderr}"
    return _strip_ansi(output).strip()


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap_line(line: str, max_chars: int) -> list[str]:
    if not line:
        return [""]
    return textwrap.wrap(
        line, width=max_chars, replace_whitespace=False, drop_whitespace=False
    ) or [line]


def _render_terminal_frame(title: str, body: str, output_path: Path) -> None:
    width = 1400
    height = 860
    bg = (14, 17, 23)
    fg = (226, 232, 240)
    prompt = (134, 239, 172)
    accent = (125, 211, 252)

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    font = _load_font(23)
    line_height = 32
    left = 36
    top = 34
    max_chars = 96

    draw.text((left, top), title, fill=accent, font=font)
    y = top + line_height + 12
    for raw in body.splitlines():
        color = prompt if raw.startswith("$") else fg
        for chunk in _wrap_line(raw, max_chars):
            if y > height - line_height:
                break
            draw.text((left, y), chunk, fill=color, font=font)
            y += line_height
        if y > height - line_height:
            break

    image.save(output_path)


def _ffmpeg_gif(frames: list[tuple[Path, float]], output_gif: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="labml_gif_") as tmp:
        manifest = Path(tmp) / "frames.txt"
        lines: list[str] = []
        for frame, duration in frames:
            lines.append(f"file '{frame.as_posix()}'")
            lines.append(f"duration {duration:.3f}")
        lines.append(f"file '{frames[-1][0].as_posix()}'")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

        vf = (
            "fps=12,"
            "scale=1000:-1:flags=lanczos,"
            "split[s0][s1];"
            "[s0]palettegen=stats_mode=full[p];"
            "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest),
                "-vf",
                vf,
                str(output_gif),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


def _build_demo_frames(work_dir: Path) -> list[tuple[Path, float]]:
    help_text = _run(["uv", "run", "labml", "--help"])
    prepare_text = _run(
        ["uv", "run", "labml", "prepare", "--config", "examples/demo_prepare.toml"]
    )
    benchmark_text = _run(
        [
            "uv",
            "run",
            "labml",
            "benchmark-regression",
            "--config",
            "examples/demo_benchmark_regression.toml",
        ]
    )

    result_file = REPO_ROOT / "_artifacts_" / "benchmarks" / "demo" / "results.xlsx"
    prepared_dir = REPO_ROOT / "_artifacts_" / "prepared" / "demo"
    artifacts = [
        prepared_dir / "data.parquet",
        prepared_dir / "folds.csv",
        prepared_dir / "metadata.json",
        result_file,
    ]
    artifact_lines = [
        f"- {path.relative_to(REPO_ROOT)} ({'OK' if path.is_file() else 'MISSING'})"
        for path in artifacts
    ]

    frames: list[tuple[Path, float]] = []

    frame1 = work_dir / "demo_01.png"
    _render_terminal_frame(
        "labML demo: command discovery",
        "\n".join(
            [
                "$ uv run labml --help",
                *help_text.splitlines()[:18],
            ]
        ),
        frame1,
    )
    frames.append((frame1, 3.0))

    frame2 = work_dir / "demo_02.png"
    _render_terminal_frame(
        "labML demo: prepare stage",
        "\n".join(
            [
                "$ uv run labml prepare --config examples/demo_prepare.toml",
                *prepare_text.splitlines()[-14:],
            ]
        ),
        frame2,
    )
    frames.append((frame2, 3.2))

    frame3 = work_dir / "demo_03.png"
    _render_terminal_frame(
        "labML demo: benchmark stage",
        "\n".join(
            [
                "$ uv run labml benchmark-regression --config examples/demo_benchmark_regression.toml",
                *benchmark_text.splitlines()[-18:],
            ]
        ),
        frame3,
    )
    frames.append((frame3, 3.4))

    frame4 = work_dir / "demo_04.png"
    _render_terminal_frame(
        "labML demo: generated artifacts",
        "\n".join(["$ ls _artifacts_", "", *artifact_lines]),
        frame4,
    )
    frames.append((frame4, 3.4))

    return frames


def _build_progress_frames(work_dir: Path) -> list[tuple[Path, float]]:
    frames: list[tuple[Path, float]] = []
    states = [
        (
            "progress_01.png",
            2.6,
            """$ uv run labml benchmark-regression --config examples/demo_benchmark_regression.toml

Overall
[====--------------------------------] 1/8 combinations completed

Current Combination
[========----------------------------] 1/4 folds complete

Status
Running: none->none->none->pls (1/8) - fold 1/4

Recent Results
(empty)
""",
        ),
        (
            "progress_02.png",
            2.2,
            """Overall
[========----------------------------] 2/8 combinations completed

Current Combination
[====================================] 4/4 folds complete

Status
Completed: none->none->none->pls (1/8)
Running: none->none->none->rf (2/8) - fold 2/4

Recent Results
✅ none->none->none->pls (1/8)
""",
        ),
        (
            "progress_03.png",
            2.2,
            """Overall
[============------------------------] 3/8 combinations completed

Current Combination
[========================------------] 3/4 folds complete

Status
Running: none->none->pca->pls (3/8) - fold 3/4

Recent Results
✅ none->none->none->pls (1/8)
✅ none->none->none->rf (2/8)
""",
        ),
        (
            "progress_04.png",
            2.4,
            """Overall
[====================----------------] 4/8 combinations completed

Current Combination
[====================================] 4/4 folds complete

Status
Completed: none->none->pca->pls (3/8)
Running: none->none->pca->rf (4/8) - fold 1/4

Recent Results
✅ none->none->none->pls (1/8)
✅ none->none->none->rf (2/8)
✅ none->none->pca->pls (3/8)
""",
        ),
        (
            "progress_05.png",
            2.2,
            """Overall
[========================------------] 5/8 combinations completed

Current Combination
[====================================] 4/4 folds complete

Status
Failed: none->none->pca->rf (4/8)
Running: norm->none->none->pls (5/8) - fold 1/4

Recent Results
✅ none->none->none->pls (1/8)
✅ none->none->none->rf (2/8)
✅ none->none->pca->pls (3/8)
✖ none->none->pca->rf (4/8)
""",
        ),
        (
            "progress_06.png",
            2.2,
            """Overall
[============================--------] 6/8 combinations completed

Current Combination
[====================================] Skipped (NMF compatibility)

Status
Skipped: norm->none->pca->pls (6/8)
Running: norm->none->none->rf (7/8) - fold 1/4

Recent Results
✅ none->none->none->rf (2/8)
✅ none->none->pca->pls (3/8)
✖ none->none->pca->rf (4/8)
✅ norm->none->none->pls (5/8)
⚠ norm->none->pca->pls (6/8)
""",
        ),
        (
            "progress_07.png",
            3.0,
            """Overall
[====================================] 8/8 combinations completed

Current Combination
[====================================] 4/4 folds complete

Status
Completed: norm->none->pca->rf (8/8)
Benchmark finished. Results saved to: _artifacts_/benchmarks/demo/results.xlsx

Recent Results
✅ none->none->pca->pls (3/8)
✖ none->none->pca->rf (4/8)
✅ norm->none->none->pls (5/8)
⚠ norm->none->pca->pls (6/8)
✅ norm->none->none->rf (7/8)
✅ norm->none->pca->rf (8/8)
""",
        ),
    ]

    for filename, duration, text in states:
        frame = work_dir / filename
        _render_terminal_frame("labML benchmark progress", text.strip("\n"), frame)
        frames.append((frame, duration))
    return frames


def main() -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="labml_demo_frames_"))
    demo_frames = _build_demo_frames(work_dir)
    progress_frames = _build_progress_frames(work_dir)

    _ffmpeg_gif(demo_frames, DEMO_GIF)
    _ffmpeg_gif(progress_frames, PROGRESS_GIF)

    print(f"Generated {DEMO_GIF.relative_to(REPO_ROOT)}")
    print(f"Generated {PROGRESS_GIF.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
