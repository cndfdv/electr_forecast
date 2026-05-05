"""Run a clean full benchmark and rebuild paper artifacts.

This script is intentionally conservative: expensive or fragile models are run
late, so partial overnight output still contains the core comparison.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIGHT_PY = Path(os.environ.get("ELECTR_FORECAST_PYTHON", sys.executable))
FOUNDATION_PY = Path(
    os.environ.get(
        "ELECTR_FORECAST_FOUNDATION_PYTHON",
        str(LIGHT_PY.parent.parent.parent / "electr-forecast-foundation" / "bin" / "python"),
    )
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(message: str, log_path: Path) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    try:
        print(line, flush=True)
    except BrokenPipeError:
        pass


def run_cmd(cmd: list[str], log_path: Path, allow_fail: bool = False, cwd: Path | None = None) -> int:
    log("$ " + " ".join(cmd), log_path)
    start = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=cwd or ROOT, stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    log(f"exit={proc.returncode} elapsed_s={elapsed:.1f}", log_path)
    if proc.returncode != 0 and not allow_fail:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def backup_results(log_path: Path) -> None:
    results = ROOT / "results"
    if results.exists():
        dst = ROOT / "results_runs" / f"results_{stamp()}"
        dst.parent.mkdir(exist_ok=True)
        shutil.move(str(results), str(dst))
        log(f"moved previous results to {dst}", log_path)
    (ROOT / "results" / "window_errors").mkdir(parents=True, exist_ok=True)


def ensure_header(path: Path) -> None:
    from src.experiments.run_experiment import RESULT_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(RESULT_COLUMNS)


def classical_jobs(seeds: list[int], horizons: list[int]) -> list[tuple[list[str], bool]]:
    jobs: list[tuple[list[str], bool]] = []
    for dataset in ["ecl", "etth1"]:
        for horizon in horizons:
            jobs.append(([str(LIGHT_PY), "-m", "src.experiments.run_experiment", "--dataset", dataset, "--model", "seasonal_naive", "--horizon", str(horizon), "--seed", str(seeds[0])], False))
            for seed in seeds:
                jobs.append(([str(LIGHT_PY), "-m", "src.experiments.run_experiment", "--dataset", dataset, "--model", "xgboost", "--horizon", str(horizon), "--seed", str(seed)], False))
    return jobs


def neural_jobs(seeds: list[int], horizons: list[int], max_steps: int) -> list[tuple[list[str], bool]]:
    jobs: list[tuple[list[str], bool]] = []
    for dataset in ["ecl", "etth1"]:
        for model in ["dlinear", "patchtst", "itransformer"]:
            for horizon in horizons:
                for seed in seeds:
                    jobs.append((
                        [
                            str(FOUNDATION_PY),
                            "-m",
                            "src.experiments.run_neuralforecast_experiment",
                            "--dataset",
                            dataset,
                            "--model",
                            model,
                            "--horizon",
                            str(horizon),
                            "--seed",
                            str(seed),
                            "--max-steps",
                            str(max_steps),
                            "--latency-repeats",
                            "10",
                        ],
                        False,
                    ))
    return jobs


def foundation_jobs(horizons: list[int]) -> list[tuple[list[str], bool]]:
    jobs: list[tuple[list[str], bool]] = []
    for dataset in ["ecl", "etth1"]:
        for horizon in horizons:
            jobs.append(([str(FOUNDATION_PY), "-m", "src.experiments.run_foundation_experiment", "--dataset", dataset, "--model", "chronos_bolt_small", "--horizon", str(horizon), "--batch-size", "64", "--latency-repeats", "5"], False))
            jobs.append(([str(FOUNDATION_PY), "-m", "src.experiments.run_foundation_experiment", "--dataset", dataset, "--model", "timesfm", "--horizon", str(horizon), "--batch-size", "64", "--checkpoint", "google/timesfm-2.5-200m-pytorch", "--latency-repeats", "5"], False))
    return jobs


def autoarima_jobs(horizons: list[int]) -> list[tuple[list[str], bool]]:
    jobs: list[tuple[list[str], bool]] = []
    for dataset in ["ecl", "etth1"]:
        for horizon in horizons:
            jobs.append(([str(LIGHT_PY), "-m", "src.experiments.run_experiment", "--dataset", dataset, "--model", "auto_arima", "--horizon", str(horizon), "--seed", "42"], True))
    return jobs


def sarima_jobs(horizons: list[int]) -> list[tuple[list[str], bool]]:
    jobs: list[tuple[list[str], bool]] = []
    for dataset in ["ecl", "etth1"]:
        for horizon in horizons:
            jobs.append(([str(LIGHT_PY), "-m", "src.experiments.run_experiment", "--dataset", dataset, "--model", "sarima", "--horizon", str(horizon), "--seed", "42", "--latency-repeats", "5"], False))
    return jobs


def _convert_figures_to_png(log_path: Path) -> None:
    """Convert figures/*.pdf to paper/docx_figures/*.png for Word/pandoc embedding."""
    src = ROOT / "figures"
    dst = ROOT / "paper" / "docx_figures"
    dst.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(src.glob("*.pdf"))
    if not pdfs:
        log("no PDF figures found to convert", log_path)
        return
    for pdf in pdfs:
        png = dst / (pdf.stem + ".png")
        # pdftoppm ships with poppler; -singlefile drops the trailing -1 page suffix.
        run_cmd(
            ["pdftoppm", "-png", "-r", "200", "-singlefile", str(pdf), str(png.with_suffix(""))],
            log_path,
            allow_fail=True,
        )


def _build_md_paper(paper_dir: Path, stem: str, log_path: Path, lang: str = "en") -> None:
    """Render <stem>.md to .tex, .pdf, .docx via pandoc + citeproc."""
    md = f"{stem}.md"
    common = [
        "pandoc", md,
        "--citeproc",
        "--bibliography", "references.bib",
        "--metadata", "link-citations=true",
    ]
    pdf_extra: list[str]
    if lang == "ru":
        pdf_extra = [
            "--pdf-engine=xelatex",
            "-V", "mainfont=Times New Roman",
            "-V", "lang=ru",
        ]
    else:
        pdf_extra = []
    run_cmd(common + ["-o", f"{stem}.tex"], log_path, allow_fail=True, cwd=paper_dir)
    run_cmd(common + pdf_extra + ["-o", f"{stem}.pdf"], log_path, allow_fail=True, cwd=paper_dir)
    run_cmd(common + ["-o", f"{stem}.docx"], log_path, allow_fail=True, cwd=paper_dir)


def _build_ieee_paper(paper_dir: Path, stem: str, log_path: Path) -> None:
    """Compile <stem>.tex with pdflatex + bibtex; also produce a .docx with PNG figures."""
    tex = f"{stem}.tex"
    run_cmd(["pdflatex", "-interaction=nonstopmode", tex], log_path, allow_fail=True, cwd=paper_dir)
    run_cmd(["bibtex", stem], log_path, allow_fail=True, cwd=paper_dir)
    run_cmd(["pdflatex", "-interaction=nonstopmode", tex], log_path, allow_fail=True, cwd=paper_dir)
    run_cmd(["pdflatex", "-interaction=nonstopmode", tex], log_path, allow_fail=True, cwd=paper_dir)

    # DOCX via pandoc, with PDF figure paths swapped for the rasterised PNGs in
    # docx_figures/ so Word actually shows the figures.
    src_path = paper_dir / tex
    src = src_path.read_text(encoding="utf-8")
    docx_src = re.sub(r"\.\./figures/(fig\d_[a-z_]+)\.pdf", r"docx_figures/\1.png", src)
    tmp_tex = paper_dir / f"_{stem}_docx.tex"
    tmp_tex.write_text(docx_src, encoding="utf-8")
    try:
        run_cmd(
            ["pandoc", tmp_tex.name, "--bibliography", "references.bib", "--citeproc",
             "-o", f"{stem}.docx"],
            log_path, allow_fail=True, cwd=paper_dir,
        )
    finally:
        tmp_tex.unlink(missing_ok=True)


def rebuild_artifacts(log_path: Path) -> None:
    run_cmd([str(LIGHT_PY), "-m", "src.evaluation.run_stat_tests", "--results", "results/main_results.csv"], log_path, allow_fail=True)
    run_cmd([str(LIGHT_PY), "-m", "src.evaluation.make_latex_tables", "--results", "results/main_results.csv"], log_path)
    run_cmd([str(LIGHT_PY), "-m", "src.figures.make_figures", "--results", "results/main_results.csv", "--output-dir", "figures"], log_path)
    _convert_figures_to_png(log_path)

    paper_dir = ROOT / "paper"
    # Flat single-column English (main.md is the source of truth).
    _build_md_paper(paper_dir, "main", log_path, lang="en")
    # Flat single-column Russian.
    _build_md_paper(paper_dir, "main_ru", log_path, lang="ru")
    # IEEE conference two-column versions (compiled directly from .tex).
    _build_ieee_paper(paper_dir, "main_ieee", log_path)
    _build_ieee_paper(paper_dir, "main_ieee_ru", log_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--horizons", nargs="+", type=int, default=[24, 96, 168])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--skip-autoarima", action="store_true")
    parser.add_argument(
        "--only-sarima",
        action="store_true",
        help="Run only the SARIMA jobs and append to existing results/main_results.csv "
             "(skips backup, header rewrite, other models, and artifact rebuild).",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    pid_path = logs / "full_clean.pid"
    log_path = logs / f"full_clean_{stamp()}.log"
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    try:
        if args.only_sarima:
            log("starting SARIMA-only run (append mode)", log_path)
            jobs = sarima_jobs(args.horizons)
        else:
            log("starting full clean benchmark", log_path)
            backup_results(log_path)
            ensure_header(ROOT / "results" / "main_results.csv")

            jobs = classical_jobs(args.seeds, args.horizons)
            jobs += neural_jobs(args.seeds, args.horizons, args.max_steps)
            jobs += foundation_jobs(args.horizons)
            jobs += sarima_jobs(args.horizons)
            if not args.skip_autoarima:
                jobs += autoarima_jobs(args.horizons)

        failures: list[str] = []
        for idx, (cmd, allow_fail) in enumerate(jobs, start=1):
            log(f"job {idx}/{len(jobs)}", log_path)
            code = run_cmd(cmd, log_path, allow_fail=allow_fail)
            if code != 0:
                failures.append(" ".join(cmd))

        if not args.only_sarima:
            rebuild_artifacts(log_path)
        if failures:
            log("non-fatal failures:", log_path)
            for item in failures:
                log(item, log_path)
        log("finished run", log_path)
    except Exception as exc:
        log(f"fatal error: {exc!r}", log_path)
        raise


if __name__ == "__main__":
    main()
