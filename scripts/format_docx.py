"""Polish the manually edited Scopus DOCX without rebuilding it.

The current master article is ``paper/main_scopus.docx``.  This script is
intentionally conservative: it does not regenerate tables, figures, authors, or
references.  It only expands selected narrative paragraphs and inserts numeric
citations that correspond to the reference list already present in the DOCX.

Run:
    mamba run -n electr-forecast python -m scripts.format_docx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCX = ROOT / "paper" / "main_scopus.docx"


REPLACEMENTS: dict[str, str] = {
    "Short-term electricity load forecasting (STLF) supports dispatch planning": (
        "Short-term electricity load forecasting (STLF) supports dispatch "
        "planning, balancing, tariff-aware operation, maintenance planning, "
        "and anomaly-aware industrial energy management. Forecasting errors "
        "directly affect operational cost, reserve scheduling, and the "
        "reliability of downstream decision-support systems. This role is "
        "consistent with the broader load-forecasting literature, where "
        "accurate point and probabilistic forecasts are treated as core inputs "
        "for smart-grid planning and operation [1]."
    ),
    "Modern energy systems generate large volumes of hourly and sub-hourly measurements": (
        "Modern energy systems generate large volumes of hourly and sub-hourly "
        "measurements. This makes it possible to train data-driven forecasting "
        "models, but it also creates a methodological problem: increasingly "
        "complex neural and foundation models must be compared against strong, "
        "transparent, and inexpensive baselines. Earlier residential and "
        "building-load studies showed that recurrent and deep neural models "
        "can improve forecasting accuracy under suitable data conditions [2]–[4], "
        "but they do not remove the need for classical statistical controls."
    ),
    "This paper presents a reproducible benchmark that compares classical statistical models": (
        "This paper presents a reproducible benchmark that compares classical "
        "statistical models, machine-learning methods, deep-learning "
        "architectures, and time-series foundation models under the same "
        "chronological evaluation protocol. The comparison is deliberately "
        "pragmatic: the goal is not to declare a universally best model, but to "
        "identify where classical SARIMA, feature-based boosting, compact "
        "neural models, and zero-shot foundation models are operationally "
        "useful."
    ),
    "Classical ARIMA-family models remain widely used in industrial forecasting": (
        "Classical ARIMA-family models remain widely used in industrial "
        "forecasting because they are interpretable, computationally "
        "inexpensive, and well understood. Their use follows the Box-Jenkins "
        "forecasting tradition [5], while modern automatic order-selection "
        "procedures and forecasting textbooks provide practical guidance for "
        "seasonal model identification and validation [6], [7]. Seasonal naive "
        "models are even simpler, but they form a necessary reference for load "
        "forecasting studies with strong daily or weekly seasonality."
    ),
    "Gradient-boosted decision trees are a strong machine-learning baseline": (
        "Gradient-boosted decision trees are a strong machine-learning baseline "
        "for energy forecasting. XGBoost is particularly attractive because it "
        "combines nonlinear feature interactions, regularization, and fast "
        "training on tabular lag features [9]. Deep-learning models, including "
        "LSTM networks [10], attention-based architectures [11], and long-horizon "
        "transformer variants such as Informer, Autoformer, and FEDformer "
        "[12]–[14], are attractive for long-horizon forecasting but often "
        "require careful tuning and sufficient data."
    ),
    "Recent time-series foundation models, including Chronos and TimesFM": (
        "Recent time-series foundation models introduce forecasting without "
        "task-specific training. TimesFM, Chronos, MOMENT, Moirai, and "
        "Lag-Llama represent a shift from dataset-specific supervised fitting "
        "towards pretrained sequence models for generic time-series prediction "
        "[20]–[24]. This is useful for cold-start settings, but public benchmark "
        "contamination must be acknowledged because ECL and ETT are widely used "
        "datasets and may have influenced model development."
    ),
    "The benchmark is formulated as a univariate rolling-origin forecasting task": (
        "The benchmark is formulated as a univariate rolling-origin forecasting "
        "task. Given a context window of L=336 hourly observations, each model "
        "produces a multi-step point forecast for the next H values, where "
        "H ∈ {24, 96, 168}. Rolling-origin evaluation is used because it "
        "matches the operational setting in which a forecaster repeatedly "
        "receives a historical context and predicts a fixed future horizon. "
        "All trained models are evaluated on a held-out chronological test "
        "split in the original target scale after inverse standardization."
    ),
    "The primary dataset is ECL (Electricity Consuming Load)": (
        "The primary dataset is ECL (Electricity Consuming Load), aggregated "
        "across 321 clients at hourly resolution; the target is aggregate "
        "electricity load. The secondary dataset is ETTh1, where HUFL is used "
        "as the load-related target variable. The OT variable in ETTh1 is oil "
        "temperature and is therefore not interpreted as electricity load. "
        "Both datasets are common public benchmarks for long- and short-term "
        "time-series forecasting [25], which supports reproducibility but also "
        "requires a cautious interpretation of zero-shot foundation-model "
        "results."
    ),
    "The model set covers four families": (
        "The model set covers four families. SeasonalNaive and SARIMA represent "
        "classical baselines, with SARIMA implemented through statistical "
        "forecasting tooling [27], [28]. XGBoost represents supervised "
        "feature-based machine learning [9]. DLinear, PatchTST, and "
        "iTransformer represent trained neural forecasting models and are run "
        "through NeuralForecast [15]–[18], [26]. Chronos-Bolt-Small and "
        "TimesFM 2.5 represent zero-shot foundation models [20], [21]. "
        "Accuracy is evaluated using MAE, RMSE, sMAPE, and wMAPE; statistical "
        "comparisons use the Diebold-Mariano test [8]."
    ),
    "Two classical baselines are included": (
        "Two classical baselines are included. SeasonalNaive repeats the most "
        "recent daily seasonal pattern and provides a deliberately simple "
        "seasonality reference. SARIMA performs an AICc-based grid search on "
        "the training split over candidate non-seasonal and seasonal orders; "
        "the selected configuration is then reused on every test window via "
        "state filtering on the new context. This follows established ARIMA "
        "forecasting methodology [5]–[7] and uses the statistical forecasting "
        "tooling reported in [27], with Python econometric infrastructure "
        "consistent with [28]."
    ),
    "XGBoost is trained as a direct multi-output regressor": (
        "XGBoost is trained as a direct multi-output regressor with one "
        "estimator per forecast step. Features comprise lag-1, lag-24, and "
        "lag-168 values of the target; rolling means over 24 and 168 hours; "
        "and cyclic calendar encodings for hour-of-day, day-of-week, and "
        "month. This design follows the common tabular forecasting strategy "
        "of exposing seasonal structure through engineered lag features while "
        "letting gradient boosting model nonlinear interactions [9]. DLinear, "
        "PatchTST, and iTransformer are trained through NeuralForecast [26], "
        "which provides a consistent interface for the neural architectures "
        "evaluated in [15]–[18]."
    ),
    "Two time-series foundation models are evaluated in zero-shot mode": (
        "Two time-series foundation models are evaluated in zero-shot mode "
        "without task-specific training: Chronos-Bolt-Small and TimesFM 2.5. "
        "Both are conditioned on the same L=336 context as the trained models "
        "to keep the comparison protocol aligned. TimesFM and Chronos are "
        "representative of the recent decoder-only and tokenized foundation "
        "model direction [20], [21], while MOMENT, Moirai, and Lag-Llama show "
        "that this research line is rapidly expanding beyond a single model "
        "family [22]–[24]."
    ),
    "Accuracy is measured with mean absolute error": (
        "Accuracy is measured with mean absolute error (MAE), root mean squared "
        "error (RMSE), symmetric mean absolute percentage error (sMAPE), and "
        "weighted mean absolute percentage error (wMAPE). These metrics are "
        "reported together because scale-dependent absolute errors are useful "
        "for operational interpretation, while percentage-style measures help "
        "compare behavior across targets with different magnitudes [6]. "
        "Efficiency is reported as training time, single-window inference "
        "latency at batch size 1, trainable parameter count, and peak GPU "
        "memory."
    ),
    "Stochastically initialized models": (
        "Stochastically initialized models (XGBoost, DLinear, PatchTST, and "
        "iTransformer) are evaluated across five seeds {42, 43, 44, 45, 46} "
        "and reported as mean ± standard deviation. Deterministic models "
        "(SeasonalNaive, SARIMA, Chronos-Bolt, and TimesFM) are evaluated once "
        "and have a standard deviation of zero by construction. Statistical "
        "comparisons use the Diebold-Mariano test on per-window absolute "
        "errors, which is the standard test for comparing predictive accuracy "
        "between competing forecasts [8]."
    ),
    "On ECL, the top group consists of TimesFM, Chronos-Bolt, and XGBoost": (
        "On ECL, the top group consists of TimesFM, Chronos-Bolt, and XGBoost. "
        "Their differences are small across all horizons. TimesFM is marginally "
        "best at H=24, while XGBoost is marginally best at H=96 and H=168. "
        "This confirms that feature-based boosting remains a difficult baseline "
        "to beat on aggregate load data when informative lag and calendar "
        "features are available [9]. SARIMA remains close to the leading group, "
        "which is important because it achieves competitive accuracy without "
        "deep representation learning."
    ),
    "On ETTh1/HUFL, the ranking changes": (
        "On ETTh1/HUFL, the ranking changes. Chronos-Bolt is best at H=24, "
        "while DLinear is best at H=96 and H=168. This result supports the "
        "view that simple linear neural baselines can remain highly competitive "
        "against more complex transformer architectures in time-series "
        "forecasting [15]. It also demonstrates why load-forecasting claims "
        "should be reported by dataset and horizon rather than as a single "
        "global leaderboard."
    ),
    "The Diebold-Mariano tests reject equal predictive accuracy": (
        "The Diebold-Mariano tests reject equal predictive accuracy for the top "
        "model pairs in all evaluated dataset-horizon combinations [8]. "
        "However, statistical significance and operational significance are not "
        "identical. On ECL, the absolute differences among the top three methods "
        "are small, so inference latency, training cost, and deployment "
        "complexity become important additional selection criteria."
    ),
    "The main finding is that no model family dominates across datasets and horizons": (
        "The main finding is that no model family dominates across datasets and "
        "horizons. On aggregate ECL load, engineered lag features with XGBoost "
        "remain a strong and inexpensive supervised solution. SARIMA provides a "
        "competitive classical reference grounded in established forecasting "
        "methodology [5]–[7]. Foundation models are competitive without "
        "task-specific training, but their practical value depends on the "
        "deployment setting rather than on accuracy alone."
    ),
    "Foundation models are attractive when task-specific training is undesirable": (
        "Foundation models are attractive when task-specific training is "
        "undesirable, for example in cold-start industrial monitoring, rapid "
        "prototyping, or sites where historical data cannot be pooled for "
        "model training. Nevertheless, the results do not support replacing "
        "supervised baselines by default. Instead, foundation models should be "
        "treated as a complementary option alongside XGBoost, SARIMA, and "
        "compact neural models [20]–[24]."
    ),
    "The trained transformer models should not be interpreted as generally ineffective": (
        "The trained transformer models should not be interpreted as generally "
        "ineffective. The study uses a compact target-only setup and a limited "
        "training budget. Broader transformer surveys and architectures show "
        "that performance can depend strongly on decomposition, patching, "
        "inverted tokenization, multivariate inputs, and hyperparameter choices "
        "[12]–[19]. The current result is therefore a statement about this "
        "reproducible protocol, not a universal rejection of transformer-based "
        "forecasting."
    ),
    "The first limitation is public-benchmark pretraining contamination": (
        "The first limitation is public-benchmark pretraining contamination. "
        "ECL and ETT are public datasets, and they may overlap with corpora "
        "used during foundation-model development. The correct interpretation "
        "is therefore that Chronos-Bolt and TimesFM are evaluated without "
        "task-specific training in this repository; the experiment does not "
        "prove dataset-unseen generalization for private industrial sites."
    ),
    "The second limitation is the target-only design": (
        "The second limitation is the target-only design. Real industrial "
        "forecasting systems often include weather, calendar events, tariffs, "
        "production schedules, and operational constraints. Adding such "
        "exogenous variables may change the ranking of model families, "
        "especially for tree-based and transformer-based approaches."
    ),
    "The third limitation is the restricted hyperparameter budget": (
        "The third limitation is the restricted hyperparameter budget. The "
        "comparison is intentionally reproducible and computationally bounded, "
        "but larger searches could improve neural and tree-based models. This "
        "trade-off is acceptable for a benchmark paper because the protocol is "
        "transparent and all reported conclusions are tied to the executed "
        "configuration."
    ),
    "This paper presented a reproducible benchmark of classical": (
        "This paper presented a reproducible benchmark of classical, "
        "machine-learning, deep-learning, and foundation-model approaches for "
        "short-term electricity load forecasting. The experiments show that "
        "XGBoost, Chronos-Bolt, and TimesFM are near-tied on ECL, whereas "
        "Chronos-Bolt and DLinear dominate different horizons on ETTh1/HUFL. "
        "These findings reinforce the need to compare modern foundation models "
        "against well-tuned classical and supervised baselines."
    ),
    "The practical conclusion is that strong simple baselines remain necessary": (
        "The practical conclusion is that strong simple baselines remain "
        "necessary. SeasonalNaive and SARIMA establish a meaningful lower bound, "
        "XGBoost provides a strong low-cost supervised model, DLinear is a "
        "useful compact neural reference, and foundation models are valuable "
        "cold-start forecasters. A deployment-oriented STLF study should report "
        "accuracy, statistical significance, and computational cost together."
    ),
}


INSERT_AFTER: dict[str, list[str]] = {
    "Recent time-series foundation models introduce forecasting without task-specific training": [
        (
            "For STLF specifically, this background implies that a useful comparison must include "
            "both domain-standard baselines and recent general-purpose models. A benchmark that "
            "contains only neural or foundation models risks overstating novelty, while a "
            "benchmark that excludes pretrained models misses a deployment pattern that is now "
            "common in industrial prototyping. Therefore, the present study evaluates classical "
            "seasonal methods, gradient boosting, compact neural architectures, and zero-shot "
            "foundation models within a single execution pipeline."
        ),
        (
            "The long-sequence forecasting literature also motivates the inclusion of simple "
            "neural baselines. Linear and decomposition-based models have repeatedly shown that "
            "strong temporal biases can match or outperform more complex attention mechanisms on "
            "standard benchmarks [15]. Patch-based transformers and inverted-tokenization "
            "architectures address different weaknesses of vanilla sequence attention [16], [18], "
            "but their advantage is not guaranteed under a target-only STLF protocol. This makes "
            "DLinear, PatchTST, and iTransformer a useful compact neural set rather than an "
            "arbitrary model list."
        ),
    ],
    "Accuracy is measured with mean absolute error (MAE), root mean squared error": [
        (
            "The comparison protocol is intentionally leakage-safe. Dataset statistics used for "
            "standardization are estimated only on the training split, validation choices are made "
            "without access to the test segment, and each test forecast uses only observations "
            "available before its forecast origin. Prophet-like or tree-based feature engineering "
            "is also restricted to lag and calendar information that would be known at prediction "
            "time. This is essential because small leakage errors can dominate benchmark rankings "
            "on public time-series datasets."
        ),
        (
            "Efficiency is treated as part of the method rather than as an afterthought. Training "
            "time captures the cost of adapting a model to a new site, while single-window "
            "latency captures the cost of producing one operational forecast after deployment. "
            "These two quantities answer different engineering questions: a zero-shot model may "
            "avoid training entirely but still be expensive at inference, whereas a supervised "
            "model may require fitting once and then produce forecasts cheaply."
        ),
    ],
    "The trained transformer models should not be interpreted as generally ineffective": [
        (
            "From a deployment perspective, the results suggest a tiered model-selection strategy. "
            "SeasonalNaive and SARIMA are useful first checks because they reveal whether the "
            "series is dominated by regular seasonality. XGBoost is an appropriate default when "
            "sufficient site history is available and feature engineering is acceptable. DLinear "
            "is a lightweight neural alternative for horizons where linear temporal structure is "
            "dominant. Foundation models are most attractive when training data are scarce, "
            "experiments must be started quickly, or the cost of maintaining site-specific models "
            "is high."
        ),
        (
            "The accuracy-latency results also caution against interpreting model quality through "
            "a single metric. A model that is marginally more accurate but substantially slower "
            "may be unattractive in an online monitoring system, while a slower zero-shot model "
            "may still be useful if it eliminates training and tuning for many small assets. "
            "Consequently, the benchmark is best read as a Pareto comparison across accuracy, "
            "latency, and adaptation cost rather than as a strict ranking."
        ),
    ],
    "The third limitation is the restricted hyperparameter budget": [
        (
            "A further limitation is that the study reports deterministic point forecasts. Many "
            "operational grid and industrial planning tasks require prediction intervals or full "
            "probabilistic forecasts to support reserve allocation and risk-aware scheduling. "
            "Extending the protocol to quantile losses, calibration metrics, and probabilistic "
            "foundation models would make the comparison more directly applicable to reliability-"
            "constrained decision making."
        ),
        (
            "Finally, the benchmark uses two public hourly datasets. This choice supports "
            "reproducibility and allows other researchers to verify the reported results, but it "
            "does not cover all STLF regimes. Future work should repeat the protocol on private "
            "industrial sites, weather-sensitive residential feeders, and datasets with known "
            "calendar events. Such studies would clarify whether the relative strength of "
            "XGBoost, DLinear, and foundation models transfers beyond standard public benchmarks."
        ),
    ],
}


def _insert_after(paragraph, text: str):
    new_paragraph = paragraph.insert_paragraph_before(text)
    paragraph._p.addnext(new_paragraph._p)
    new_paragraph.style = paragraph.style
    new_paragraph.alignment = paragraph.alignment
    return new_paragraph


def _set_no_proof(paragraph) -> None:
    """Disable Word spell/grammar marking for a paragraph while preserving style."""
    for run in paragraph.runs:
        rpr = run._r.get_or_add_rPr()
        if rpr.find(qn("w:noProof")) is None:
            rpr.append(OxmlElement("w:noProof"))
        lang = rpr.find(qn("w:lang"))
        if lang is None:
            lang = OxmlElement("w:lang")
            rpr.append(lang)
        lang.set(qn("w:val"), "en-US")
        lang.set(qn("w:eastAsia"), "en-US")


def polish_docx(path: Path) -> tuple[int, int]:
    doc = Document(path)
    replaced = 0
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        for prefix, new_text in REPLACEMENTS.items():
            if text.startswith(prefix):
                if paragraph.text != new_text:
                    paragraph.text = new_text
                    replaced += 1
                break
        _set_no_proof(paragraph)

    inserted = 0
    for prefix, paragraphs_to_insert in INSERT_AFTER.items():
        anchor = next((paragraph for paragraph in doc.paragraphs if paragraph.text.strip().startswith(prefix)), None)
        if anchor is None:
            continue
        cursor = anchor
        for text in paragraphs_to_insert:
            if any(paragraph.text.strip() == text for paragraph in doc.paragraphs):
                continue
            cursor = _insert_after(cursor, text)
            _set_no_proof(cursor)
            inserted += 1

    # Also mark table text as no-proof; tables themselves are not regenerated.
    table_runs = 0
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    _set_no_proof(paragraph)
                    table_runs += len(paragraph.runs)

    doc.save(path)
    return replaced + inserted, table_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx", type=Path, default=DEFAULT_DOCX)
    parser.add_argument("--backup", action="store_true",
                        help="Save a .bak copy next to the DOCX before editing.")
    args = parser.parse_args()

    if not args.docx.exists():
        raise SystemExit(f"File not found: {args.docx}")

    if args.backup:
        backup = args.docx.with_suffix(args.docx.suffix + ".bak")
        shutil.copy2(args.docx, backup)
        print(f"Backup -> {backup}")

    replaced, table_runs = polish_docx(args.docx)
    print(f"Expanded paragraphs : {replaced}")
    print(f"Table runs checked  : {table_runs}")
    print(f"Saved -> {args.docx}")


if __name__ == "__main__":
    main()
