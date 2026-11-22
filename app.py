"""Student Performance Classifier - Shiny for Python app.

Loads the trained pipeline (model.pkl) and the results saved by train_model.py.
The model is NOT retrained when the app starts.
"""
import base64
import functools
import io
import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

import charts

ROOT = Path(__file__).parent

#Developer's info
DEVELOPERS = ["Librea, Marrah Princess C."]
COURSE_SECTION = "BSCS-DS 3"
UNIVERSITY = "University of Southeastern Philippines"
ACADEMIC_YEAR = "2026-2027"
DEVELOPER_EMAIL = "mpclibrea01202401033@usep.edu.ph"
# =====================================================================



def _find_bundles():
    """Look for model folders under models/, each holding model.pkl and model_metrics.json.
    A run is classified by its metrics: 'leak_check' marks the original-data run."""
    found = {}
    for p in sorted((ROOT / "models").glob("*/model.pkl")):
        m = json.loads((p.parent / "model_metrics.json").read_text())
        _csv = p.parent / "feature_importance.csv"
        found["original" if "leak_check" in m else "dedup"] = {
            "model": joblib.load(p), "M": m, "dir": p.parent,
            "IMP": pd.read_csv(_csv) if _csv.exists() else None}
    return found if {"original", "dedup"} <= set(found) else None


BUNDLES = _find_bundles()
TWO_MODELS = BUNDLES is not None
if TWO_MODELS:  # both models available: the original-data run is the primary one
    model, M = BUNDLES["original"]["model"], BUNDLES["original"]["M"]
    _imp = BUNDLES["original"]["dir"] / "feature_importance.csv"
    IMP = pd.read_csv(_imp if _imp.exists() else ROOT / "feature_importance.csv")
else:  # single model in the project root
    model = joblib.load(ROOT / "model.pkl")
    M = json.loads((ROOT / "model_metrics.json").read_text())
    IMP = pd.read_csv(ROOT / "feature_importance.csv")
RAW = pd.read_csv(ROOT / "data" / "augmented_balanced_dataset_m.csv")
DF = RAW.drop_duplicates().reset_index(drop=True)  # the 2,720 unique records

FEATURES = M["features"]
L = charts.LABELS
T = M["test"]
D = M["dataset"]

# Metrics files differ between runs. Normalize them so the labels always match the numbers.
if "leak_check" in M:  # run evaluated on the original split; T is the original-data result
    _lk = M["leak_check"]
    EVAL = "original data"
    ORIG = {"acc": T["accuracy"], "n": D["test_rows"]}
    DEDUP = {"acc": _lk["dedup_split_accuracy"], "ci": _lk["dedup_split_ci95"], "n": _lk["dedup_split_test_rows"]}
    DUP_TEST = (_lk["test_rows_with_copy_in_train"], _lk["test_rows"])
    CLEAN_NOTE = f"duplicate rows kept as supplied ({D['exact_duplicate_copies']})."
    SPLIT_NOTE = ""
    DATA_NOTE = f"augmented file with {D['exact_duplicate_copies']} duplicate rows, present in both training and test data."
else:  # run evaluated on deduplicated data; T is the deduplicated result
    _rs = M["raw_split_comparison"]
    EVAL = "deduplicated data"
    ORIG = {"acc": _rs["test_accuracy"], "n": _rs["test_rows"]}
    DEDUP = {"acc": T["accuracy"], "ci": T["accuracy_ci95"], "n": D["test_rows"]}
    DUP_TEST = (_rs["test_rows_with_exact_copy_in_train"], _rs["test_rows"])
    CLEAN_NOTE = "duplicate rows removed before splitting."
    SPLIT_NOTE = " after deduplication"
    DATA_NOTE = f"augmented file; {D['exact_duplicate_copies']} duplicate rows removed."
ERR_NOTE = f"about {1 - T['accuracy']:.0%} of test predictions were incorrect."
SEL_NOTE = (" Selection used training data only; the test set was evaluated once."
            if M.get("protocol", {}).get("test_evaluations") == 1 else "")
MODEL_NAMES = {"original": "Original-data model", "dedup": "Deduplicated model"}
MODELS_LI = ""
if TWO_MODELS:
    _mb = BUNDLES["dedup"]["M"]
    DEDUP = {"acc": _mb["test"]["accuracy"], "ci": _mb["test"]["accuracy_ci95"], "n": _mb["dataset"]["test_rows"]}
    CLEAN_NOTE = f"duplicate rows ({D['exact_duplicate_copies']}) kept for the original-data model and removed for the deduplicated model."
    DATA_NOTE = (f"augmented file with {D['exact_duplicate_copies']} duplicate rows; kept in the original-data model, "
                 f"removed in the deduplicated model. {DUP_TEST[0]} of {DUP_TEST[1]} original-data test rows also appear in training.")
    ERR_NOTE = (f"incorrect test predictions: about {1 - ORIG['acc']:.0%} (original data), "
                f"about {1 - DEDUP['acc']:.0%} (deduplicated).")
    MODELS_LI = (f"<li><b>Models:</b> original-data model ({M['selected_model']}) and deduplicated model "
                 f"({_mb['selected_model']}); either can be chosen on Predict.</li>")
CORR = pd.DataFrame(M["correlation"]).loc[FEATURES, FEATURES]
STATS = {s["feature"]: s for s in M["feature_stats"]}
IMP_ORDER = list(IMP.sort_values("perm_mean", ascending=False)["feature"])
IMP_MAP = IMP.set_index("feature")
DROP = {d["feature"]: d["cv_accuracy_without"] for d in M["ablation"]["drop_one"]}
BANDS = {b["band"]: b for b in M["confidence_bands"]}


def _view(Mb, imp):
    """Everything on Insights that belongs to one model."""
    return {"M": Mb, "T": Mb["test"], "D": Mb["dataset"], "IMP": imp,
            "EVAL": "original data" if "leak_check" in Mb else "deduplicated data",
            "DROP": {d["feature"]: d["cv_accuracy_without"] for d in Mb["ablation"]["drop_one"]}}


VIEWS = {"original": _view(M, IMP)}
if TWO_MODELS:
    VIEWS["dedup"] = _view(BUNDLES["dedup"]["M"], BUNDLES["dedup"]["IMP"])


def _order(v):
    """Activities from most to least important for one model."""
    if v["IMP"] is not None:
        return list(v["IMP"].sort_values("perm_mean", ascending=False)["feature"])
    return sorted(FEATURES, key=lambda f: v["DROP"][f])

IDS = {"Submission.Actions": "submission", "Quiz.Actions": "quiz", "Forum.Actions": "forum",
       "Create.Actions": "create", "Read.Actions": "read", "Update.Actions": "update",
       "Delete.Actions": "delete", "Days.Logged.In": "days"}
MEDIAN_ALL = DF[FEATURES].median()
MEDIAN_HIGH = DF[DF.rating == "High"][FEATURES].median()
MEDIAN_LOW = DF[DF.rating == "Low"][FEATURES].median()
SORTED = {f: np.sort(DF[f].values) for f in FEATURES}


def fig_html(fig, alt):
    """Render a matplotlib figure as an inline PNG so its size/fonts are exactly as designed."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return ui.HTML(f'<img src="{src}" alt="{alt}" style="max-width:100%;height:auto;display:block;margin:0 auto">')


def plot_ui(alt):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper():
            return fig_html(fn(), alt)
        return render.ui(wrapper)
    return deco


def pctl(f, v):
    return 100.0 * np.searchsorted(SORTED[f], v, side="right") / len(SORTED[f])


def pc(x, d=1):
    return f"{100 * x:.{d}f}%"


def pval(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def table(headers, rows, num=(), hl=None):
    th = "".join(f'<th class="{"num" if i in num else ""}">{h}</th>' for i, h in enumerate(headers))
    body = ""
    for r_i, r in enumerate(rows):
        cls = ' class="hl"' if hl is not None and r_i == hl else ""
        body += f"<tr{cls}>" + "".join(
            f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
    return ui.HTML(f'<div class="tbl-wrap"><table class="t"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')


def kv(k, v):
    return ui.HTML(f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>')


def callout(kind, *children):
    return ui.div(*children, class_=f"callout {kind}")


def stat(label, number, detail):
    return ui.div(ui.div(label, class_="l"), ui.div(number, class_="n"), ui.div(detail, class_="d"), class_="stat")


def icon(name):
    paths = {
        "home": '<path d="M3 11l9-8 9 8v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
        "predict": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
        "insights": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
        "about": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>',
    }
    return ui.HTML(f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
                   f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>')


def nav_button(key, label, active=False):
    return ui.input_action_button(f"nav_{key}", ui.TagList(icon(key), ui.span(label)),
                                  class_="nav-btn" + (" active" if active else ""))


# ---------------------------------------------------------------- sidebar
sidebar = ui.div(
    ui.div(
        ui.div(ui.HTML('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" '
                       'stroke-linecap="round" stroke-linejoin="round"><path d="M3 20h18M6 20V11M12 20V5M18 20v-6"/></svg>'),
               class_="brand-mark"),
        ui.div(ui.div("Student Performance", class_="brand-name"),
               ui.div("LMS activity classifier", class_="brand-sub")),
        class_="brand"),
    ui.div(nav_button("home", "Home", True), nav_button("predict", "Predict"),
           nav_button("insights", "Insights"), nav_button("about", "About"), class_="nav"),
    ui.div(ui.div("Original data", class_="k"),
           ui.div(pc(ORIG["acc"]), class_="v"),
           ui.div("Deduplicated", class_="k", style="margin-top:10px"),
           ui.div(pc(DEDUP["acc"]), class_="v"),
           ui.p("Decision support only."),
           class_="side-card"),
    class_="side")

# ---------------------------------------------------------------- HOME
home = ui.div(
    ui.div(ui.h1("Student Performance Classifier"),
           ui.p("Predicts whether a student is likely to pass or fail from LMS activity."), class_="page-head"),
    ui.div(
        stat("Unique records", f"{D['unique_rows']:,}", f"from {D['rows_supplied']:,} rows"),
        stat("Model", M["selected_model"], f"{M['selected_params']['n_estimators']} trees"),
        stat("Original data", pc(ORIG["acc"]), f"{ORIG['n']} test records"),
        stat("Deduplicated", pc(DEDUP["acc"]), f"{DEDUP['n']} test records"),
        class_="stat-strip"),
    ui.div(
        ui.h2("How to use"),
        ui.div(
            ui.div(ui.div("1", class_="no"), ui.HTML("<b>Enter activity</b><span>Type eight LMS counts, or use a typical profile." + (" Choose a model." if TWO_MODELS else "") + "</span>"), class_="step"),
            ui.div(ui.div("2", class_="no"), ui.HTML("<b>Read the estimate</b><span>Likely outcome and confidence.</span>"), class_="step"),
            ui.div(ui.div("3", class_="no"), ui.HTML("<b>Check Insights</b><span>See which activities matter.</span>"), class_="step"),
            class_="steps"),
        ui.div(ui.input_action_button("cta_predict", "Make a prediction", class_="btn-cta p"),
               ui.input_action_button("cta_insights", "Explore insights", class_="btn-cta s"), class_="cta-row"),
        class_="panel"),
    ui.div(
        ui.h2("Accuracy"),
        kv("Original data", pc(ORIG["acc"])),
        kv("Deduplicated data", pc(DEDUP["acc"])),
        kv("Duplicate rows", f"{D['exact_duplicate_copies']}"),
        kv(f"Majority-class baseline, {EVAL}", pc(D["majority_baseline_train"], 0)),
        class_="panel"),
    class_="page")

# ---------------------------------------------------------------- PREDICT
def _model_row(name, Mb, acc):
    d, t = Mb["dataset"], Mb["test"]
    return [name, f"{Mb['selected_model']} ({Mb['selected_params']['n_estimators']} trees)", f"{d['train_rows']:,}",
            f"{d['test_rows']:,}", pc(acc), pc(t["report"]["High"]["recall"]), pc(t["report"]["Low"]["recall"])]


MODEL_HEADERS = ["Model", "Algorithm", "Train", "Test", "Accuracy", "High recall", "Low recall"]
if TWO_MODELS:
    MODEL_ROWS = [_model_row("Original data", BUNDLES["original"]["M"], ORIG["acc"]),
                  _model_row("Deduplicated", BUNDLES["dedup"]["M"], DEDUP["acc"])]
else:
    MODEL_ROWS = [_model_row("Original data" if EVAL == "original data" else "Deduplicated", M, T["accuracy"])]


def field(f):
    i = IDS[f]
    return ui.div(
        ui.input_numeric(f"in_{i}", L[f], value=int(MEDIAN_ALL[f]), min=0, step=1),
        ui.div(f"Typical: {MEDIAN_ALL[f]:g}", class_="hint"), class_="field")


predict_page = ui.div(
    ui.div(ui.h1("Predict"), ui.p("Enter LMS activity counts to estimate whether a student is likely to pass or fail."), class_="page-head"),
    ui.div(
        ui.div(
            ui.h2("Student LMS activity"),
            ui.p("Whole numbers, 0 or more.", class_="sub"),
            *([ui.input_radio_buttons("model_pick", "Model",
                                      {"original": f"Original data ({pc(ORIG['acc'])})",
                                       "dedup": f"Deduplicated ({pc(DEDUP['acc'])})"},
                                      selected="original", inline=True)] if TWO_MODELS else []),
            ui.div("Quick start", class_="preset-lab"),
            ui.div(ui.input_action_button("preset_high", "Typical High values", class_="btn-ghost"),
                   ui.input_action_button("preset_low", "Typical Low values", class_="btn-ghost"),
                   ui.input_action_button("preset_reset", "Reset", class_="btn-ghost"), class_="preset-row"),
            ui.div(*[field(f) for f in FEATURES], class_="field-grid"),
            ui.input_action_button("predict", "Predict outcome", class_="btn-primary-x"),
            class_="panel"),
        ui.div(
            ui.output_ui("result_panel"),
            ui.output_ui("result_charts"),
        ),
        class_="grid-predict"),
    ui.div(
        ui.h2("Model"),
        table(MODEL_HEADERS, MODEL_ROWS, num={2, 3, 4, 5, 6}),
        class_="panel"),
    class_="page")

# ---------------------------------------------------------------- INSIGHTS
sel_choices = {f: L[f] for f in IMP_ORDER}
insights_page = ui.div(
    ui.div(ui.h1("Insights"), ui.p("Data, model and reliability."), class_="page-head"),
    *([ui.input_radio_buttons("ins_model", "Model",
                              {"original": f"Original data ({pc(ORIG['acc'])})",
                               "dedup": f"Deduplicated ({pc(DEDUP['acc'])})"},
                              selected="original", inline=True)] if TWO_MODELS else []),

    ui.div(ui.span("Summary"), class_="section-title"),
    ui.output_ui("findings"),

    ui.div(ui.span("A. Dataset overview"), class_="section-title"),
    ui.div(
        stat("Rows in file", f"{D['rows_supplied']:,}", "before cleaning"),
        stat("Duplicate rows", f"{D['exact_duplicate_copies']}", f"{pc(D['exact_duplicate_copies'] / D['rows_supplied'], 0)} of the file"),
        stat("Unique records", f"{D['unique_rows']:,}", "after deduplication"),
        stat("High share (unique)", pc(D['class_counts_unique']['High'] / D['unique_rows'], 0), f"{D['class_counts_unique']['High']:,} High, {D['class_counts_unique']['Low']:,} Low"),
        class_="stat-strip"),
    ui.div(
        ui.div(ui.h2("Class balance"),
               ui.output_ui("plot_balance"), class_="panel"),
        ui.div(ui.h2("Data quality"),
               kv("Duplicate rows", f"{D['exact_duplicate_copies']}"),
               kv("Conflicting-label groups", f"{D['conflicting_label_groups']}"),
               ui.output_ui("desc_table"), class_="panel"),
        class_="grid-2"),

    ui.div(ui.span("B. Student activity analysis"), class_="section-title"),
    ui.div(
        ui.h2("Activity by rating (log scale)"),
        ui.input_select("feat_pick", None, sel_choices, selected=IMP_ORDER[0]),
        ui.output_ui("plot_dist"),
        class_="panel"),
    ui.div(
        ui.div(ui.h2("High vs Low comparison"),
               ui.output_ui("compare_table"),
               class_="panel"),
        ui.div(ui.h2("Correlation (Spearman)"),
               ui.output_ui("plot_heat"), class_="panel"),
        class_="grid-2"),

    ui.div(ui.span("C. Model performance"), class_="section-title"),
    ui.output_ui("perf_stats"),
    ui.div(
        ui.div(ui.output_ui("cm_head"),
               ui.output_ui("plot_cm"), class_="panel"),
        ui.div(ui.output_ui("rep_head"),
               ui.output_ui("report_table"),
               ui.h3("Models compared (5-fold CV)"),
               ui.output_ui("comparison_table"), class_="panel"),
        class_="grid-2"),
    ui.div(
        ui.h2("Original vs deduplicated data"),
        ui.output_ui("target_note"), class_="panel"),

    ui.div(ui.span("D. What the model relies on"), class_="section-title"),
    ui.div(
        ui.div(ui.h2("Permutation importance"),
               ui.output_ui("imp_panel"),
               class_="panel"),
        ui.div(ui.h2("Drop-one check"),
               ui.output_ui("imp_table"), class_="panel"),
        class_="grid-2"),
    ui.div(
        ui.h2("Effect of each activity"),
        ui.input_select("pdp_pick", None, sel_choices, selected=IMP_ORDER[0]),
        ui.output_ui("plot_pdp"), class_="panel"),
    ui.div(ui.span("E. Reliability"), class_="section-title"),
    ui.div(
        ui.div(ui.h2("Calibration"),
               ui.output_ui("plot_cal"), class_="panel"),
        ui.div(ui.h2("Accuracy by confidence (test records)"),
               ui.output_ui("conf_table"), class_="panel"),
        class_="grid-2"),
    class_="page")

# ---------------------------------------------------------------- ABOUT
VAR_DESC = {
    "Submission.Actions": "Number of submission-related actions",
    "Quiz.Actions": "Number of quiz-related actions",
    "Forum.Actions": "Number of forum-related actions",
    "Create.Actions": "Number of create-related actions",
    "Read.Actions": "Number of read-related actions",
    "Update.Actions": "Number of update-related actions",
    "Delete.Actions": "Number of delete-related actions",
    "Days.Logged.In": "Number of days the student logged into the LMS",
}
about_page = ui.div(
    ui.div(ui.h1("About the project"), ui.p("Pass or fail prediction from LMS activity."), class_="page-head"),
    ui.div(
        ui.div(
            ui.div(ui.h2("Purpose"),
                   ui.p("Estimates whether a student is likely to pass or fail from LMS activity, to help identify who may need support early."),
                   ui.p("High is read as likely to pass and Low as likely to fail. The dataset provides High and Low labels only."), class_="panel"),
            ui.div(ui.h2("Dataset"),
                   ui.p(f"augmented_balanced_dataset_m.csv: {D['rows_supplied']:,} rows, 8 predictors, 1 target (rating). {D['exact_duplicate_copies']} duplicate rows; {D['unique_rows']:,} unique ({D['class_counts_unique']['High']:,} High, {D['class_counts_unique']['Low']:,} Low). Source not stated."),
                   table(["Variable", "Description"], [[f"<code>{f}</code>", VAR_DESC[f]] for f in FEATURES] +
                         [["<code>rating</code>", "Target class: High or Low"]]), class_="panel"),
            ui.div(ui.h2("Method"),
                   ui.HTML(f"""<ul class="clean">
                   <li><b>Cleaning:</b> {CLEAN_NOTE} No missing values; Read.Actions has extreme outliers (max {int(DF['Read.Actions'].max()):,}).</li>
                   <li><b>Split:</b> stratified 80/20 ({D['train_rows']:,} train, {D['test_rows']:,} test{SPLIT_NOTE}).</li>
                   <li><b>Evaluation:</b> 80/20 split on original and deduplicated data.</li>
                   <li><b>Selection:</b> seven classifiers compared with 5-fold cross-validation, then tuned.{SEL_NOTE}</li>
                   <li><b>Final model:</b> {M['selected_model']} ({M['selected_params']['n_estimators']} trees), saved with Joblib.</li>
                   {MODELS_LI}
                   <li><b>Interpretation:</b> permutation importance, drop-one checks, partial dependence, calibration.</li></ul>"""),
                   class_="panel"),
            ui.div(ui.h2("Model results"),
                   ui.p(f"{EVAL.capitalize()} ({D['test_rows']} test records): {pc(T['accuracy'])} accuracy, 95% CI {pc(T['accuracy_ci95'][0])}–{pc(T['accuracy_ci95'][1])}."),
                   ui.output_ui("about_target"), class_="panel"),
            ui.div(ui.h2("Interpreting a prediction"),
                   ui.HTML("""<ul class="clean">
                   <li><b>Likely to pass:</b> activity resembles High-rated students.</li>
                   <li><b>Likely to fail:</b> activity resembles Low-rated students.</li>
                   <li><b>Use:</b> a support cue alongside human judgment.</li></ul>"""),
                   class_="panel"),
            ui.div(ui.h2("Limitations"),
                   ui.HTML(f"""<ul class="clean">
                   <li><b>Data:</b> {DATA_NOTE} Near-copies may remain.</li>
                   <li><b>Measures:</b> LMS click counts. Offline study is not captured; the rating's definition is not documented.</li>
                   <li><b>Scope:</b> one course population; very large values are sparse in training.</li>
                   <li><b>Interpretation:</b> importance reflects model use, not cause.</li>
                   <li><b>Errors:</b> {ERR_NOTE}</li></ul>"""),
                   class_="panel"),
            class_="col-main"),
        ui.div(
            ui.div(ui.h2("At a glance"),
                   kv("Records (unique)", f"{D['unique_rows']:,}"), kv("Predictors", f"{D['n_predictors']}"),
                   kv("Algorithm", M["selected_model"]), kv("Trees", f"{M['selected_params']['n_estimators']}"),
                   kv("Accuracy, original data", pc(ORIG["acc"])), kv("Accuracy, deduplicated data", pc(DEDUP["acc"])),
                   class_="panel"),
            ui.div(ui.h2("Built with"),
                   kv("Language", "Python"), kv("Framework", "Shiny for Python"),
                   kv("Model package", "scikit-learn"), kv("Data", "pandas, NumPy"),
                   kv("Charts", "Matplotlib"), kv("Model file", "Joblib"), class_="panel"),
            ui.div(ui.h2("Developer information"), kv("Developer", DEVELOPERS[0]), kv("Email", DEVELOPER_EMAIL),
                   kv("University", UNIVERSITY), class_="panel"),
            class_="col-side"),
        class_="grid-2-wide", style="align-items:start"),
    class_="page")

# ---------------------------------------------------------------- page shell
NAV_JS = """
Shiny.addCustomMessageHandler('nav_active', function(p){
  document.querySelectorAll('.nav-btn').forEach(function(b){b.classList.toggle('active', b.id==='nav_'+p);});
  window.scrollTo({top:0});
});
"""
app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap"),
        ui.tags.link(rel="stylesheet", href="styles.css"),
    ),
    ui.div(
        sidebar,
        ui.div(
            ui.navset_hidden(
                ui.nav_panel(None, home, value="home"),
                ui.nav_panel(None, predict_page, value="predict"),
                ui.nav_panel(None, insights_page, value="insights"),
                ui.nav_panel(None, about_page, value="about"),
                id="page"),
            class_="main"),
        class_="shell"),
    ui.tags.script(NAV_JS),
    title="Student Performance Classifier",
)


# ---------------------------------------------------------------- server
def server(input, output, session):
    # ---- navigation
    async def go(p):
        ui.update_navset("page", selected=p)
        await session.send_custom_message("nav_active", p)

    def wire(trigger, page):
        @reactive.effect
        @reactive.event(trigger)
        async def _():
            await go(page)

    for k in ("home", "predict", "insights", "about"):
        wire(input[f"nav_{k}"], k)
    wire(input.cta_predict, "predict")
    wire(input.cta_insights, "insights")

    # ---- presets
    def fill(series):
        for f in FEATURES:
            ui.update_numeric(f"in_{IDS[f]}", value=float(round(series[f])))

    @reactive.effect
    @reactive.event(input.preset_high)
    def _():
        fill(MEDIAN_HIGH)

    @reactive.effect
    @reactive.event(input.preset_low)
    def _():
        fill(MEDIAN_LOW)

    @reactive.effect
    @reactive.event(input.preset_reset)
    def _():
        fill(MEDIAN_ALL)

    # ---- prediction
    def read_inputs():
        vals, errs = {}, []
        for f in FEATURES:
            v = input[f"in_{IDS[f]}"]()
            if v is None or (isinstance(v, float) and math.isnan(v)):
                errs.append(f"{L[f]}: enter a number.")
            elif v < 0:
                errs.append(f"{L[f]}: cannot be negative.")
            elif v != int(v):
                errs.append(f"{L[f]}: use a whole number.")
            elif v > 10_000_000:
                errs.append(f"{L[f]}: value is too large to be realistic.")
            else:
                vals[f] = int(v)
        return vals, errs

    _triggers = (input.predict, input.model_pick) if TWO_MODELS else (input.predict,)

    @reactive.calc
    @reactive.event(*_triggers)
    def result():
        vals, errs = read_inputs()
        if errs:
            return {"errors": errs}
        pick = input.model_pick() if TWO_MODELS else "original"
        mdl, Mb = (BUNDLES[pick]["model"], BUNDLES[pick]["M"]) if TWO_MODELS else (model, M)
        row = pd.DataFrame([vals], columns=FEATURES)
        classes = list(mdl.classes_)

        def p_high(r):
            return float(mdl.predict_proba(r)[0][classes.index("High")])

        p = p_high(row)
        contrib = {}
        for f in FEATURES:
            alt = row.copy()
            alt[f] = MEDIAN_ALL[f]
            contrib[f] = p - p_high(alt)
        label = "High" if p >= 0.5 else "Low"
        conf = max(p, 1 - p)
        k = min(int(round((conf - 0.5) / 0.1, 6)), 4)
        lo = 50 + 10 * k
        band = {b["band"]: b for b in Mb["confidence_bands"]}.get(f"{lo}-{min(lo + 10, 100)}%")
        high_vals = [L[f] for f in FEATURES if vals[f] > Mb["feature_p99"][f]]
        return {"vals": vals, "p": p, "label": label, "conf": conf, "band": band,
                "contrib": contrib, "outliers": high_vals, "model": pick}

    @render.ui
    def result_panel():
        if input.predict() == 0:
            return ui.div(ui.div(ui.tags.b("No prediction yet"),
                                 "Enter values, then press Predict outcome."), class_="empty panel")
        r = result()
        if "errors" in r:
            return ui.div(ui.h2("Please fix the inputs"),
                          callout("err", ui.HTML("<br>".join(r["errors"]))), class_="panel")
        hi = r["label"] == "High"
        pct_h, pct_l = 100 * r["p"], 100 * (1 - r["p"])
        bits = []
        if TWO_MODELS:
            acc_b = ORIG["acc"] if r["model"] == "original" else DEDUP["acc"]
            bits.append(ui.p(f"{MODEL_NAMES[r['model']]}: {pc(acc_b)} test accuracy.", class_="sub"))
        if r["band"]:
            bits.append(callout("info", ui.HTML(
                f"Accuracy at {r['band']['band']} confidence: <b>{pc(r['band']['accuracy'], 0)}</b> ({r['band']['n']} test records).")))
        if r["outliers"]:
            bits.append(callout("info", ui.HTML(
                "Above the 99th percentile of training data: " + ", ".join(r["outliers"]) + ".")))
        return ui.div(
            ui.h2("Result"),
            ui.div(ui.div("Likely to pass" if hi else "Likely to fail", class_=f"badge-res {'high' if hi else 'low'}"),
                   ui.div(ui.HTML(f"Model confidence<br><b>{pct_h if hi else pct_l:.1f}%</b>"), class_="conf"),
                   class_="result-top"),
            ui.HTML(f'<div class="prob"><div class="h" style="width:{pct_h:.1f}%"></div><div class="l" style="width:{pct_l:.1f}%"></div></div>'
                    f'<div class="prob-legend"><span>Pass <b>{pct_h:.1f}%</b></span><span>Fail <b>{pct_l:.1f}%</b></span></div>'),
            ui.div(*bits, style="margin-top:14px"), class_="panel")

    @render.ui
    def result_charts():
        if input.predict() == 0:
            return ui.div()
        r = result()
        if "errors" in r:
            return ui.div()
        return ui.div(
            ui.div(ui.h2("Where this student sits"),
                   ui.output_ui("plot_profile"), class_="panel"),
            ui.div(ui.h2("What pushed this prediction"),
                   ui.output_ui("plot_local"), class_="panel"))

    @plot_ui("Where the entered student sits compared with typical High and Low students")
    def plot_profile():
        r = result()
        return charts.fig_profile({f: pctl(f, r["vals"][f]) for f in FEATURES},
                                  {f: pctl(f, MEDIAN_HIGH[f]) for f in FEATURES},
                                  {f: pctl(f, MEDIAN_LOW[f]) for f in FEATURES})

    @plot_ui("Bar chart of how each activity moved the predicted chance of High")
    def plot_local():
        return charts.fig_local(result()["contrib"])

    # ---- insights: the selected model
    @reactive.calc
    def cur():
        return VIEWS[input.ins_model()] if TWO_MODELS else VIEWS["original"]

    @render.ui
    def perf_stats():
        v = cur(); t = v["T"]
        return ui.div(
            stat(f"Accuracy, {v['EVAL']}", pc(t["accuracy"]), f"95% CI {pc(t['accuracy_ci95'][0])}\u2013{pc(t['accuracy_ci95'][1])}"),
            stat("Balanced accuracy", pc(t["balanced_accuracy"]), "mean recall"),
            stat("Macro F1", pc(t["macro_f1"]), "mean F1"),
            stat("ROC AUC", f"{t['roc_auc']:.2f}", "area under curve"),
            class_="stat-strip")

    @render.ui
    def cm_head():
        v = cur()
        return ui.h2(f"Confusion matrix ({v['EVAL']}, n = {v['D']['test_rows']})")

    @render.ui
    def rep_head():
        return ui.h2(f"Per-class results ({cur()['EVAL']})")

    # ---- insights: charts
    @plot_ui("Class counts in the original file and after removing duplicates")
    def plot_balance():
        return charts.fig_class_balance(D["class_counts_supplied"], D["class_counts_unique"])

    @plot_ui("Distribution of the chosen activity for High and Low")
    def plot_dist():
        return charts.fig_feature_dist(DF, input.feat_pick())

    @plot_ui("Correlation heatmap of LMS activities")
    def plot_heat():
        return charts.fig_heatmap(CORR)

    @plot_ui("Confusion matrix on the test set")
    def plot_cm():
        return charts.fig_confusion(cur()["T"]["confusion_matrix"])

    @render.ui
    def imp_panel():
        v = cur()
        if v["IMP"] is None:
            return ui.p("Not available for this model.")
        return fig_html(charts.fig_importance(v["IMP"]), "Permutation importance of each activity")

    @plot_ui("Partial dependence of predicted High probability")
    def plot_pdp():
        return charts.fig_pdp(cur()["M"]["pdp"], input.pdp_pick())

    @plot_ui("Calibration curve")
    def plot_cal():
        return charts.fig_calibration(cur()["M"]["calibration_curve"])

    # ---- insights: tables and text
    @render.ui
    def desc_table():
        rows = []
        for d in M["describe"]:
            rows.append([L[d["feature"]], f"{d['mean']:,.1f}", f"{d['50%']:,.0f}", f"{d['max']:,.0f}"])
        return ui.div(ui.h3("Activity counts (unique records)"), table(["Activity", "Mean", "Median", "Max"], rows, num={1, 2, 3}))

    @render.ui
    def compare_table():
        rows = []
        for f in IMP_ORDER:
            s = STATS[f]
            rows.append([L[f], f"{s['median_high']:g}", f"{s['median_low']:g}", f"{s['p_high_gt_low']:.2f}", pval(s["mw_p_value"])])
        return table(["Activity", "Median High", "Median Low", "P(High > Low)", "p-value"], rows, num={1, 2, 3, 4})

    @render.ui
    def report_table():
        rep = cur()["T"]["report"]
        rows = [[c, pc(rep[c]["precision"]), pc(rep[c]["recall"]), pc(rep[c]["f1-score"]), f"{int(rep[c]['support'])}"] for c in ("High", "Low")]
        return table(["Class", "Precision", "Recall", "F1", "Records"], rows, num={1, 2, 3, 4})

    @render.ui
    def comparison_table():
        rows = [[c["model"], pc(c["cv_accuracy"]), pc(c["cv_balanced_accuracy"])] for c in cur()["M"]["model_comparison"]]
        return table(["Model", "Accuracy", "Balanced acc."], rows, num={1, 2}, hl=0)

    def target_block():
        return [
            kv("Accuracy, original data", pc(ORIG["acc"])),
            kv("Accuracy, deduplicated data", pc(DEDUP["acc"])),
            kv("Test rows duplicated in training", f"{DUP_TEST[0]} of {DUP_TEST[1]}"),
            kv("Algorithms compared", f"{len(cur()['M']['model_comparison'])}"),
            kv("Best cross-validated accuracy", pc(cur()["M"]["model_comparison"][0]["cv_accuracy"])),
        ]

    @render.ui
    def target_note():
        return ui.div(*target_block())

    @render.ui
    def about_target():
        if EVAL == "original data":
            return kv("Accuracy, deduplicated data", pc(DEDUP["acc"]))
        return kv("Accuracy, original data", pc(ORIG["acc"]))

    @render.ui
    def imp_table():
        v = cur()
        full = v["M"]["ablation"]["full_cv_accuracy"]
        rows = []
        for f in _order(v):
            shuffle = f"{100 * v['IMP'].set_index('feature').loc[f, 'perm_mean']:+.1f} pts" if v["IMP"] is not None else "\u2013"
            rows.append([L[f], shuffle, f"{100 * (v['DROP'][f] - full):+.1f} pts"])
        return table(["Activity", "Shuffle drop", "Retrain without"], rows, num={1, 2})

    @render.ui
    def conf_table():
        rows = [[b["band"], f"{b['n']}", pc(b["accuracy"], 0)] for b in cur()["M"]["confidence_bands"]]
        return table(["Model confidence", "Test records", "Actually correct"], rows, num={1, 2})

    @render.ui
    def findings():
        v = cur(); t, Dv, Mv = v["T"], v["D"], v["M"]
        order = _order(v)
        top3 = order[:3]
        q, fo, dy = STATS["Quiz.Actions"], STATS["Forum.Actions"], STATS["Days.Logged.In"]
        rep = t["report"]
        lowb = [b for b in Mv["confidence_bands"] if b["band"] in ("50-60%", "60-70%")]
        n_low = sum(b["n"] for b in lowb)
        lo_acc, hi_acc = min(b["accuracy"] for b in lowb), max(b["accuracy"] for b in lowb)
        items = [
            ("Top activities", f"{L[top3[0]]}, {L[top3[1]]}, {L[top3[2]]}"),
            ("Median quiz actions", f"High {q['median_high']:g}, Low {q['median_low']:g}"),
            ("Median forum actions", f"High {fo['median_high']:g}, Low {fo['median_low']:g}"),
            ("Median days logged in", f"High {dy['median_high']:g}, Low {dy['median_low']:g}"),
            ("Confidence under 70%", f"{n_low} of {Dv['test_rows']} test records, {pc(lo_acc, 0)} to {pc(hi_acc, 0)} correct"),
            ("F1 by class", f"High {pc(rep['High']['f1-score'], 0)}, Low {pc(rep['Low']['f1-score'], 0)}"),
        ]
        return ui.div(*[ui.div(ui.div(str(i + 1), class_="ic"), ui.div(ui.tags.b(t), ui.p(b)), class_="finding")
                        for i, (t, b) in enumerate(items)], class_="panel")


app = App(app_ui, server, static_assets=ROOT / "assets")
