import json
from html import escape
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

import ui
from pipeline import C, COLS, ContractError, features, midterm_totals, validate

HERE = Path(__file__).parent
APP_NAME, SUBTITLE = "EduGuard", "Midterm early warning from LMS activity"
AUTHOR = "Cristine Jean Mahilum · BSCS DS-3"
TOP_SHARE = 0.15
PAGES = ["Overview", "Check a student", "Check a class", "How it works"]
INFO = ["Student.ID", "Student.Name", "Class"]
PRESETS = {
    "Low activity": dict(zip(COLS, [2, 0, 0, 3, 40, 0, 0, 4])),
    "Typical activity": dict(zip(COLS, [6, 40, 3, 10, 150, 4, 0, 14])),
    "High activity": dict(zip(COLS, [12, 120, 15, 25, 600, 12, 0, 30])),
}

st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide")
ui.inject()


# ---------- data + models ----------
@st.cache_resource
def load_models():
    out = {}
    for mid in "abc":
        pkl, card = HERE / f"model_{mid}.pkl", HERE / f"metrics_{mid}.json"
        if pkl.exists() and card.exists():
            bundle = joblib.load(pkl)
            if hasattr(bundle["model"], "n_jobs"):
                bundle["model"].n_jobs = 1   # many users share one small server: 1 core per request
            out[mid.upper()] = {"bundle": bundle, "card": json.load(open(card))}
    return out


@st.cache_data
def load_demo():
    files = sorted((HERE / "demo").glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


MODELS, DEMO = load_models(), load_demo()


def predict(mid, df8):
    """Fail risk + label for rows with the 8 input columns."""
    b = MODELS[mid]["bundle"]
    X = features(df8)[b["features"]]
    risk = b["model"].predict_proba(X)[:, list(b["model"].classes_).index(b["fail_class"])]
    return pd.DataFrame({"Prediction": ["Fail" if r >= b["cutoff"] else "Pass" for r in risk],
                         "Fail risk": risk.round(3)}, index=df8.index)


@st.cache_resource
def explainer(mid):
    import shap
    return shap.TreeExplainer(MODELS[mid]["bundle"]["model"])


def why(mid, row8):
    b = MODELS[mid]["bundle"]
    sv = explainer(mid).shap_values(features(row8)[b["features"]])
    k = list(b["model"].classes_).index(b["fail_class"])
    vals = sv[0][:, k] if getattr(sv, "ndim", 0) == 3 else (sv[k][0] if isinstance(sv, list) else sv[0])
    s = pd.Series(vals, index=b["features"])
    return s.reindex(s.abs().sort_values(ascending=False).index[:6])


@st.cache_data(show_spinner=False)
def demo_scores(mid, cls):
    """Score demo classes once per model and class, then reuse."""
    return score_table(mid, DEMO if cls == "All demo classes" else DEMO[DEMO["Class"] == cls])


def score_table(mid, weekly):
    mt = midterm_totals(weekly)
    out = pd.concat([mt[INFO], predict(mid, mt[COLS]), mt[COLS]], axis=1)
    return out.sort_values("Fail risk", ascending=False).reset_index(drop=True)


def dark(chart):
    return (chart.configure(background="transparent")
            .configure_axis(labelColor=ui.MUTED, titleColor=ui.MUTED, gridColor="#222", domainColor="#333")
            .configure_legend(labelColor=ui.MUTED, titleColor=ui.MUTED).configure_view(stroke=None))


def keep(key):
    """Segmented buttons can be clicked off. Put the last choice back so a state is always picked."""
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get(f"_last_{key}")
    st.session_state[f"_last_{key}"] = st.session_state[key]


def safe_csv(df):
    """Stop spreadsheet formula injection: text cells starting with = + - @ get a leading quote."""
    out = df.copy()
    for c in out.select_dtypes(include="object"):
        out[c] = out[c].astype(str).where(~out[c].astype(str).str.match(r"^[=+\-@]"), "'" + out[c].astype(str))
    return out.to_csv(index=False)


def go_predict(values):
    for c in COLS:
        st.session_state[f"in_{c}"] = int(values[c])
    st.session_state.page = "Check a student"


# ---------- sidebar ----------
with st.sidebar:
    ui.html(f'<div style="font-size:20px;font-weight:500;letter-spacing:-0.01em;color:#fff">{APP_NAME}</div>'
            f'<div class="muted" style="font-size:12px;margin-bottom:18px">{SUBTITLE}</div>')
    page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
    st.markdown("**Model**")
    st.session_state.setdefault("model", list(MODELS)[0])
    st.session_state.setdefault("_last_model", st.session_state.model)
    mid = st.segmented_control("Model", list(MODELS), key="model", on_change=keep, args=("model",),
                               format_func=lambda m: f"{m} · {MODELS[m]['card']['accuracy']:.1%}",
                               label_visibility="collapsed")
    card = MODELS[mid]["card"]
    st.caption(f"{card['name'].split(': ')[1].capitalize()}. {card['split']}")
    ui.card(f'<div class="kpi-label">Test set (kept out of training)</div>'
            f'<div style="color:#fff;font-weight:600">{card["test_rows"]} of 3,602 rows</div>'
            f'{ui.bar(card["test_rows"] / 3602, ui.VIOLET)}'
            f'<div class="muted" style="font-size:12px;margin-top:6px">Demo students are made up. Uploaded files are not stored.</div>',
            "dark")
    ui.html(f'<div class="author">{AUTHOR}<br>Machine Problem 3</div>')

# ---------- 1. dashboard ----------
if page == "Overview":
    a, b = st.columns([1.6, 1])
    with a:
        ui.card('<div class="h-hero">Find students who may need help before finals.</div>'
                '<div class="muted">EduGuard reads 8 LMS activity counts from the first half of the term (logins, reading, '
                'quizzes, submissions) and estimates who is likely to Fail. Instructors use the flag to reach out early.</div>',
                "hero")
    with b:
        ui.card(ui.rings([("Accuracy", card["accuracy"], ui.VIOLET),
                          ("At-risk students caught", card["recall_fail"], ui.EMERALD),
                          ("Ranking quality (ROC-AUC)", card["roc_auc"], ui.AMBER)]), "dark")

    k = st.columns(4)
    with k[0]: ui.kpi("Test accuracy", f"{card['accuracy']:.2%}", card["name"], ui.VIOLET)
    with k[1]: ui.kpi("At-risk students caught", f"{card['recall_fail']:.1%}", f"{card['precision_fail']:.0%} of flags are correct")
    with k[2]: ui.kpi("Rows in the dataset", "3,602", "2,709 unique · about 2,000 after near-copies", ui.MUTED)
    with k[3]: ui.kpi("Trees in the model", "500", card["algo"], ui.MUTED)

    if DEMO.empty:
        st.stop()
    scored = demo_scores(mid, "All demo classes")
    ui.html('<div class="section"></div>')
    left, right = st.columns([1.6, 1])
    with left:
        st.session_state.setdefault("view", "Activity map")
        st.session_state.setdefault("_last_view", "Activity map")
        mode = st.segmented_control("Demo classes", ["Activity map", "Weekly activity"], key="view",
                                    on_change=keep, args=("view",))
        if mode == "Weekly activity":
            wk = DEMO.assign(Total=DEMO[[c for c in COLS if c.endswith(".Actions")]].sum(axis=1))
            wk = wk.groupby(["Class", "Week"], as_index=False)["Total"].mean()
            ch = alt.Chart(wk).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("Week:O"), y=alt.Y("Total:Q", title="Average actions per student"),
                color=alt.Color("Class:N", scale=alt.Scale(range=[ui.VIOLET, ui.EMERALD, ui.AMBER])))
        else:
            med_d, med_r = scored["Days.Logged.In"].median(), scored["Read.Actions"].median()
            plot = scored.rename(columns={"Days.Logged.In": "Days", "Read.Actions": "Reads"})
            pts = alt.Chart(plot).mark_circle(size=90, opacity=.85).encode(
                x=alt.X("Days:Q", title="Days logged in"),
                y=alt.Y("Reads:Q", title="Read actions (log scale)", scale=alt.Scale(type="symlog")),
                color=alt.Color("Prediction:N", scale=alt.Scale(domain=["Fail", "Pass"], range=[ui.AMBER, ui.VIOLET])),
                tooltip=[alt.Tooltip("Student\\.ID:N", title="ID"), alt.Tooltip("Student\\.Name:N", title="Name"),
                         "Prediction:N", alt.Tooltip("Fail risk:Q", format=".0%")])
            rx = alt.Chart(pd.DataFrame({"x": [med_d]})).mark_rule(strokeDash=[4, 4], color="#555").encode(x="x:Q")
            ry = alt.Chart(pd.DataFrame({"y": [med_r]})).mark_rule(strokeDash=[4, 4], color="#555").encode(y="y:Q")
            ch = pts + rx + ry
        st.altair_chart(dark(ch.properties(height=330)), width="stretch")
    with right:
        st.markdown("### Highest risk right now")
        for _, r in scored.head(5).iterrows():
            c1, c2 = st.columns([3, 1])
            c1.markdown(f'{ui.badge(r["Prediction"])} **{escape(r["Student.Name"])}** '
                        f'<span class="muted">{escape(str(r["Student.ID"]))} · risk {r["Fail risk"]:.0%}</span>', unsafe_allow_html=True)
            c2.button("Check", key=f"w_{r['Student.ID']}", on_click=go_predict, args=(r[COLS].to_dict(),))

        n_fail = int((scored["Prediction"] == "Fail").sum())
        by_cls = scored.groupby("Class")["Prediction"].apply(lambda s: (s == "Fail").mean()).sort_values(ascending=False)
        late = DEMO["Student.ID"].nunique() - len(scored)
        alerts = [f"{n_fail} of {len(scored)} demo students are likely to Fail.",
                  f"{by_cls.index[0]} has the most students at risk ({by_cls.iloc[0]:.0%}).",
                  f"{late} student(s) don't have all 8 weeks logged yet, so there's no estimate for them."]
        if card["test_rows_with_copy_in_train"]:
            alerts.append(f"{card['test_rows_with_copy_in_train']} of {card['test_rows']} test rows also appear in "
                          f"training for {card['name'].split(':')[0]}, so its accuracy looks higher than it really is.")
        ui.card("<b style='color:#fff'>Things to look at</b>" +
                "".join(f'<div class="row"><span class="muted">{escape(t)}</span></div>' for t in alerts), "dark")

# ---------- 2. single predictor ----------
elif page == "Check a student":
    st.markdown("## Check a student")
    st.caption("Enter one student's LMS activity up to midterm, or start from an example.")
    for c in COLS:
        st.session_state.setdefault(f"in_{c}", PRESETS["Typical activity"][c])

    def apply_preset():
        for c, v in PRESETS[st.session_state.preset].items():
            st.session_state[f"in_{c}"] = v

    st.radio("Start from an example", list(PRESETS), key="preset", horizontal=True, index=None, on_change=apply_preset)

    left, right = st.columns([1.3, 1])
    with left:
        cols = st.columns(2)
        vals = {}
        for i, (c, r) in enumerate(C["inputs"].items()):
            vals[c] = cols[i % 2].number_input(f"{c} ({r['min']}–{r['max']})", min_value=r["min"],
                                               max_value=r["max"], step=1, key=f"in_{c}",
                                               on_change=lambda: st.session_state.update(preset=None))
        df = pd.DataFrame([vals])
        try:
            validate(df, with_target=False)
        except ContractError as e:
            st.error(f"Invalid input: {e}")
            st.stop()
        ui.html('<div class="section"></div>')
        st.markdown("### What the model calculates from these numbers")
        f = features(df).iloc[0]
        g = st.columns(3)
        for i, (name, formula) in enumerate(C["features"].items()):
            with g[i % 3]:
                ui.card(f'<div style="min-height:96px"><div class="kpi-label">{name}</div><div style="font-size:22px;color:#fff;font-weight:600">'
                        f'{f[name]:,.2f}</div><div class="formula">{escape(formula)}</div></div>', "dark")
    with right:
        r = predict(mid, df).iloc[0]
        p_fail = float(r["Fail risk"])
        fail = r["Prediction"] == "Fail"
        advice = ("Reach out this week. Ask how they're doing, check for missing work, and offer consultation or tutoring."
                  if fail else "Looks on track. Check again at the next checkpoint.")
        ui.card(f'<div class="kpi-label">Estimate · {escape(card["name"])}</div>'
                f'<div style="margin:6px 0 14px">{ui.badge(r["Prediction"])}</div>'
                f'<div class="row"><span class="muted">Chance of Fail</span><b style="color:#fbbf24">{p_fail:.0%}</b></div>'
                f'{ui.bar(p_fail, ui.AMBER)}'
                f'<div class="row"><span class="muted">Chance of Pass</span><b style="color:#c084fc">{1 - p_fail:.0%}</b></div>'
                f'{ui.bar(1 - p_fail, ui.VIOLET)}'
                f'<div class="muted" style="margin-top:14px">{advice}</div>'
                f'<div class="muted" style="font-size:12px;margin-top:8px">Flagged when the chance of Fail is {MODELS[mid]["bundle"]["cutoff"]:.0%} or more. '
                f'Based on LMS activity only: use it to start a conversation, not to grade.</div>')
        ui.html('<div class="section"></div>')
        st.markdown("### Same student, every model")
        rows = []
        for m in MODELS:
            p = predict(m, df).iloc[0]
            rows.append({"Model": m, "Test acc.": f"{MODELS[m]['card']['accuracy']:.1%}",
                         "Prediction": p["Prediction"], "Fail risk": f"{p['Fail risk']:.0%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---------- 3. batch scorer ----------
elif page == "Check a class":
    st.markdown("## Check a class")
    st.caption("Pick a demo class or upload a CSV. Students are sorted from highest to lowest risk.")
    src = st.selectbox("Class", ["All demo classes"] + sorted(DEMO["Class"].unique()) + ["Upload CSV"])
    weekly = None
    if src == "Upload CSV":
        up = st.file_uploader("CSV with the 8 activity columns. Optional: Student.ID, Student.Name, Class, Week (weekly rows are added up to midterm).", type="csv")
        if not up:
            st.stop()
        try:
            raw = pd.read_csv(up)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError):
            st.error("File rejected: not a readable CSV file.")
            st.stop()
        if raw.empty:
            st.error("File rejected: the CSV has no rows.")
            st.stop()
        try:
            if "Week" in raw:                       # weekly log -> midterm totals
                weekly = raw
                table = score_table(mid, raw)
            else:                                   # one row per student, already totals
                info = raw[[c for c in INFO if c in raw]]
                validate(raw.drop(columns=info.columns), with_target=False)
                if "Student.ID" not in info:
                    info = info.assign(**{"Student.ID": [f"Row {i + 1}" for i in range(len(raw))]})
                for c in INFO:
                    if c not in info:
                        info[c] = ""
                table = pd.concat([info[INFO], predict(mid, raw[COLS].astype(int)), raw[COLS].astype(int)], axis=1)
                table = table.sort_values("Fail risk", ascending=False).reset_index(drop=True)
        except ContractError as e:
            st.error(f"File rejected: {e}")
            st.stop()
    else:
        weekly = DEMO if src == "All demo classes" else DEMO[DEMO["Class"] == src]
        table = demo_scores(mid, src)

    n, n_fail = len(table), int((table["Prediction"] == "Fail").sum())
    k = st.columns(3)
    with k[0]: ui.kpi("Students checked", f"{n}", src, ui.MUTED)
    with k[1]: ui.kpi("Likely to Fail", f"{n_fail}", "start with the top 15%", ui.AMBER)
    with k[2]: ui.kpi("Share at risk", f"{n_fail / max(n, 1):.0%}", card["name"], ui.VIOLET)

    ui.html('<div class="section"></div>')
    s1, s2 = st.columns([2, 1])
    q = s1.text_input("Search by name or ID", "")
    st.session_state.setdefault("show", "All")
    st.session_state.setdefault("_last_show", "All")
    flt = s2.segmented_control("Show", ["All", "Likely Fail", "On track"], key="show", on_change=keep, args=("show",))
    view = table.copy()
    view.insert(0, "Top 15%", view.index < max(1, round(TOP_SHARE * n)))
    if q:
        view = view[view["Student.ID"].astype(str).str.contains(q, case=False)
                    | view["Student.Name"].astype(str).str.contains(q, case=False)]
    if flt == "Likely Fail":
        view = view[view["Prediction"] == "Fail"]
    elif flt == "On track":
        view = view[view["Prediction"] == "Pass"]
    st.dataframe(view.style.apply(
        lambda r: ["background-color: rgba(245,159,11,.16)" if r["Prediction"] == "Fail" else ""] * len(r), axis=1),
        hide_index=True, width="stretch", height=380,
        column_config={"Fail risk": st.column_config.ProgressColumn("Fail risk", min_value=0, max_value=1, format="percent")})
    export = pd.concat([table, features(table[COLS]).drop(columns=COLS)], axis=1)
    st.download_button("Download results (CSV)", safe_csv(export), "scored_students.csv", "text/csv")

    if len(table):
        with st.expander("Look closer at one student"):
            names = dict(zip(table["Student.ID"], table["Student.Name"]))   # one lookup, not one per option
            sid = st.selectbox("Student", table["Student.ID"], format_func=lambda s: f"{s} · {names[s]}")
            row = table[table["Student.ID"] == sid]
            if weekly is not None and "Week" in weekly:
                one = weekly.set_index("Student.ID").loc[[sid]].sort_values("Week")
                long = one.melt("Week", ["Read.Actions", "Quiz.Actions", "Submission.Actions"], "Activity", "Count")
                st.altair_chart(dark(alt.Chart(long).mark_line(point=True).encode(
                    x="Week:O", y="Count:Q", color=alt.Color("Activity:N", scale=alt.Scale(
                        range=[ui.VIOLET, ui.EMERALD, ui.AMBER]))).properties(height=220)), width="stretch")
            w = why(mid, row[COLS]).rename("Effect on fail risk").reset_index().rename(columns={"index": "Feature"})
            st.altair_chart(dark(alt.Chart(w).mark_bar(cornerRadius=4).encode(
                x="Effect on fail risk:Q", y=alt.Y("Feature:N", sort=None),
                color=alt.condition("datum['Effect on fail risk'] > 0", alt.value(ui.AMBER), alt.value(ui.VIOLET))
            ).properties(height=200)), width="stretch")
            st.caption("Amber bars push the student toward Fail and violet bars push toward Pass (SHAP values). The weekly chart is for viewing only; the model uses the week 1–8 totals.")

# ---------- 4. docs ----------
else:
    st.markdown("## How it works")
    t1, t2, t3 = st.tabs(["Data & features", "Results", "Deployment & ethics"])
    with t1:
        a, b = st.columns(2)
        with a:
            ui.card("<b style='color:#fff'>8 LMS inputs (counts up to midterm)</b>" + "".join(
                f'<div class="row"><span>{c}</span><span class="muted">{r["min"]}–{r["max"]}</span></div>'
                for c, r in C["inputs"].items()))
        with b:
            ui.card("<b style='color:#fff'>6 features the model adds</b>" + "".join(
                f'<div class="row"><span>{n}</span><span class="formula">{escape(fm)}</span></div>'
                for n, fm in C["features"].items()))
        st.caption("Data: augmented_balanced_dataset_m.csv (MP2), 3,602 rows, half High and half Low. High is shown as Pass and Low as Fail, as confirmed by the instructor. The +1 avoids dividing by zero. Training and this app use the same features() code.")
    with t2:
        cm = card["confusion"]
        a, b = st.columns([1, 1.2])
        with a:
            ui.card(f'<b style="color:#fff">Test results · {escape(card["name"])}</b>'
                    f'<div class="muted" style="margin:6px 0 12px">{card["test_rows"]} students the model never saw in training</div>'
                    f'<div class="cm"><div class="h"></div><div class="h">Predicted Fail</div><div class="h">Predicted Pass</div>'
                    f'<div class="h">Actual Fail</div><div class="ok"><b>{cm["fail_fail"]}</b>caught</div>'
                    f'<div class="bad"><b>{cm["fail_pass"]}</b>missed</div>'
                    f'<div class="h">Actual Pass</div><div class="bad"><b>{cm["pass_fail"]}</b>false alarm</div>'
                    f'<div class="ok"><b>{cm["pass_pass"]}</b>correct</div></div>')
        with b:
            m = MODELS[mid]["bundle"]
            imp = pd.DataFrame({"Feature": m["features"], "Importance": m["model"].feature_importances_})
            st.altair_chart(dark(alt.Chart(imp).mark_bar(cornerRadius=4, color=ui.VIOLET).encode(
                x=alt.X("Importance:Q", axis=alt.Axis(format="%")),
                y=alt.Y("Feature:N", sort="-x", axis=alt.Axis(labelOverlap=False, labelLimit=200)),
                tooltip=[alt.Tooltip("Importance:Q", format=".1%")]).properties(height=430, title="What the model relies on most")),
                width="stretch")
        ui.html('<div class="section"></div>')
        st.markdown("### All models compared")
        st.dataframe(pd.DataFrame([{
            "Model": c["name"], "Algorithm": c["algo"], "Split": c["split"], "Accuracy": f"{c['accuracy']:.1%}",
            "At-risk caught": c["recall_fail"], "Flags correct": c["precision_fail"], "ROC-AUC": c["roc_auc"],
            "Test rows also in training": f"{c['test_rows_with_copy_in_train']} of {c['test_rows']}"}
            for c in (MODELS[x]["card"] for x in MODELS)]), hide_index=True, width="stretch")
    with t3:
        a, b = st.columns(2)
        with a:
            ui.card("<b style='color:#fff'>Deployment</b><div class='muted' style='margin-top:8px'>"
                    "Each model is trained in Python, saved with joblib, and loaded by this Streamlit app on Streamlit Community Cloud. "
                    "The real dataset is not uploaded. The demo classes are made-up students. "
                    "Plan: retrain each semester with new LMS data and compare predictions with final ratings.</div>")
            st.code('joblib.dump({"model": model, "features": FEATURES, "cutoff": 0.5}, "model_a.pkl")\n'
                    'bundle = joblib.load("model_a.pkl")   # app.py\n'
                    'risk = bundle["model"].predict_proba(features(df)[bundle["features"]])', language="python")
        with b:
            ui.card("<b style='color:#fff'>Limits & ethics</b>"
                    "<div class='row'><span class='muted'>The dataset was augmented to balance the classes, so it has 882 exact duplicates "
                    "and many near-copies. With a random split (Model A), copies end up in both training and testing, which "
                    "raises the score. Models B and C keep copies together, so their scores are closer to real use.</span></div>"
                    "<div class='row'><span class='muted'>Only 8 activity counts are used: no grades, timestamps, or student IDs. "
                    "A student can be busy and still struggle, so some flags will be wrong.</span></div>"
                    "<div class='row'><span class='muted'>Data Privacy Act of 2012 (RA 10173): no real student records are stored. "
                    "Uploaded files are processed in memory and cleared on refresh.</span></div>"
                    "<div class='row'><span class='muted'>Human in the loop: a flag means an instructor reaches out to help. "
                    "It is never a grade or a penalty, and the instructor makes the final call.</span></div>"
                    "<div class='row'><span class='muted'>The original target is rating (High/Low). It is shown as Pass/Fail "
                    "because the instructor confirmed High = Pass and Low = Fail.</span></div>")