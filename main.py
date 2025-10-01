import os
import pathlib
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Tuple

import pandas as pd
import requests
import streamlit as st

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

# =========================
# Minimal "Statistic → LLM presence" Streamlit app
# =========================

APP_OUTPUT_DIR = pathlib.Path("app_outputs")
APP_OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_CSV = APP_OUTPUT_DIR / "llm_presence_results.csv"
LATEST_JSON = APP_OUTPUT_DIR / "latest_presence_check.json"

# ---------- Helpers: storage ----------
CSV_COLUMNS = [
    "created_at",
    "user_id",
    "statistic",
    "prompt",
    "llm_presence_verdict",
]

def _ensure_csv(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(path, index=False)

def _append_row(row: dict) -> Tuple[str, int]:
    _ensure_csv(RESULTS_CSV)
    df = pd.DataFrame([{
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "user_id": (row.get("user_id") or "").strip(),
        "statistic": (row.get("statistic") or "").strip(),
        "prompt": row.get("prompt") or "",
        "llm_presence_verdict": (row.get("llm_presence_verdict") or "").strip(),
    }], columns=CSV_COLUMNS)
    df.to_csv(RESULTS_CSV, mode="a", index=False, header=False)
    return str(RESULTS_CSV.resolve()), len(df)

# ---------- Web search (Google CSE) ----------
@st.cache_data(show_spinner=False)
def google_cse(query: str, api_key: str, cse_id: str) -> List[dict]:
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key ,
        "cx": cse_id,
        "q": query + " site:.uk",
        "num": 10,
        "safe": "active",
        "hl": "en",
        "gl": "uk",
    }
    try:
        r = requests.get(endpoint, params=params, timeout=20)
        if r.status_code == 200:
            return r.json().get("items", [])
    except requests.RequestException:
        pass
    return []

def _is_gov_uk(link: str) -> bool:
    try:
        h = urlparse(link).hostname or ""
        return h.endswith(".gov.uk") or h in {"gov.uk", "www.gov.uk"}
    except Exception:
        return False

def fetch_non_govuk_snippets(stat_text: str, api_key: str, cse_id: str) -> List[Tuple[str, str, str]]:
    query = stat_text
s    items = google_cse(query, api_key, cse_id)
    results = []
    for it in items:
        link = it.get("link") or ""
        results.append((it.get("title", ""), link, it.get("snippet", "")))
    return results

def snippets_block(results: List[Tuple[str, str, str]]) -> str:
    if not results:
        return "No non-gov.uk search hits found."
    return "\n".join(f"- {t} ({l}): {s}" for t, l, s in results)

# ---------- LLM ----------
@st.cache_resource(show_spinner=False)
def get_llm() -> ChatVertexAI:
    return ChatVertexAI(model="gemini-2.5-flash-lite", temperature=0)

def build_prompt(stat_text: str, snippets_markdown: str) -> str:
    return f"""
Task: Determine if the statistic appears in official public sources **other than** UK Government (gov.uk).

Statistic:
\"\"\"{stat_text}\"\"\"

Search results:
{snippets_markdown}

Instructions:
- Only answer 'Yes', 'No', or 'Unsure' followed by a short explanation.
- 'Yes' if at least one **official and non gov.uk** source states essentially the same statistic (allow small wording differences, same numeric claim).
- Official source is a reliable source where statistics are produced by governmental or intergovernmental bodies following rigorous standards
- 'No' if results do not contain that statistic or clearly contradict it.
- 'Unsure' if evidence is ambiguous or too weak.
- Consider credibility of the sources when explaining.
""".strip()

# NEW: allow custom prompt with smart fallback/templating
def prepare_prompt(stat_text: str, snippets_markdown: str, custom_prompt: str | None) -> str:  # NEW
    if custom_prompt and custom_prompt.strip():
        cp = custom_prompt.strip()
        # If user used placeholders, substitute; else append context.
        if ("{statistic}" in cp) or ("{snippets}" in cp):
            try:
                return cp.format(statistic=stat_text, snippets=snippets_markdown)
            except Exception:
                # If formatting fails, just append context to be safe
                return f"{cp}\n\nStatistic:\n\"\"\"{stat_text}\"\"\"\n\nSearch results:\n{snippets_markdown}"
        else:
            return f"{cp}\n\nStatistic:\n\"\"\"{stat_text}\"\"\"\n\nSearch results:\n{snippets_markdown}"
    return build_prompt(stat_text, snippets_markdown)

def llm_presence_check(stat_text: str, api_key: str, cse_id: str, custom_prompt: str | None = None) -> Tuple[str, str]:  # UPDATED
    """Returns (prompt, verdict)."""
    non_gov_snips = fetch_non_govuk_snippets(stat_text, api_key, cse_id)
    snips_md = snippets_block(non_gov_snips)
    prompt = prepare_prompt(stat_text, snips_md, custom_prompt)  # UPDATED

    llm = get_llm()
    msg = llm.invoke([HumanMessage(content=prompt)])
    verdict = (msg.content or "").strip()
    return prompt, verdict

# ---------- UI ----------
st.set_page_config(page_title="LLM Presence Check (Non-gov.uk)", layout="centered")
st.title("🔎 Statistic Presence Check ")
st.caption("Enter a statistic. We'll search the web and ask the LLM if the same claim appears in sources other than gov.uk.")

# enter api_key = ""
# enter cse_id = ""

# NEW: collect user_id safely (you had it commented out but used later)
user_id = st.text_input("Your name/email (optional)", key="user_id")  # NEW

statistic = st.text_area("Statistic", placeholder="e.g. 'Over 60% of households recycled their waste in 2024.'", height=120)
custom_prompt = st.text_area(  # NEW
    "Custom LLM prompt (optional)",
    placeholder=(
        "Write your own instructions.\n"
        "Tip: you can use {statistic} and {snippets} placeholders.\n"
        "If you don't include them, the app will append the statistic and search snippets below your prompt."
    ),
    height=180,
)

run = st.button("Run presence check", type="primary")

if run:
    if not statistic.strip():
        st.error("Please enter a statistic.")
        st.stop()
    if not api_key or not cse_id:
        st.error("Please provide your Google CSE API key and CSE ID.")
        st.stop()

    with st.spinner("Checking presence via non-gov.uk sources…"):
        try:
            prompt, verdict = llm_presence_check(statistic.strip(), api_key.strip(), cse_id.strip(), custom_prompt)  # UPDATED
        except Exception as e:
            st.exception(e)
            st.stop()

    # ---- Display ONLY prompt + verdict ----
    st.subheader("Prompt sent to LLM")
    st.code(prompt)

    st.subheader("LLM presence verdict")
    st.write(verdict)

    # ---- Persist results ----
    row = {
        "user_id": user_id,  # now always defined
        "statistic": statistic.strip(),
        "prompt": prompt,
        "llm_presence_verdict": verdict,
    }

    csv_path, _ = _append_row(row)

    try:
        import json
        with open(LATEST_JSON, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)
        st.success(f"Saved latest result to {LATEST_JSON}")
    except Exception as e:
        st.warning(f"Could not write JSON file: {e}")

    st.download_button(
        label="Download this result (JSON)",
        data=pd.Series(row).to_json(indent=2).encode("utf-8"),
        file_name="presence_check_result.json",
        mime="application/json",
        use_container_width=True,
    )

    st.info(f"Appended to results CSV: `{csv_path}`")
