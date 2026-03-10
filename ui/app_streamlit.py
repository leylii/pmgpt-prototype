# app_streamlit.py — Evidence-based Estimation (RAG + Intake)
# Run: python -m streamlit run .\ui\app_streamlit.py

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_DEFAULT_DEVICE"] = "cpu"
os.environ["TRANSFORMERS_NO_META_DEVICE"] = "1"

import json
import re
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import sys
import uuid
import io
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# -----------------------------
# Local RAG (robust fallback)
# -----------------------------
try:
    from rag_minimal import RagIndex  # type: ignore
except Exception:
    class RagIndex:
        def __init__(self, *a, **k):
            pass

        def search(self, *a, **k):
            return []

        def nearest_chunks_with_doc_estimates(self, *a, **k):
            return []



# -----------------------------
# Helpers
# -----------------------------
def build_submission_zip() -> bytes:
    participant_id = ss.get("participant_id", "anon")
    participant_slug = _safe_slug(participant_id)
    today_tag = dt.datetime.now().strftime("%Y%m%d")

    experiment_log = {
        "participant_id": participant_id,
        "timestamp": dt.datetime.now().isoformat(),
        "inputs": {"slots": ss.get("slots", {})},
        "artifacts": {
            "wbs_json": ss.get("wbs_json", []),
            "estimation_tasks": ss.get("estimation_tasks", []),
            "estimation_deterministic_tasks": ss.get("estimation_det_tasks", []),
            "poker_tasks": ss.get("poker_tasks", []),
            "gantt_mermaid": ss.get("gantt_mermaid", ""),
            "gantt_csv": ss.get("gantt_csv", ""),
            "gantt_cp": ss.get("gantt_cp", []),
        },
        "raw": {
            "wbs_raw": ss.get("wbs_raw", ""),
            "baseline_estimates_raw": ss.get("baseline_estimates_raw", ""),
            "estimates_raw": ss.get("estimates_raw", ""),
            "poker_raw": ss.get("poker_raw", ""),
            "gantt_raw": ss.get("gantt_raw", ""),
        },
        "llm_prompts": {
            "wbs": ss.get("wbs_prompt", ""),
            "estimation_context_only": ss.get("estimation_prompt_context_only", ""),
            "estimation_evidence": ss.get("estimation_prompt_evidence", ""),
            "planning_poker": ss.get("planning_poker_prompt", ""),
            "gantt": ss.get("gantt_prompt", ""),
        },
        "llm_outputs": {
            "project_assessment": ss.get("llm_project_assessment", {}),
        },
        "survey": ss.get("survey_responses", []),
    }

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{participant_slug}_setup_{today_tag}.json",
            json.dumps(ss.get("slots", {}), ensure_ascii=False, indent=2),
        )
        zf.writestr(
            f"{participant_slug}_wbs_{today_tag}.json",
            json.dumps(ss.get("wbs_json", []), ensure_ascii=False, indent=2),
        )
        zf.writestr(
            f"{participant_slug}_estimates_{today_tag}.json",
            json.dumps(ss.get("estimation_tasks", []), ensure_ascii=False, indent=2),
        )
        zf.writestr(
            f"{participant_slug}_estimates_deterministic_{today_tag}.json",
            json.dumps(ss.get("estimation_det_tasks", []), ensure_ascii=False, indent=2),
        )
        zf.writestr(
            f"{participant_slug}_poker_{today_tag}.json",
            json.dumps(ss.get("poker_tasks", []), ensure_ascii=False, indent=2),
        )
        zf.writestr(
            f"{participant_slug}_estimates_{today_tag}.csv",
            ss.get("estimation_csv", "") or "",
        )
        zf.writestr(
            f"{participant_slug}_gantt_{today_tag}.mmd",
            ss.get("gantt_mermaid", "") or "",
        )
        zf.writestr(
            f"{participant_slug}_gantt_{today_tag}.csv",
            ss.get("gantt_csv", "") or "",
        )
        zf.writestr(
            f"{participant_slug}_log_{today_tag}.json",
            json.dumps(experiment_log, ensure_ascii=False, indent=2),
        )
        zf.writestr(
            f"{participant_slug}_capacity_analysis_{today_tag}.json",
            json.dumps(
                {
                    "baseline": ss.get("capacity_summary_baseline", {}),
                    "evidence": ss.get("capacity_summary_evidence", {}),
                    "planning_poker": ss.get("capacity_summary_poker", {}),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    mem.seek(0)
    return mem.getvalue()


def generate_participant_id(prefix: str = "PID") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12].upper()}"

def _safe_slug(s: str) -> str:
    s = (s or "project").strip()
    s = re.sub(r"[^A-Za-z0-9\-_]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_").lower()

def _csv_from_estimates(tasks: list) -> str:
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id","name","optimistic_h","most_likely_h","pessimistic_h","duration_days","effort_hours","deps"])
    for t in tasks:
        w.writerow([
            t.get("id",""),
            t.get("name",""),
            t.get("optimistic_h",0),
            t.get("most_likely_h",0),
            t.get("pessimistic_h",0),
            t.get("duration_days",0),
            t.get("effort_hours",0),
            ",".join(t.get("deps",[]) or []),
        ])
    return buf.getvalue()

def validate_dag(items: list) -> Dict[str, Any]:
    from collections import defaultdict, deque
    graph = defaultdict(list)
    indeg = defaultdict(int)
    nodes = set()

    for it in items:
        nid = (it.get("id") or "").strip()
        if not nid:
            continue
        nodes.add(nid)
        for dep in it.get("deps", []):
            if dep:
                graph[dep].append(nid)
                indeg[nid] += 1
                nodes.add(dep)

    q = deque([n for n in nodes if indeg[n] == 0])
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for v in graph[n]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    cycles = []
    if len(order) < len(nodes):
        remain = [n for n in nodes if indeg[n] > 0]
        cycles.append(remain)

    return {"ok": len(cycles) == 0, "cycles": cycles, "order": order}


def _parse_llm_json_object(text: str) -> Optional[dict]:
    if not text:
        return None

    # 1) fenced json
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.I)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # 2) any fenced block containing an object
    m = re.search(r"```(?:\w+)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # 3) raw JSON object
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # 4) last-resort regex
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    return None


# -----------------------------
# LLM provider
# -----------------------------
def run_llm(prompt: str, provider: str = "dry-run", model: str = "gpt-4o-mini") -> str:
    if provider == "dry-run":
        return f"[DRY-RUN] Prompt preview:\n\n{prompt}"

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return "[ERROR] OPENAI_API_KEY is not set. Switch to Dry-run or set the key."

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are PMGPT, a rigorous tutor for student software projects."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            return resp.choices[0].message.content
        except Exception as ex:
            return f"[ERROR calling OpenAI]: {ex}"

    return "[ERROR] Unknown provider."



# -----------------------------
# Prompt templates
# -----------------------------
def prompt_wbs(docs: List[Dict[str, Any]], slots: Dict[str, Any]) -> str:
    doc_text = "\n\n---\n".join([f"Source: {d.get('source','?')}\n{d.get('text','')}" for d in docs])
    return f"""
System:
You are PMGPT, a tutor for student software projects. Teach briefly and produce a correct WBS.

Context (RAG):
{doc_text}

User Slots (JSON):
{json.dumps(slots, ensure_ascii=False, indent=2)}

Task:
Create a Work Breakdown Structure in {slots.get('style')} style for a {slots.get('lifecycle')} student project.
Depth: 3 levels minimum.
Allow 4 levels for implementation tasks when needed.
Enforce:
- 100% rule per parent ({slots.get('constraints', {}).get('wbs_100_percent_rule', True)})
- Mutually exclusive branches ({slots.get('constraints', {}).get('mutually_exclusive', True)})
- Leaves must be actionable and ≤ {slots.get('constraints', {}).get('leaf_max_days', 5)} days of effort
- Prefer nouns for deliverables (artifact-driven), verbs allowed for activity-driven

Also adapt to lifecycle:
- Agile: Backlog → Sprints → Increments; include Planning/Review/Retro
- V-model: Verification/Validation pairs
- Waterfall: sequential phases with gated milestones

Output:
1) JSON array WBS items:
   {{ "id":"1.1", "parent_id":"1", "name":"...", "type":"deliverable|activity", "notes":"" }}

2) Human-readable outline (numbered)

3) 5 likely dependencies (short list). Rules for dependencies:
   - Only create dependencies between LEAF items
   - Never depend on parent/phase items
   - Refer to tasks by exact names from JSON
   - Prefer deliverable-to-deliverable flow

4) 5 sanity-check comments
""".strip()

def prompt_gantt(docs: List[Dict[str, Any]], slots: Dict[str, Any], estimates_json: str) -> str:
    doc_text = "\n\n---\n".join([f"Source: {d.get('source','?')}\n{d.get('text','')}" for d in docs])
    return f"""
System:
You are PMGPT. Build a schedule and Gantt that respects dependencies, capacity and calendar.

Context (RAG):
{doc_text}

Inputs:
- Estimated tasks (JSON):
{estimates_json}
- Slots:
{json.dumps(slots, ensure_ascii=False, indent=2)}

Task:
Build a feasible schedule:
- Respect all dependencies between tasks
- Pack work by capacity ({slots.get('hours_per_week')}×{slots.get('team_size')} hours per week)
- Skip work during break weeks {slots.get('break_weeks')}
- Fit all dates within {slots.get('start_date')} .. {slots.get('end_date')}
- Compute the critical path (CPM) and mark which tasks are on the critical path

Return:
1) Mermaid Gantt code. STRICT RULES:
   - Use: `gantt` diagram with `dateFormat  YYYY-MM-DD`.
   - For every task object in the JSON input, create **exactly one** line.
   - Use the exact `name` field from JSON as the task label.
   - Use **only this format** for all tasks (no other variants):

       `Task Name :task_id, YYYY-MM-DD, duration_days d`

     where:
       - `task_id` is a stable identifier derived from the JSON `id`
         (e.g. `id_1_2_3` or the raw `id` string).
       - `YYYY-MM-DD` is the **concrete start date** you have scheduled
         for that task.
       - `duration_days` is the numeric `duration_days` value from the JSON.

   - You MUST compute a concrete start date for every task:
       * If the task has **no dependencies** (its `deps` list is empty),
         it may start on or after the project start date.
       * If the task **has dependencies**, its start date must be **on or after**
         the latest finish date of all its dependencies.
         (Do not start a task before any of its dependencies finish.)
   - Respect capacity and calendar:
       * Per week, the total scheduled hours must not exceed
         `team_size × hours_per_week`.
       * Do not schedule work inside the listed break weeks.
       * All dates must stay within the project start and end dates.
   - Do **not** use Mermaid `after` syntax at all (no `after id_x`).
   - Do **not** invent or change dependencies.
   - Do **not** drop any tasks.

2) CSV with headers: `task,start,end,deps,owner,estimate`
   - `task`: task name (same as in Mermaid).
   - `start`: scheduled start date (YYYY-MM-DD).
   - `end`: scheduled finish date (YYYY-MM-DD), consistent with `duration_days`.
   - `deps`: comma-separated list of dependency IDs (from the JSON `deps` list).
   - `owner`: set to "Team".
   - `estimate`: the PERT expected effort in hours (`effort_hours`) for that task.

3) Critical path list (ordered)
   - A plain ordered list of task names that lie on the critical path,
     from first to last.

4) 5 short validation notes
   - Briefly explain how you respected dependencies, dates, capacity,
     break weeks and critical path.

Output format:
- First the Mermaid Gantt code in a ```mermaid fenced block.
- Then the CSV in a ```csv fenced block.
- Then the critical path and validation notes as markdown text.
""".strip()
# -----------------------------
# Defaults & Session State
# -----------------------------
DEFAULT_SLOTS = {
    "project_title": "Student Web Application for Course Management",
    "brief_description": "A student team builds a web-based system for course registration and assignment feedback.",
    "style": "artifact-driven",
    "lifecycle": "Waterfall",
    "team_size": 4,
    "hours_per_week": 12,
    "start_date": "2026-03-03",
    "end_date": "2026-05-11",
    "break_weeks": [5],
    "deliverables": ["Requirements doc","Design doc","Prototype","Final product","User manual"],

    "estimation_params": {"sp_to_hours": 2},
    "buffers": {"learning_percent": 20, "integration_percent": 15},

    "req_pages": 10,
    "req_items": 15,
    "screens": 6,
    "apis": 4,
    "integrations": 2,
    "novelty_level": 1,
    "risk_level": 1,
    "team_experience": 1,

    "use_rag": True,
    "k_docs_wbs": 5,
    "min_similarity": 0.25,

    "constraints": {"leaf_max_days": 5, "wbs_100_percent_rule": True, "mutually_exclusive": True},
}

ss = st.session_state
ss.setdefault("rag", None)
ss.setdefault("slots", DEFAULT_SLOTS.copy())

ss.setdefault("participant_id", generate_participant_id())

ss.setdefault("wbs_raw", "")
ss.setdefault("wbs_json", [])
ss.setdefault("wbs_outline", "")

ss.setdefault("estimates_raw", "")
ss.setdefault("estimation_tasks", [])
ss.setdefault("estimation_det_tasks", [])  # deterministic baseline per task
ss.setdefault("estimation_csv", "")

ss.setdefault("poker_raw", "")
ss.setdefault("last_poker_signature", None)

ss.setdefault("gantt_raw", "")
ss.setdefault("gantt_mermaid", "")
ss.setdefault("gantt_csv", "")
ss.setdefault("gantt_cp", [])


ss.setdefault("estimation_prompt_context_only", "")
ss.setdefault("estimation_prompt_evidence", "")
ss.setdefault("planning_poker_prompt", "")

ss.setdefault("rag_project_summary", {})

ss.setdefault("capacity_summary_baseline", {})
ss.setdefault("capacity_summary_evidence", {})
ss.setdefault("capacity_summary_poker", {})

ss.setdefault("llm_project_assessment", {})

ss.setdefault("estimation_tasks_original", [])
ss.setdefault("estimation_det_tasks_original", [])
ss.setdefault("poker_tasks_original", [])

ss.setdefault("gantt_raw_original", "")
ss.setdefault("gantt_raw_editable", "")
ss.setdefault("gantt_raw_editor_pending", None)

ss.setdefault("wbs_prompt", "")
ss.setdefault("gantt_prompt", "")
ss.setdefault("baseline_estimates_raw", "")


# -----------------------------
# Wizard State Machine
# -----------------------------
WIZARD_STEPS = ["setup", "wbs_draft", "wbs_review", "estimation", "planning_poker", "gantt", "survey"]
ss.setdefault("wizard_step", "setup")
ss.setdefault("wbs_approved", False)
ss.setdefault("estimation_approved", False)
ss.setdefault("poker_approved", False)
ss.setdefault("poker_tasks", [])
ss.setdefault("gantt_approved", False)



def go(step: str):
    ss.wizard_step = step
    st.rerun()

def can_enter(step: str) -> bool:
    if step in ("setup", "wbs_draft", "wbs_review"):
        return True
    if step == "estimation":
        return ss.wbs_approved
    if step == "planning_poker":
        return ss.wbs_approved and ss.estimation_approved
    if step == "gantt":
        return ss.wbs_approved and ss.estimation_approved and ss.poker_approved
    if step == "survey":
        return ss.wbs_approved and ss.estimation_approved and ss.gantt_approved and ss.poker_approved
    return True


# -----------------------------
# Stepper UI
# -----------------------------
def render_stepper():
    labels = {
        "setup": "Setup",
        "wbs_draft": "WBS Draft",
        "wbs_review": "WBS Review",
        "estimation": "PERT Estimation",
        "planning_poker": "Planning Poker",
        "gantt": "Gantt",
        "survey": "Survey",
    }

    parts = []
    for step in WIZARD_STEPS:
        label = labels.get(step, step)
        if step == ss.wizard_step:
            parts.append(f"**{label}**")
        else:
            parts.append(label)

    st.markdown(" → ".join(parts))


# -----------------------------
# Sidebar (FIXED: no 'with ... else')
# -----------------------------
st.sidebar.title("PMGPT – Prototype")
provider = "openai"
model = "gpt-4o-mini"

st.sidebar.text_input("LLM Provider", value=provider, disabled=True)
st.sidebar.text_input("Model (OpenAI)", value=model, disabled=True)

use_rag = st.sidebar.checkbox(
    "Use RAG",
    value=True,
    disabled=True
)
ss.slots["use_rag"] = True



# defaults first, then allow override inside expander


ss.setdefault("index_dir", "./pmgpt_index")
with st.sidebar.expander("Advanced RAG settings", expanded=False):

    # ✅ Top-K stored in slots
    ss.slots["k_docs_wbs"] = st.number_input(
        "Top-K docs (RAG)",
        min_value=1,
        max_value=10,
        value=int(ss.slots.get("k_docs_wbs", DEFAULT_SLOTS["k_docs_wbs"])),
        step=1,
        help="Number of most similar chunks/documents to retrieve from the RAG index."
    )

    ss.slots["min_similarity"] = st.slider(
        "Min similarity (RAG)",
        min_value=0.05,
        max_value=1.0,
        value=float(ss.slots.get("min_similarity", DEFAULT_SLOTS["min_similarity"])),
        step=0.01,
        help=(
            "Similarity threshold for accepting evidence from RAG. "
            "Higher values = only very similar past tasks are used as evidence."
        ),
    )

st.sidebar.markdown("---")

st.sidebar.markdown("### Quick Guide")

st.sidebar.markdown("""
**Workflow**

1. Setup project  
2. Generate WBS  
3. Review WBS  
4. Estimate tasks  
5. Planning Poker  
6. Generate Gantt  
7. Submit survey
""")


# -----------------------------
# Lazy-load RAG once
# -----------------------------
if ss.rag is None or ss.get("rag_index_dir", None) != ss.index_dir:
    try:
        ss.rag = RagIndex(ss.index_dir)
        ss.rag_index_dir = ss.index_dir
    except Exception as ex:
        st.sidebar.warning(f"RAG index load failed: {ex}")
        class _FallbackRagIndex:
            def search(self, *_args, **_kwargs):
                return []

            def nearest_chunks_with_doc_estimates(self, *_args, **_kwargs):
                return []
        ss.rag = _FallbackRagIndex()

# -----------------------------
# Render header + stepper
# -----------------------------
st.title("PMGPT – Prototype")

render_stepper()

with st.expander("How to use this tool", expanded=False):

    st.markdown("""
### Instructions

1. **Project Setup**  
   Enter a **project title** and a short **project description**.  
   You can also adjust the project settings such as lifecycle, team size, dates, and estimation parameters.

2. **WBS Draft**  
   Click **Generate WBS** to create an initial Work Breakdown Structure based on your project description.

3. **WBS Review**  
   Review the generated WBS, inspect the outline and dependencies, and edit the WBS JSON if needed before continuing.

4. **PERT Estimation**  
   Review the generated effort estimates.  
   You may edit the estimation JSON manually before moving to the next step.

5. **Planning Poker**  
   Review the planning poker estimates and edit them if needed.

6. **Gantt**  
   Generate the schedule and review the Gantt output.  
   You may edit the generated Gantt text before continuing.

7. **Survey and Submission**  
   Complete the feedback survey at the end.  
   Then download the final submission file and send it by email.
""")
st.divider()

# -----------------------------
# Step 0: Setup
# -----------------------------
def render_setup():
    st.header("Project Setup")
    st.caption("Fill the slots. Then continue to WBS Draft.")

    s = ss.slots

    st.text_input(
        "Participant ID",
        value=ss.participant_id,
        disabled=True
    )

    s["project_title"] = st.text_input("Project title", s["project_title"])
    s["brief_description"] = st.text_area("Brief description (2–3 sentences)", s.get("brief_description",""), height=80)

    c1, c2 = st.columns(2)
    with c1:
        s["style"] = st.selectbox("WBS style", ["artifact-driven", "activity-driven"],
                                  index=0 if s.get("style")=="artifact-driven" else 1)
        s["start_date"] = st.text_input("Start date (YYYY-MM-DD)", s.get("start_date", ""))
        s["team_size"] = st.number_input("Team size", min_value=1, max_value=12, value=int(s.get("team_size",4)))
        breaks_str = st.text_input("Break weeks (comma-separated)", ",".join(map(str, s.get("break_weeks", []))))
        try:
            s["break_weeks"] = [int(x.strip()) for x in breaks_str.split(",") if x.strip()]
        except Exception:
            st.warning("Break weeks parse failed; keeping previous value.")
    with c2:
        s["lifecycle"] = st.selectbox("Lifecycle", ["Waterfall", "Agile", "V-model"],
                                      index=["Waterfall", "Agile", "V-model"].index(s.get("lifecycle", "Waterfall")))
        s["end_date"] = st.text_input("End date (YYYY-MM-DD)", s.get("end_date",""))
        s["hours_per_week"] = st.number_input("Hours/week per student", min_value=1, max_value=40,
                                              value=int(s.get("hours_per_week", 12)))


    deliv_in = st.text_input("Deliverables (comma-separated)", ", ".join(s.get("deliverables", [])))
    s["deliverables"] = [x.strip() for x in deliv_in.split(",") if x.strip()]

    # -----------------------------
    # Estimation settings
    # -----------------------------
    st.subheader("Estimation settings")

    col_left, col_right = st.columns(2)

    with col_left:
        s.setdefault("estimation_params", {})

        sp_h = st.number_input(
            "SP → hours (1 SP ≈ ? hours)",
            min_value=1,
            max_value=10,
            value=int(s["estimation_params"].get("sp_to_hours", 2)),
            step=1,
            help=(
                "Used only in the Planning Poker step.\n"
                "This defines how many hours 1 Story Point approximately represents.\n"
                "Example: If 1 SP ≈ 2 hours, then 8 SP ≈ 16 hours."
            ),
        )

        s["estimation_params"]["sp_to_hours"] = int(sp_h)

    # Right column intentionally left empty for future parameters
    with col_right:
        st.caption(" ")




    # -----------------------------
    # Constraints & buffers
    # -----------------------------
    st.subheader("Constraints & buffers")

    col5, col6, col7 = st.columns(3)

    # Maximum allowed effort per leaf task (in days)
    with col5:
        s.setdefault("constraints", {})
        s["constraints"]["leaf_max_days"] = int(
            st.number_input(
                "Leaf max days",
                min_value=1,
                max_value=15,
                value=int(s.get("constraints", {}).get("leaf_max_days", 5)),
                step=1,
                help=(
                    "Maximum allowed duration (in days) for a leaf task in the WBS. "
                ),
            )
        )

    # Learning buffer percentage
    with col6:
        s.setdefault("buffers", {})
        s["buffers"]["learning_percent"] = int(
            st.number_input(
                "Learning buffer %",
                min_value=0,
                max_value=100,
                value=int(s.get("buffers", {}).get("learning_percent", 20)),
                step=1,
                help=(
                    "Extra percentage added on top of the base estimates to account for "
                    "learning and exploration (reading docs, trying tools, experiments), "
                    "which is especially relevant for student teams."
                ),
            )
        )

    # Integration and testing buffer percentage
    with col7:
        s.setdefault("buffers", {})
        s["buffers"]["integration_percent"] = int(
            st.number_input(
                "Integration buffer %",
                min_value=0,
                max_value=100,
                value=int(s.get("buffers", {}).get("integration_percent", 15)),
                step=1,
                help=(
                    "Extra percentage added to cover integration and system-testing overhead "
                    "(wiring components together, dealing with external APIs, fixing "
                    "integration bugs)."
                ),
            )
        )

    st.subheader("Quantitative scope")

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        s["req_pages"] = st.number_input(
            "Req pages",
            min_value=0,
            max_value=500,
            value=int(s.get("req_pages", 10)),
            help=(
                "Approximate number of pages in the requirements document. "
                "More pages usually mean more analysis and clarification effort."
            ),
        )

    with q2:
        s["req_items"] = st.number_input(
            "Req items",
            min_value=0,
            max_value=500,
            value=int(s.get("req_items", 15)),
            help=(
                "Number of individual requirement statements (functional/non-functional). "
                "Each item adds to functional scope and estimation complexity."
            ),
        )

    with q3:
        s["screens"] = st.number_input(
            "UI screens",
            min_value=0,
            max_value=200,
            value=int(s.get("screens", 6)),
            help=(
                "How many distinct UI screens / views the system will have. "
                "More screens → more design, implementation and testing work."
            ),
        )

    with q4:
        s["apis"] = st.number_input(
            "APIs/endpoints",
            min_value=0,
            max_value=200,
            value=int(s.get("apis", 4)),
            help=(
                "Number of backend API endpoints or external services you expose/consume. "
                "More endpoints increase backend and integration complexity."
            ),
        )

    q5, q6, q7, q8 = st.columns(4)

    with q5:
        s["integrations"] = st.number_input(
            "Integrations",
            min_value=0,
            max_value=50,
            value=int(s.get("integrations", 2)),
            help=(
                "How many external systems you must integrate with "
                "(payment gateways, LMS, auth provider, legacy system, etc.). "
                "Integrations are a major source of uncertainty and extra effort."
            ),
        )

    with q6:
        s["novelty_level"] = st.selectbox(
            "Novelty (0–2)",
            [0, 1, 2],
            index=int(s.get("novelty_level", 1)),
            help=(
                "How new the technology/domain is for the team:\n"
                "0 = familiar, 1 = somewhat new, 2 = completely new.\n"
                "Higher novelty generally increases learning and exploration time."
            ),
        )

    with q7:
        s["risk_level"] = st.selectbox(
            "Risk (0–2)",
            [0, 1, 2],
            index=int(s.get("risk_level", 1)),
            help=(
                "Overall project risk level:\n"
                "0 = low, 1 = medium, 2 = high.\n"
                "Higher risk widens the uncertainty range in estimates."
            ),
        )

    with q8:
        s["team_experience"] = st.selectbox(
            "Team experience (0–2)",
            [0, 1, 2],
            index=int(s.get("team_experience", 1)),
            help=(
                "Experience of this student team with similar projects/tech stack:\n"
                "0 = little/no experience, 1 = some experience, 2 = strong experience.\n"
                "More experience usually reduces effort and rework."
            ),
        )

    # -----------------------------
    # Import / Export (slots.json)
    # -----------------------------
    st.subheader("Export / Import Project Setup")

    # One-time flag to avoid re-processing the same uploaded file
    ss.setdefault("slots_import_loaded", False)

    uploaded = st.file_uploader(
        "Import Project Setup (JSON)",
        type=["json"],
        accept_multiple_files=False
    )

    # If no file is selected, allow the next uploaded file to be processed again
    if uploaded is None:
        ss.slots_import_loaded = False

    if uploaded is not None and not ss.slots_import_loaded:
        try:
            loaded = json.loads(uploaded.getvalue().decode("utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("slots.json must be a JSON object (dictionary).")

            # Overwrite current slots with imported configuration
            ss.slots = loaded

            # Ensure nested dicts exist (for backward compatibility)
            ss.slots.setdefault("constraints", {})
            ss.slots.setdefault("buffers", {})
            ss.slots.setdefault("estimation_params", {})

            # Mark this uploaded file as processed (prevents infinite reruns)
            ss.slots_import_loaded = True

            st.success("slots.json loaded. Refreshing form…")
            st.rerun()

        except Exception as ex:
            st.error(f"Failed to load slots.json: {ex}")

    with st.expander("Preview Project Setup (JSON)", expanded=False):
        st.json(ss.slots)

    slots_payload = json.dumps(ss.slots, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        label="⬇️ Export Project Setup (JSON)",
        data=slots_payload,
        file_name=f"{_safe_slug(ss.get('participant_id', 'anon'))}_setup.json",
        mime="application/json",
        width="stretch"
    )

    st.markdown("---")

    if st.button("➡️ Continue to WBS Draft", width="stretch"):
        ss.wizard_step = "wbs_draft"
        st.rerun()


# -----------------------------
# Step 1: WBS Draft (generate)
# -----------------------------
def render_wbs_draft():
    st.header("WBS Draft")
    st.caption("Generate a draft WBS using your slots and optional RAG context.")

    query = f"{ss.slots['style']} WBS for {ss.slots.get('brief_description','')} using {ss.slots['lifecycle']}"
    k_docs_wbs = int(ss.slots.get("k_docs_wbs", 5))
    docs = ss.rag.search(query, k=k_docs_wbs) if (use_rag and hasattr(ss.rag, "search")) else []

    min_sim = float(ss.slots.get("min_similarity", DEFAULT_SLOTS["min_similarity"]))
    if docs and isinstance(docs, list) and "score" in (docs[0] or {}):
        docs = [d for d in docs if d.get("score", 0.0) >= min_sim]

    with st.expander("RAG snippets (top-k)", expanded=False):
        if docs:
            for d in docs:
                st.markdown(f"**{d.get('source','?')}** (score={d.get('score',0):.3f})")
                text = d.get("text", "")
                st.write(text[:600] + ("…" if len(text) > 600 else ""))
        else:
            st.caption("No RAG docs available.")

    wbs_prompt = prompt_wbs(docs, ss.slots)
    ss.wbs_prompt = wbs_prompt
    if st.button("Generate WBS", width="stretch"):
        ss.wbs_raw = run_llm(wbs_prompt, provider=provider, model=model)
        st.success("WBS generated. You can review/edit it when you're ready.")
        # stay on this page; do NOT auto-navigate
        st.rerun()

    if ss.get("wbs_raw"):
        with st.expander("LLM raw output", expanded=False):
            st.text_area("WBS (LLM output)", ss.wbs_raw, height=260)

    st.markdown("---")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("⬅️ Back to Setup", width="stretch"):
            go("setup")
    with c2:
        if st.button("➡️ Go to WBS Review", width="stretch", disabled=not bool(ss.get("wbs_raw"))):
            go("wbs_review")

# -----------------------------
# Step 2: WBS Review/Edit + Approve (GATE)
# -----------------------------
def render_wbs_review():
    st.header("WBS Review / Edit")
    st.caption("Edit WBS JSON, preview it as a table/tree, validate DAG, then continue.")

    raw_wbs = ss.get("wbs_raw", "")
    if not raw_wbs:
        st.warning("No WBS draft yet. Go back and generate one.")
        if st.button("⬅️ Back to WBS Draft", use_container_width=True):
            go("wbs_draft")
        return

    def extract_json_array(text: str):
        if not text:
            return None
        m = re.search(r"```json(.*?)```", text, flags=re.S | re.I)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass
        m = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def extract_section(text: str, header_keywords: List[str]) -> str:
        if not text:
            return ""
        for kw in header_keywords:
            pattern = rf"(?:^|\n)#+.*?{re.escape(kw)}.*?\n(.*?)(?:\n#+|\Z)"
            m = re.search(pattern, text, flags=re.S | re.I)
            if m:
                return m.group(1).strip()
        for kw in header_keywords:
            pattern = rf"(?:^|\n)\s*\d+\)\s*\d*\s*{re.escape(kw)}.*?\n(.*?)(?:\n\s*\d+\)\s|\Z)"
            m = re.search(pattern, text, flags=re.S | re.I)
            if m:
                return m.group(1).strip()
        return ""

    def build_outline_from_json(items):
        lines = []
        for it in items:
            item_id = str(it.get("id", "") or "")
            item_name = str(it.get("name", "") or "")
            lvl = item_id.count(".")
            lines.append(("   " * lvl) + f"- {item_id} {item_name}".strip())
        return "\n".join(lines)

    def parse_dependencies(deps_text: str, items: list):
        id_by_name_lc = {
            str(it.get("name", "")).lower(): str(it.get("id", ""))
            for it in items
            if str(it.get("name", "")).strip() and str(it.get("id", "")).strip()
        }

        def normalize_label(s: str) -> str:
            s = s.strip().lower()
            s = re.sub(r"^(the|a|an)\s+", "", s)
            s = re.sub(
                r"\b(can be (written|prepared|delivered|reviewed)|is (finalized|completed|approved)|can take place|takes place)\b.*$",
                "",
                s,
            )
            return s.strip()

        def find_id(label: str):
            lbl = normalize_label(label)
            if lbl in id_by_name_lc:
                return id_by_name_lc[lbl]
            for it in items:
                nm = str(it.get("name", "")).lower()
                if lbl in nm or nm in lbl:
                    return str(it.get("id", ""))
            lbl_tokens = set(lbl.split())
            best, score = None, 0
            for it in items:
                nm_tokens = set(str(it.get("name", "")).lower().split())
                sc = len(lbl_tokens & nm_tokens)
                if sc > score:
                    best, score = str(it.get("id", "")), sc
            return best

        deps_map = {}
        for raw in deps_text.splitlines():
            line = raw.strip()
            if not line:
                continue

            line = re.sub(r"^\s*(?:[\-\*]|\d+[\.\)])\s*", "", line)
            line = line.replace("**", "").replace("*", "").replace("_", "").replace("`", "")

            m = (
                re.search(r"(.+?)\s+must\s+be\s+.*?\s+before\s+(.+?)(?:\.)?$", line, re.I)
                or re.search(r"(.+?)\s+before\s+(.+?)(?:\.)?$", line, re.I)
                or re.search(r"(.+?)\s+depends\s+on\s+(?:the\s+completion\s+of\s+)?(.+?)(?:\.)?$", line, re.I)
                or re.search(r"\"?(.+?)\"?\s*(?:→|->)\s*\"?(.+?)\"?\s*$", line)
            )

            if not m:
                continue

            before_lbl = m.group(1).strip()
            after_lbl = m.group(2).strip()

            before_id = find_id(before_lbl)
            after_id = find_id(after_lbl)
            if before_id and after_id:
                deps_map.setdefault(after_id, []).append(before_id)

        for it in items:
            it.setdefault("deps", [])
            it["deps"] = deps_map.get(str(it.get("id", "")), it.get("deps", []))
        return items

    def _normalize_wbs_items(items: list) -> list:
        normalized = []
        for it in items:
            if not isinstance(it, dict):
                continue

            deps_val = it.get("deps", [])
            if isinstance(deps_val, str):
                deps = [x.strip() for x in deps_val.split(",") if x.strip()]
            elif isinstance(deps_val, list):
                deps = [str(x).strip() for x in deps_val if str(x).strip()]
            else:
                deps = []

            item_type = str(it.get("type", "") or "").strip()
            if item_type not in {"deliverable", "activity", ""}:
                item_type = ""

            normalized.append({
                "id": str(it.get("id", "") or "").strip(),
                "parent_id": str(it.get("parent_id", "") or "").strip(),
                "name": str(it.get("name", "") or "").strip(),
                "type": item_type,
                "notes": str(it.get("notes", "") or "").strip(),
                "deps": deps,
            })
        return normalized

    def _items_to_df(items: list) -> "pd.DataFrame":
        rows = []
        for it in items:
            rows.append({
                "id": it.get("id", ""),
                "parent_id": it.get("parent_id", ""),
                "name": it.get("name", ""),
                "type": it.get("type", ""),
                "notes": it.get("notes", ""),
                "deps": ",".join(it.get("deps", []) or []),
            })
        return pd.DataFrame(rows)

    def _graphviz_tree(items: list) -> str:
        id_to_name = {it["id"]: it.get("name", "") for it in items if it.get("id")}
        children = {}
        for it in items:
            pid = it.get("parent_id") or ""
            cid = it.get("id") or ""
            if not cid:
                continue
            children.setdefault(pid, []).append(cid)

        all_ids = set(id_to_name.keys())
        roots = [cid for cid in children.get("", [])] + [
            it["id"] for it in items
            if (it.get("id") and (not it.get("parent_id") or it.get("parent_id") not in all_ids))
        ]

        seen = set()
        roots = [x for x in roots if not (x in seen or seen.add(x))]

        def label(i: str) -> str:
            nm = id_to_name.get(i, "")
            nm = nm.replace('"', '\\"')
            return f"{i} {nm}".strip()

        lines = []
        lines.append("digraph WBS {")
        lines.append('  rankdir=TB;')
        lines.append('  node [shape=box];')

        for it_id in id_to_name.keys():
            lines.append(f'  "{it_id}" [label="{label(it_id)}"];')

        for it in items:
            cid = it.get("id") or ""
            pid = it.get("parent_id") or ""
            if cid and pid and pid in all_ids:
                lines.append(f'  "{pid}" -> "{cid}";')

        lines.append("}")
        return "\n".join(lines)

    # initial source from LLM
    llm_items = extract_json_array(raw_wbs)
    if not llm_items:
        st.error("No JSON detected in WBS output.")
        if st.button("⬅️ Back to WBS Draft", width="stretch"):
            go("wbs_draft")
        st.info("Tip: Your LLM output must include a JSON array like: ```json [ { ... } ] ```")
        return

    llm_items = _normalize_wbs_items(llm_items)

    # single source of truth for this page
    if not ss.get("wbs_json"):
        ss.wbs_json = llm_items

    ss.wbs_json = _normalize_wbs_items(ss.get("wbs_json", []))

    st.markdown("### 1️⃣ Edit JSON")

    if "wbs_json_editor" not in ss:
        ss.wbs_json_editor = json.dumps(ss.wbs_json, ensure_ascii=False, indent=2)

    if "wbs_json_editor_pending" not in ss:
        ss.wbs_json_editor_pending = None

    if ss.wbs_json_editor_pending is not None:
        ss.wbs_json_editor = ss.wbs_json_editor_pending
        ss.wbs_json_editor_pending = None

    json_text = st.text_area(
        "WBS JSON",
        height=420,
        key="wbs_json_editor",
    )

    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("Apply to Preview", use_container_width=True):
            try:
                parsed = json.loads(json_text)
                if not isinstance(parsed, list):
                    raise ValueError("WBS JSON must be a JSON array.")

                normalized = _normalize_wbs_items(parsed)
                ss.wbs_json = normalized
                ss.wbs_json_editor_pending = json.dumps(normalized, ensure_ascii=False, indent=2)

                st.rerun()
            except Exception as ex:
                st.error(f"Invalid JSON: {ex}")

    with b2:
        if st.button("Format JSON", use_container_width=True):
            try:
                parsed = json.loads(json_text)
                if not isinstance(parsed, list):
                    raise ValueError("WBS JSON must be a JSON array.")

                normalized = _normalize_wbs_items(parsed)
                ss.wbs_json = normalized
                ss.wbs_json_editor_pending = json.dumps(normalized, ensure_ascii=False, indent=2)

                st.rerun()
            except Exception as ex:
                st.error(f"Cannot format invalid JSON: {ex}")

    with b3:
        if st.button("Reset from LLM draft", use_container_width=True):
            ss.wbs_json = llm_items
            ss.wbs_json_editor_pending = json.dumps(llm_items, ensure_ascii=False, indent=2)
            ss.apply_deps = False
            st.rerun()

    current_items = _normalize_wbs_items(ss.get("wbs_json", []))

    st.markdown("### 2️⃣ Preview")
    tab_table, tab_tree = st.tabs(["📋 Table Preview", "🌳 Tree Preview"])

    with tab_table:
        df_preview = _items_to_df(current_items)
        st.dataframe(df_preview, use_container_width=True, hide_index=True)

    with tab_tree:
        st.caption("Visual preview based on parent_id relationships.")
        dot = _graphviz_tree(current_items)
        st.graphviz_chart(dot, width="stretch")



    st.markdown("### 3️⃣ Outline")
    outline_md = extract_section(raw_wbs, ["Human-readable Outline", "Outline"])
    with st.expander("LLM Outline (original)", expanded=False):
        if outline_md:
            st.code(outline_md.strip(), language="markdown")
        else:
            st.caption("No outline section detected in the LLM output.")

    final_outline = build_outline_from_json(current_items)
    st.markdown("**Current Outline (from JSON)**")
    st.code(final_outline, language="markdown")
    ss.wbs_outline = final_outline

    st.markdown("### 4️⃣ Likely Dependencies")

    st.markdown(
        """
        These dependencies were generated by the LLM.
    
        You can apply them into the current JSON.
        Applying them will simply add the corresponding `deps` fields in the JSON.
    
        You can still edit the JSON manually afterwards.
        """
    )


    deps_txt = extract_section(raw_wbs, ["Likely Dependencies", "Dependencies"])
    st.markdown(deps_txt if deps_txt else "—")

    ss.setdefault("apply_deps", False)
    ss.setdefault("wbs_base_before_deps", None)

    def _on_toggle_apply_deps():
        import copy

        current = _normalize_wbs_items(ss.get("wbs_json", []))

        if ss.apply_deps:
            ss.wbs_base_before_deps = copy.deepcopy(current)
            applied = parse_dependencies(deps_txt or "", copy.deepcopy(current))
            ss.wbs_json = _normalize_wbs_items(applied)
        else:
            if ss.wbs_base_before_deps is not None:
                ss.wbs_json = _normalize_wbs_items(copy.deepcopy(ss.wbs_base_before_deps))

        st.rerun()

    st.checkbox(
        "Apply dependencies into WBS JSON",
        key="apply_deps",
        value=bool(ss.apply_deps),
        disabled=not bool(deps_txt),
        on_change=_on_toggle_apply_deps,
    )

    current_items = _normalize_wbs_items(ss.get("wbs_json", []))

    parent_ids = {it["id"] for it in current_items if any(ch.get("parent_id") == it["id"] for ch in current_items)}
    bad = []
    for it in current_items:
        cleaned = []
        for dep in (it.get("deps", []) or []):
            if dep == it["id"] or dep in parent_ids:
                bad.append((it["id"], dep))
                continue
            cleaned.append(dep)
        it["deps"] = cleaned

    if bad:
        st.warning(
            "Some dependencies were removed because they referred to parent tasks. "
            "Dependencies are allowed only between leaf tasks."
        )
        ss.wbs_json = _normalize_wbs_items(current_items)

    st.markdown("### 5️⃣ DAG Validation")
    dag_result = validate_dag(current_items)
    if dag_result["ok"]:
        st.success(f"✅ Valid DAG ({len(dag_result['order'])} nodes, no cycles).")
    else:
        st.error("❌ Dependency cycle(s) detected!")
        for cyc in dag_result["cycles"]:
            st.warning("Cycle: " + " → ".join(cyc))

    editor_issues = []

    empty_id_rows = [i for i, it in enumerate(current_items) if not (it.get("id") or "").strip()]
    empty_name_rows = [i for i, it in enumerate(current_items) if not (it.get("name") or "").strip()]

    if empty_id_rows:
        editor_issues.append(f"Some rows have empty id: rows {', '.join(map(str, empty_id_rows))}")
    if empty_name_rows:
        editor_issues.append(f"Some rows have empty name: rows {', '.join(map(str, empty_name_rows))}")

    ids = [(it.get("id") or "").strip() for it in current_items if (it.get("id") or "").strip()]
    if len(ids) != len(set(ids)):
        editor_issues.append("Duplicate ids detected. Please ensure all WBS item ids are unique.")

    ss.wbs_editor_ok = (len(editor_issues) == 0)

    if editor_issues:
        st.warning("JSON issues (fix before approving):")
        for msg in editor_issues:
            st.write(f"- {msg}")
    else:
        st.success("JSON looks good (no empty/duplicate id/name).")

    st.markdown("---")
    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("⬅️ Back to WBS Draft", width="stretch"):
            go("wbs_draft")

    with c2:
        can_continue = bool(dag_result["ok"] and ss.get("wbs_editor_ok", False))
        if st.button("Continue ➡️", width="stretch", disabled=not can_continue):
            ss.wbs_json = current_items
            ss.wbs_approved = True
            go("estimation")
# -----------------------------
# Estimation helpers (shared by PERT + Planning Poker)
# -----------------------------
def _get_leaf_items(wbs_items: list) -> list:
    ids_with_children = {
        it.get("parent_id") for it in wbs_items if it.get("parent_id")
    }
    return [it for it in wbs_items if it.get("id") not in ids_with_children]

def _project_context_for_llm(slots: dict) -> dict:
    return {
        "project_title": slots.get("project_title"),
        "brief_description": slots.get("brief_description"),
        "style": slots.get("style"),
        "lifecycle": slots.get("lifecycle"),
        "team_size": slots.get("team_size"),
        "hours_per_week": slots.get("hours_per_week"),
        "start_date": slots.get("start_date"),
        "end_date": slots.get("end_date"),
        "break_weeks": slots.get("break_weeks", []),
        "deliverables": slots.get("deliverables", []),
        "req_pages": slots.get("req_pages"),
        "req_items": slots.get("req_items"),
        "screens": slots.get("screens"),
        "apis": slots.get("apis"),
        "integrations": slots.get("integrations"),
        "novelty_level": slots.get("novelty_level"),
        "risk_level": slots.get("risk_level"),
        "team_experience": slots.get("team_experience"),
        "buffers": slots.get("buffers", {}),
        "constraints": slots.get("constraints", {}),
        "estimation_params": slots.get("estimation_params", {}),
    }

def _compact_wbs(items: list) -> list:
    out = []
    for it in items:
        out.append({
            "id": it.get("id", ""),
            "parent_id": it.get("parent_id", ""),
            "name": it.get("name", ""),
            "type": it.get("type", ""),
            "deps": it.get("deps", []) or [],
        })
    return out

def _collect_rag_evidence_for_tasks(rag: Any, leaf_items: list, k: int, min_sim: float) -> dict:
    evidence = {}

    if rag is None or not hasattr(rag, "nearest_chunks_with_doc_estimates"):
        for it in leaf_items:
            evidence[it.get("id", "")] = {
                "task_name": it.get("name", ""),
                "hits": [],
            }
        return evidence

    for it in leaf_items:
        task_id = it.get("id", "")
        task_name = it.get("name", "") or ""
        query = f"task: {task_name}"

        try:
            hits = rag.nearest_chunks_with_doc_estimates(query, k=k) or []
        except Exception:
            hits = []

        cleaned = []
        seen = set()

        for h in hits:
            try:
                score = float(h.get("score", 0.0))
            except Exception:
                score = 0.0

            if score < min_sim:
                continue

            key = (h.get("doc_id"), h.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)

            est = h.get("estimated_hours", {}) or {}
            req_est = est.get("requirements", {}) if isinstance(est, dict) else {}
            des_est = est.get("design", {}) if isinstance(est, dict) else {}
            impl_est = est.get("implementation", {}) if isinstance(est, dict) else {}

            cleaned.append({
                "source": h.get("source"),
                "score": round(score, 4),
                "section": h.get("section"),
                "snippet": (h.get("text", "") or "")[:500],
                "estimated_hours": est,
                "project_basis": {
                    "project_size": req_est.get("project_size"),
                    "total_pages": req_est.get("total_pages"),
                    "complexity_multiplier": req_est.get("complexity_multiplier"),
                    "estimation_basis": req_est.get("estimation_basis"),
                },
                "hours_summary": {
                    "requirements_h": req_est.get("hours"),
                    "design_h": des_est.get("hours"),
                    "implementation_h": impl_est.get("hours"),
                },
            })

        evidence[task_id] = {
            "task_name": task_name,
            "hits": cleaned,
        }

    return evidence

def _estimate_effective_project_capacity(slots: dict) -> dict:
    import datetime as dt
    start = dt.datetime.strptime(slots["start_date"], "%Y-%m-%d").date()
    end = dt.datetime.strptime(slots["end_date"], "%Y-%m-%d").date()

    total_days = (end - start).days + 1
    total_weeks = total_days / 7.0
    break_weeks = len(slots.get("break_weeks", []))
    effective_weeks = max(total_weeks - break_weeks, 0)

    weekly_capacity = float(slots.get("team_size", 1)) * float(slots.get("hours_per_week", 1))
    total_capacity_hours = effective_weeks * weekly_capacity

    return {
        "total_weeks": round(total_weeks, 2),
        "break_weeks": break_weeks,
        "effective_weeks": round(effective_weeks, 2),
        "weekly_capacity_hours": round(weekly_capacity, 2),
        "total_capacity_hours": round(total_capacity_hours, 2),
    }

def _compute_capacity_summary_from_rows(tasks: list, slots: dict, effort_key: str) -> dict:
    cap = _estimate_effective_project_capacity(slots)

    def _num(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    total_effort = sum(_num(t.get(effort_key, 0.0)) for t in (tasks or []))
    total_capacity_hours = _num(cap.get("total_capacity_hours", 0.0))
    ratio = (total_effort / total_capacity_hours) if total_capacity_hours > 0 else None

    return {
        "total_estimated_effort_hours": round(total_effort, 2),
        "available_capacity_hours": round(total_capacity_hours, 2),
        "capacity_usage_ratio": round(ratio, 4) if ratio is not None else None,
        "capacity_usage_percent": round(ratio * 100, 2) if ratio is not None else None,
        "capacity_details": cap,
    }

def _build_unique_source_signals(rag_evidence: dict) -> dict:
    best_hit_by_source = {}

    for _task_id, payload in (rag_evidence or {}).items():
        for hit in payload.get("hits", []):
            source = str(hit.get("source", "") or "").strip()
            if not source:
                continue

            try:
                score = float(hit.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0

            est = hit.get("estimated_hours", {}) or {}
            req_est = est.get("requirements", {}) if isinstance(est, dict) else {}
            des_est = est.get("design", {}) if isinstance(est, dict) else {}
            impl_est = est.get("implementation", {}) if isinstance(est, dict) else {}

            prev = best_hit_by_source.get(source)
            if prev is None or score > prev["score"]:
                best_hit_by_source[source] = {
                    "source": source,
                    "score": round(score, 4),
                    "section": str(hit.get("section", "") or "").strip(),
                    "estimated_hours": est,
                    "hours_summary": {
                        "requirements_h": req_est.get("hours"),
                        "design_h": des_est.get("hours"),
                        "implementation_h": impl_est.get("hours"),
                    },
                    "project_basis": {
                        "project_size": req_est.get("project_size"),
                        "total_pages": req_est.get("total_pages"),
                        "complexity_multiplier": req_est.get("complexity_multiplier"),
                        "estimation_basis": req_est.get("estimation_basis"),
                    },
                }

    sources = sorted(
        best_hit_by_source.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    avg_similarity = round(
        sum(x["score"] for x in sources) / len(sources), 4
    ) if sources else 0.0

    evidence_strength = (
        "strong" if avg_similarity >= 0.45 else
        "medium" if avg_similarity >= 0.32 else
        "weak"
    )

    return {
        "similar_sources_found": len(sources),
        "avg_similarity": avg_similarity,
        "evidence_strength": evidence_strength,
        "sources": sources,
        "note": "Unique source-level evidence from similar projects; each source appears only once using its best-matching hit."
    }

def _normalize_estimation_items(items: list, hours_per_day: float) -> list:
    normalized = []

    for it in items:
        if not isinstance(it, dict):
            continue

        deps_val = it.get("deps", [])
        if isinstance(deps_val, str):
            deps = [x.strip() for x in deps_val.split(",") if x.strip()]
        elif isinstance(deps_val, list):
            deps = [str(x).strip() for x in deps_val if str(x).strip()]
        else:
            deps = []

        def _num(v):
            try:
                if v is None or v == "":
                    return None
                return round(float(v), 2)
            except Exception:
                return None

        O = _num(it.get("optimistic_h"))
        M = _num(it.get("most_likely_h"))
        P = _num(it.get("pessimistic_h"))

        failed = O is None or M is None or P is None

        if failed:
            duration_days = None
            effort_hours = None
        else:
            effort_hours = _pert_mean_hours(O, M, P)
            duration_days = round(effort_hours / max(hours_per_day, 0.1), 3)

        normalized.append({
            "id": str(it.get("id", "") or "").strip(),
            "name": str(it.get("name", "") or "").strip(),
            "domain": str(it.get("domain", "") or "").strip(),
            "optimistic_h": O,
            "most_likely_h": M,
            "pessimistic_h": P,
            "duration_days": duration_days,
            "effort_hours": effort_hours,
            "deps": deps,
            "baseline_h": _num(it.get("baseline_h")),
            "llm_rationale": str(it.get("llm_rationale", "") or "").strip(),
            "llm_error": str(it.get("llm_error", "") or "").strip(),
            "engine_used": str(it.get("engine_used", "") or "").strip(),
        })

    return normalized

def _normalize_poker_items(items: list, sp_to_hours: float, hours_per_day: float) -> list:
    fib_deck = [1, 2, 3, 5, 8, 13, 21]

    def _nearest_fib(v):
        try:
            if v is None or v == "":
                return 1
            v = float(v)
        except Exception:
            return 1
        return min(fib_deck, key=lambda x: abs(x - v))

    normalized = []

    for it in items:
        if not isinstance(it, dict):
            continue

        sp_min = _nearest_fib(it.get("sp_min"))
        sp_max = _nearest_fib(it.get("sp_max"))

        if sp_max < sp_min:
            sp_max = sp_min

        sp_mid = round((sp_min + sp_max) / 2.0, 2)
        hours_estimate = round(sp_mid * sp_to_hours, 2)
        duration_days = round(
            hours_estimate / max(hours_per_day, 0.1),
            2
        )

        normalized.append({
            "id": str(it.get("id", "") or "").strip(),
            "name": str(it.get("name", "") or "").strip(),
            "domain": str(it.get("domain", "") or "").strip(),
            "sp_min": int(sp_min),
            "sp_max": int(sp_max),
            "sp_mid": sp_mid,
            "hours_estimate": hours_estimate,
            "duration_days": duration_days,
            "reasoning": str(it.get("reasoning", "") or "").strip(),
            "risk_indicators": str(it.get("risk_indicators", "") or "").strip(),
            "baseline_h": None if it.get("baseline_h") in (None, "") else float(it.get("baseline_h")),
            "llm_error": str(it.get("llm_error", "") or "").strip(),
            "engine_used": str(it.get("engine_used", "") or "").strip(),
        })

    return normalized

# -----------------------------
# Estimation Prompt templates
# -----------------------------
def prompt_pert_context_only_batch(slots, wbs_items, leaf_items):
    ctx = _project_context_for_llm(slots)

    return f"""
Estimate PERT triples (hours) for ALL leaf tasks.

Project context:
{json.dumps(ctx, ensure_ascii=False, indent=2)}

Full WBS:
{json.dumps(_compact_wbs(wbs_items), ensure_ascii=False, indent=2)}

Leaf tasks:
{json.dumps(_compact_wbs(leaf_items), ensure_ascii=False, indent=2)}

Task:
Estimate effort for each leaf task using PERT.

Rules:
- Return ONLY a valid JSON array
- Output exactly one row for every leaf task
- Preserve the exact id and exact name of each leaf task
- Estimates must represent effort hours, not calendar dates
- This is effort estimation only, not scheduling
- Do not assign start dates or build a schedule
- Use the project context, task semantics, and WBS structure to inform your estimates

Each item must follow this schema:
{{
  "id": "...",
  "name": "...",
  "O": number,
  "M": number,
  "P": number,
  "rationale": "short explanation"
}}

PERT constraints:
- O < M < P
- Values must be hours
""".strip()

def prompt_pert_evidence_batch(
    slots,
    wbs_items,
    leaf_items,
    rag_project_summary
):
    ctx = _project_context_for_llm(slots)

    return f"""
You are a software project estimation engine.

Estimate PERT triples in HOURS for ALL leaf tasks.

Important:
The RAG evidence below is primarily PROJECT-LEVEL evidence from similar past student software projects.
It is not direct task-level evidence.
Use it to infer the likely overall effort range and domain-level effort distribution for this project.

Inputs:

Project context:
{json.dumps(ctx, ensure_ascii=False, indent=2)}

Full WBS:
{json.dumps(_compact_wbs(wbs_items), ensure_ascii=False, indent=2)}

Leaf tasks:
{json.dumps(_compact_wbs(leaf_items), ensure_ascii=False, indent=2)}

Project-level RAG summary from similar projects:
The summary below may include, for each similar source:
- project_size
- total_pages
- complexity_multiplier
- section-level estimated hours for requirements/design/implementation
- an estimation_basis string explaining how the estimate was derived
{json.dumps(rag_project_summary, ensure_ascii=False, indent=2)}

Task:
Estimate O/M/P hours for every leaf task.

Required reasoning process:
Infer project size using these signals:

Scope:
- number of requirements
- number of screens
- number of APIs
- number of integrations
- number of WBS leaf tasks

Complexity:
- novelty_level
- risk_level
- team_experience

Artifacts:
- documentation deliverables

Analogy evidence:
- RAG project-level effort
- similarity strength

1. Combine these signals to classify the project as small, medium, or large.
2. Use RAG evidence from similar projects as a prior for the likely total/domain-level effort.
3. Reconcile RAG evidence into a realistic overall project effort level.
4. Allocate that overall effort across the leaf tasks according to:
   - task domain
   - task semantics
   - WBS position
   - project scope signals (req_items, screens, apis, integrations)
   - novelty, risk, and team experience
5. Then convert the allocated effort for each task into PERT values:
   - O = optimistic hours
   - M = most likely hours
   - P = pessimistic hours

Instructions:
- For each leaf task, classify its domain as exactly one of:
  "requirements", "design", "implementation"
- Use RAG evidence as project-level similarity evidence, not as direct task-level labels
- Use RAG evidence as the anchor
- Keep the final total effort across tasks broadly consistent with the project-level evidence
- Do not copy document-level RAG hours directly into one task
- Do not assign nearly identical estimates to all tasks in the same domain unless clearly justified
- Make implementation tasks generally larger than requirements/design tasks when appropriate
- Keep estimates realistic for a student team project

Output:
Return ONLY a valid JSON object.
No markdown. No code fences. No extra text.

Schema:
{{
  "project_assessment": {{
    "project_size": "small|medium|large",
    "estimated_total_project_hours": number,
    "domain_split_percent": {{
      "requirements": number,
      "design": number,
      "implementation": number
    }},
    "main_drivers": [
      "short phrase",
      "short phrase",
      "short phrase"
    ]
  }},
  "tasks": [
    {{
      "id": "exact leaf id",
      "name": "exact leaf name",
      "domain": "requirements|design|implementation",
      "O": number,
      "M": number,
      "P": number,
      "rationale": "short explanation mentioning allocation logic"
    }}
  ]
}}

Rules:
- Output exactly one row for every leaf task
- Do not add or remove tasks
- Preserve exact id and exact name
- domain must be one of the 3 allowed values
- O < M < P
- JSON must be parseable
""".strip()

def prompt_poker_batch(
    slots,
    wbs_items,
    leaf_items,
    rag_project_summary
):
    ctx = _project_context_for_llm(slots)

    return f"""
You are an agile estimation engine performing Planning Poker for a student software project.

Estimate Planning Poker story point ranges for ALL leaf tasks.

Important:
- Story points are RELATIVE estimates, not direct hour estimates.
- Use story points to reflect relative effort, complexity, uncertainty, coordination overhead, and technical risk.
- The RAG evidence below is primarily PROJECT-LEVEL evidence from similar past student software projects.
  It is not direct task-level evidence.
- Use the RAG evidence as a prior for overall project scale and likely effort distribution,
  but do not copy document-level estimates directly into a single task.

Inputs:

Project context:
{json.dumps(ctx, ensure_ascii=False, indent=2)}

Full WBS:
{json.dumps(_compact_wbs(wbs_items), ensure_ascii=False, indent=2)}

Leaf tasks:
{json.dumps(_compact_wbs(leaf_items), ensure_ascii=False, indent=2)}

Project-level RAG summary from similar projects:
{json.dumps(rag_project_summary, ensure_ascii=False, indent=2)}

Fibonacci deck:
[1, 2, 3, 5, 8, 13, 21]

Interpret the deck as relative scale:
- 1 = tiny / very low uncertainty
- 2 = very small
- 3 = small but non-trivial
- 5 = moderate
- 8 = substantial
- 13 = large or high uncertainty
- 21 = very large / very uncertain / likely should be split further

Instructions:

For each leaf task:

1. Determine its domain as exactly one of:
   - requirements
   - design
   - implementation

2. Consider:
   - project context
   - WBS structure and task position
   - task semantics
   - project-level RAG evidence
   - novelty, risk, team experience
   - screens, APIs, integrations, requirements count
   - documentation and coordination overhead where relevant

3. Estimate a Planning Poker range:
   - sp_min
   - sp_max

4. Use narrower ranges when the task is familiar and well-bounded.
   Use wider ranges when the task has uncertainty, integration risk, ambiguity, or learning overhead.

Rules:
- sp_min and sp_max must come from the Fibonacci deck
- 1 ≤ sp_min ≤ sp_max ≤ 21
- Preserve relative realism across tasks
- Implementation tasks are often larger than requirements/design tasks, but not always
- Do not assign nearly identical point ranges to all tasks
- Use 21 only when the task is exceptionally large or highly uncertain
- If a task looks too large for a leaf task, still estimate it, but reflect that concern in reasoning

Output:
Return ONLY a valid JSON array.
No markdown. No code fences. No extra text.

Schema:
[
  {{
    "id": "exact leaf id",
    "name": "exact leaf task name",
    "domain": "requirements|design|implementation",
    "sp_min": number,
    "sp_max": number,
    "reasoning": "short explanation focused on relative size and uncertainty",
    "risk_indicators": ["short phrase", "short phrase"]
  }}
]

Constraints:
- Output exactly one row for every leaf task
- Do not add or remove tasks
- Preserve exact id and exact name
- domain must be one of the 3 allowed values
- sp_min and sp_max must be Fibonacci values from the given deck
- JSON must be parseable
""".strip()



def _parse_llm_json_array(text: str) -> Optional[list]:
    if not text:
        return None

    # ```json [ ... ] ```
    m = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", text, flags=re.I)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # any fenced block containing [ ... ]
    m = re.search(r"```(?:\w+)?\s*(\[[\s\S]*?\])\s*```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # raw array
    m = re.search(r"(\[[\s\S]*\])", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    return None




def validate_pert_context_only_batch(
    rows: list,
    leaf_items: list,
) -> Tuple[List[dict], List[str]]:
    """
    Matches prompt_pert_context_only_batch schema exactly.

    Expected item:
    {
      "id": "...",
      "name": "...",
      "O": number,
      "M": number,
      "P": number,
      "rationale": "short"
    }
    """
    leaf_by_id = _leaf_index(leaf_items)
    valid: List[dict] = []
    errors: List[str] = []

    if not isinstance(rows, list):
        return [], ["Output is not a JSON array."]

    seen_ids = set()

    for idx, obj in enumerate(rows):
        if not isinstance(obj, dict):
            errors.append(f"Row {idx}: item is not an object.")
            continue

        task_id = str(obj.get("id", "")).strip()
        if not task_id:
            errors.append(f"Row {idx}: missing id.")
            continue

        if task_id not in leaf_by_id:
            errors.append(f"Row {idx}: unknown leaf id '{task_id}'.")
            continue

        if task_id in seen_ids:
            errors.append(f"Row {idx}: duplicate id '{task_id}'.")
            continue
        seen_ids.add(task_id)

        expected_name = str(leaf_by_id[task_id].get("name", "")).strip()
        name = str(obj.get("name", "")).strip() or expected_name

        O = _coerce_float(obj.get("O"))
        M = _coerce_float(obj.get("M"))
        P = _coerce_float(obj.get("P"))
        rationale = str(obj.get("rationale", "")).strip()

        if O is None or M is None or P is None:
            errors.append(f"Row {idx} ({task_id}): O/M/P must be numeric.")
            continue

        if not (O < M < P):
            errors.append(f"Row {idx} ({task_id}): must satisfy O < M < P.")
            continue

        if O <= 0 or M <= 0 or P <= 0:
            errors.append(f"Row {idx} ({task_id}): O/M/P must be > 0.")
            continue

        valid.append({
            "id": task_id,
            "name": expected_name,  # enforce exact leaf name
            "O": round(O, 2),
            "M": round(M, 2),
            "P": round(P, 2),
            "rationale": rationale[:300],
        })

    missing = [tid for tid in leaf_by_id if tid not in seen_ids]
    for tid in missing:
        errors.append(f"Missing leaf task: '{tid}'.")

    return valid, errors


def validate_pert_evidence_batch(
    rows: list,
    leaf_items: list,
) -> Tuple[List[dict], List[str]]:
    """
    Matches prompt_pert_evidence_batch schema exactly.

    Expected item:
    {
      "id": "exact leaf id",
      "name": "exact leaf name",
      "domain": "requirements|design|implementation",
      "O": number,
      "M": number,
      "P": number,
      "rationale": "short explanation"
    }
    """
    leaf_by_id = _leaf_index(leaf_items)
    allowed_domains = {"requirements", "design", "implementation"}
    valid: List[dict] = []
    errors: List[str] = []

    if not isinstance(rows, list):
        return [], ["Output is not a JSON array."]

    seen_ids = set()

    for idx, obj in enumerate(rows):
        if not isinstance(obj, dict):
            errors.append(f"Row {idx}: item is not an object.")
            continue

        task_id = str(obj.get("id", "")).strip()
        if not task_id:
            errors.append(f"Row {idx}: missing id.")
            continue

        if task_id not in leaf_by_id:
            errors.append(f"Row {idx}: unknown leaf id '{task_id}'.")
            continue

        if task_id in seen_ids:
            errors.append(f"Row {idx}: duplicate id '{task_id}'.")
            continue
        seen_ids.add(task_id)

        expected_name = str(leaf_by_id[task_id].get("name", "")).strip()
        domain = str(obj.get("domain", "")).strip().lower()
        O = _coerce_float(obj.get("O"))
        M = _coerce_float(obj.get("M"))
        P = _coerce_float(obj.get("P"))
        rationale = str(obj.get("rationale", "")).strip()

        if domain not in allowed_domains:
            errors.append(f"Row {idx} ({task_id}): invalid domain '{domain}'.")
            continue

        if O is None or M is None or P is None:
            errors.append(f"Row {idx} ({task_id}): O/M/P must be numeric.")
            continue

        if not (O < M < P):
            errors.append(f"Row {idx} ({task_id}): must satisfy O < M < P.")
            continue

        if O <= 0 or M <= 0 or P <= 0:
            errors.append(f"Row {idx} ({task_id}): O/M/P must be > 0.")
            continue

        valid.append({
            "id": task_id,
            "name": expected_name,  # enforce exact leaf name
            "domain": domain,
            "O": round(O, 2),
            "M": round(M, 2),
            "P": round(P, 2),
            "rationale": rationale[:300],
        })

    missing = [tid for tid in leaf_by_id if tid not in seen_ids]
    for tid in missing:
        errors.append(f"Missing leaf task: '{tid}'.")

    return valid, errors


def validate_poker_batch(
    rows: list,
    leaf_items: list,
) -> Tuple[List[dict], List[str]]:
    """
    Matches prompt_poker_batch schema exactly.

    Expected item:
    {
      "id": "leaf id",
      "name": "leaf task name",
      "domain": "requirements|design|implementation",
      "sp_min": number,
      "sp_max": number,
      "reasoning": "short explanation",
      "risk_indicators": ["short phrase", "short phrase"]
    }
    """
    leaf_by_id = _leaf_index(leaf_items)
    allowed_domains = {"requirements", "design", "implementation"}
    fib = {1, 2, 3, 5, 8, 13, 21}
    valid: List[dict] = []
    errors: List[str] = []

    if not isinstance(rows, list):
        return [], ["Output is not a JSON array."]

    seen_ids = set()

    for idx, obj in enumerate(rows):
        if not isinstance(obj, dict):
            errors.append(f"Row {idx}: item is not an object.")
            continue

        task_id = str(obj.get("id", "")).strip()
        if not task_id:
            errors.append(f"Row {idx}: missing id.")
            continue

        if task_id not in leaf_by_id:
            errors.append(f"Row {idx}: unknown leaf id '{task_id}'.")
            continue

        if task_id in seen_ids:
            errors.append(f"Row {idx}: duplicate id '{task_id}'.")
            continue
        seen_ids.add(task_id)

        expected_name = str(leaf_by_id[task_id].get("name", "")).strip()
        domain = str(obj.get("domain", "")).strip().lower()
        sp_min = _coerce_int(obj.get("sp_min"))
        sp_max = _coerce_int(obj.get("sp_max"))
        reasoning = str(obj.get("reasoning", "")).strip()
        risk_indicators = obj.get("risk_indicators", [])

        if domain not in allowed_domains:
            errors.append(f"Row {idx} ({task_id}): invalid domain '{domain}'.")
            continue

        if sp_min is None or sp_max is None:
            errors.append(f"Row {idx} ({task_id}): sp_min/sp_max must be integers.")
            continue

        if sp_min not in fib or sp_max not in fib:
            errors.append(f"Row {idx} ({task_id}): sp_min/sp_max must be Fibonacci values.")
            continue

        if not (1 <= sp_min <= sp_max <= 21):
            errors.append(f"Row {idx} ({task_id}): must satisfy 1 <= sp_min <= sp_max <= 21.")
            continue

        if not isinstance(risk_indicators, list):
            errors.append(f"Row {idx} ({task_id}): risk_indicators must be a list.")
            continue

        valid.append({
            "id": task_id,
            "name": expected_name,  # enforce exact leaf name
            "domain": domain,
            "sp_min": sp_min,
            "sp_max": sp_max,
            "reasoning": reasoning[:300],
            "risk_indicators": [str(x).strip()[:120] for x in risk_indicators if str(x).strip()],
        })

    missing = [tid for tid in leaf_by_id if tid not in seen_ids]
    for tid in missing:
        errors.append(f"Missing leaf task: '{tid}'.")

    return valid, errors




def run_context_only_batch_estimation(slots, wbs_items, provider, model):
    leaf_items = _get_leaf_items(wbs_items)

    prompt = prompt_pert_context_only_batch(slots, wbs_items, leaf_items)
    ss.estimation_prompt_context_only = prompt
    raw = run_llm(prompt, provider=provider, model=model)

    arr = _parse_llm_json_array(raw) or []
    valid_rows, errors = validate_pert_context_only_batch(arr, leaf_items)

    valid_by_id = {row["id"]: row for row in valid_rows}

    rows = []
    for it in leaf_items:
        tid = str(it.get("id", "")).strip()
        name = str(it.get("name", "")).strip()

        obj = valid_by_id.get(tid)
        if not obj:
            rows.append({
                "id": tid,
                "name": name,
                "domain": _domain_of(name),   # fallback only
                "O": None,
                "M": None,
                "P": None,
                "baseline_h": None,
                "llm_rationale": "",
                "llm_error": "; ".join(errors)[:300],
                "engine_used": "llm_context_only_failed",
            })
            continue

        rows.append({
            "id": tid,
            "name": name,
            "domain": _domain_of(name),   # prompt does not return domain here
            "O": obj["O"],
            "M": obj["M"],
            "P": obj["P"],
            "baseline_h": _pert_mean_hours(obj["O"], obj["M"], obj["P"]),
            "llm_rationale": obj.get("rationale", ""),
            "llm_error": "",
            "engine_used": "llm_context_only",
        })

    return {
        "rows": rows,
        "raw": raw,
        "errors": errors,
    }

def run_evidence_batch_estimation(
    slots,
    wbs_items,
    baseline_rows,
    rag_evidence,
    provider,
    model,
):
    leaf_items = _get_leaf_items(wbs_items)

    rag_project_summary = _build_unique_source_signals(rag_evidence)

    prompt = prompt_pert_evidence_batch(
        slots,
        wbs_items,
        leaf_items,
        rag_project_summary,
    )
    ss.estimation_prompt_evidence = prompt
    raw = run_llm(prompt, provider=provider, model=model)

    parsed = _parse_llm_json_object(raw) or {}
    arr = parsed.get("tasks", []) if isinstance(parsed, dict) else []
    project_assessment = parsed.get("project_assessment", {}) if isinstance(parsed, dict) else {}

    valid_rows, errors = validate_pert_evidence_batch(arr, leaf_items)
    ss.llm_project_assessment = project_assessment
    valid_by_id = {row["id"]: row for row in valid_rows}

    baseline_by_id = {
        str(r.get("id", "")).strip(): r
        for r in (baseline_rows or [])
    }

    rows = []
    hours_per_day = _hours_per_day_from_slots(slots)

    for it in leaf_items:
        tid = str(it.get("id", "")).strip()
        name = str(it.get("name", "")).strip()
        deps = it.get("deps", []) or []

        obj = valid_by_id.get(tid)
        base = baseline_by_id.get(tid, {})

        if not obj:
            rows.append({
                "id": tid,
                "name": name,
                "domain": base.get("domain"),
                "optimistic_h": None,
                "most_likely_h": None,
                "pessimistic_h": None,
                "duration_days": None,
                "effort_hours": None,
                "deps": deps,
                "baseline_h": base.get("baseline_h"),
                "llm_rationale": "",
                "llm_error": "; ".join(errors)[:300],
                "engine_used": "llm_evidence_failed",
            })
            continue

        O = obj["O"]
        M = obj["M"]
        P = obj["P"]

        rows.append({
            "id": tid,
            "name": name,
            "domain": obj["domain"],
            "optimistic_h": O,
            "most_likely_h": M,
            "pessimistic_h": P,
            "duration_days": _pert_days(O, M, P, hours_per_day),
            "effort_hours": _pert_mean_hours(O, M, P),
            "deps": deps,
            "baseline_h": base.get("baseline_h"),
            "llm_rationale": obj.get("rationale", ""),
            "llm_error": "",
            "engine_used": "llm_evidence",
        })

    return {
        "rows": rows,
        "raw": raw,
        "errors": errors,
        "rag_project_summary": rag_project_summary,
    }


def run_poker_batch_estimation(
    slots,
    wbs_items,
    rag_evidence,
    provider,
    model,
):
    leaf_items = _get_leaf_items(wbs_items)

    rag_project_summary = _build_unique_source_signals(rag_evidence)

    prompt = prompt_poker_batch(
        slots,
        wbs_items,
        leaf_items,
        rag_project_summary,
    )
    ss.planning_poker_prompt = prompt
    raw = run_llm(prompt, provider=provider, model=model)

    arr = _parse_llm_json_array(raw) or []
    valid_rows, errors = validate_poker_batch(arr, leaf_items)
    valid_by_id = {row["id"]: row for row in valid_rows}

    sp_to_hours = float(slots.get("estimation_params", {}).get("sp_to_hours", 2))
    hours_per_day = _hours_per_day_from_slots(slots)

    rows = []
    for it in leaf_items:
        tid = str(it.get("id", "")).strip()
        name = str(it.get("name", "")).strip()

        obj = valid_by_id.get(tid)
        if not obj:
            hours_est = 1.0 * sp_to_hours
            rows.append({
                "id": tid,
                "name": name,
                "domain": None,
                "sp_min": 1,
                "sp_max": 1,
                "sp_mid": 1.0,
                "hours_estimate": round(hours_est, 2),
                "duration_days": _pert_days(hours_est * 0.7, hours_est, hours_est * 1.6, hours_per_day),
                "reasoning": "",
                "risk_indicators": "",
                "baseline_h": None,
                "llm_error": "; ".join(errors)[:300],
                "engine_used": "poker_batch_failed",
            })
            continue

        sp_min = obj["sp_min"]
        sp_max = obj["sp_max"]
        sp_mid = (sp_min + sp_max) / 2.0
        hours_est = sp_mid * sp_to_hours

        rows.append({
            "id": tid,
            "name": name,
            "domain": obj["domain"],
            "sp_min": sp_min,
            "sp_max": sp_max,
            "sp_mid": round(sp_mid, 2),
            "hours_estimate": round(hours_est, 2),
            "duration_days": _pert_days(hours_est * 0.7, hours_est, hours_est * 1.6, hours_per_day),
            "reasoning": obj.get("reasoning", ""),
            "risk_indicators": ", ".join(obj.get("risk_indicators", [])),
            "baseline_h": None,
            "llm_error": "",
            "engine_used": "poker_batch_llm",
        })

    return {
        "rows": rows,
        "raw": raw,
        "errors": errors,
    }




















def _coerce_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _coerce_int(x: Any) -> Optional[int]:
    try:
        # allow "5" or 5.0, but reject 5.5
        f = float(x)
        i = int(f)
        if f != i:
            return None
        return i
    except Exception:
        return None


def _leaf_index(leaf_items: list) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in leaf_items:
        tid = str(it.get("id", "")).strip()
        if tid:
            out[tid] = it
    return out




_DEF_DOMAIN = {
    "requirements": ["requirement", "spec", "backlog"],
    "design": ["design", "diagram", "ui", "ux", "wireframe", "architecture"],
    "implementation": [
        "prototype", "implementation", "coding", "final product",
        "build", "deploy", "user manual", "testing"
    ],
}


def _domain_of(name: str) -> str:
    t = (name or "").lower()
    for dom, keys in _DEF_DOMAIN.items():
        if any(k in t for k in keys):
            return dom
    return "implementation"

def _pert_mean_hours(O: float, M: float, P: float) -> float:
    return round((O + 4 * M + P) / 6.0, 2)


def _pert_days(O: float, M: float, P: float, hours_per_day: float = 8.0) -> float:
    pert_mean_hours = _pert_mean_hours(O, M, P)
    return round(pert_mean_hours / max(hours_per_day, 0.1), 3)




def _hours_per_day_from_slots(slots: dict) -> float:
    hours_per_week = float(slots.get("hours_per_week", 12))
    days_per_week = 5.0
    return max(1.0, hours_per_week / days_per_week)



import pandas as pd

def _is_missing(x) -> bool:
    return pd.isna(x)







# -----------------------------
# Step 3: Estimation (GATED)
# -----------------------------
# NOTE: Here we reuse your existing outputs.
# To keep the file shorter, I only kept the gating + UI rendering,
# and I keep the JSON output in the same session fields: ss.estimation_tasks / ss.estimates_raw.
def _estimation_signature(slots: dict, provider: str, model: str, index_dir: str) -> str:
    """
    Build a small, stable signature of all inputs that affect estimation.
    If this signature changes, we must rebuild estimates.
    """
    sig_payload = {
        # Quantitative scope
        "req_pages": slots.get("req_pages"),
        "req_items": slots.get("req_items"),
        "screens": slots.get("screens"),
        "apis": slots.get("apis"),
        "integrations": slots.get("integrations"),

        # Risk modifiers
        "novelty_level": slots.get("novelty_level"),
        "risk_level": slots.get("risk_level"),
        "team_experience": slots.get("team_experience"),

        # RAG knobs
        "use_rag": bool(slots.get("use_rag", True)),
        "k_docs_wbs": slots.get("k_docs_wbs"),
        "min_similarity": slots.get("min_similarity"),

        # Estimation config
        "sp_to_hours": slots.get("estimation_params", {}).get("sp_to_hours"),

        # Provider / model / index
        "provider": provider,
        "model": model,
        "index_dir": index_dir,
    }


    return json.dumps(sig_payload, sort_keys=True)





def render_estimation():
    st.header("PERT Estimation")
    st.caption(
        "Review and edit the AI-generated PERT estimates."
    )
    st.info(
        "• RAG-assisted estimate = project context + retrieved evidence.  \n"
        "• Baseline without RAG = project context only."
    )


    if not ss.wbs_approved:
        st.warning("Estimation is locked. Finish WBS Review and press Continue first.")
        if st.button("⬅️ Go to WBS Review", width="stretch"):
            go("wbs_review")
        return

    wbs_items_local = ss.get("wbs_json", [])
    if not wbs_items_local:
        st.error("No finalized WBS found in session.")
        return

    hours_per_day = _hours_per_day_from_slots(ss.slots)
    ss.setdefault("last_estimation_signature", None)

    def _needs_rebuild_estimation() -> bool:
        tasks = ss.get("estimation_tasks")
        if not tasks or not isinstance(tasks, list):
            return True

        det = ss.get("estimation_det_tasks")
        if not det or not isinstance(det, list):
            return True

        sample_task = tasks[0] if tasks else {}
        sample_det = det[0] if det else {}

        if "engine_used" not in sample_task:
            return True
        if "engine_used" not in sample_det:
            return True

        current_sig = _estimation_signature(ss.slots, provider, model, ss.index_dir)
        if ss.get("last_estimation_signature") != current_sig:
            return True

        return False

    if _needs_rebuild_estimation():
        import copy

        leaf_items = _get_leaf_items(wbs_items_local)

        rag_obj = ss.rag if ss.slots.get("use_rag", True) else None
        rag_evidence = _collect_rag_evidence_for_tasks(
            rag=rag_obj,
            leaf_items=leaf_items,
            k=int(2),
            min_sim=float(ss.slots.get("min_similarity", DEFAULT_SLOTS["min_similarity"])),
        )

        baseline_result = run_context_only_batch_estimation(
            ss.slots,
            wbs_items_local,
            provider,
            model,
        )
        ss.baseline_estimates_raw = baseline_result.get("raw", "")
        baseline_rows = baseline_result["rows"]

        evidence_result = run_evidence_batch_estimation(
            ss.slots,
            wbs_items_local,
            baseline_rows,
            rag_evidence,
            provider,
            model,
        )

        ss.rag_project_summary = evidence_result.get("rag_project_summary", {})
        ss.estimation_det_tasks = baseline_rows
        ss.estimation_tasks = evidence_result["rows"]

        # Stable snapshot of the initial LLM output
        ss.estimation_det_tasks_original = copy.deepcopy(baseline_rows)
        ss.estimation_tasks_original = copy.deepcopy(evidence_result["rows"])

        ss.estimates_raw = evidence_result["raw"]
        ss.last_estimation_signature = _estimation_signature(
            ss.slots,
            provider,
            model,
            ss.index_dir,
        )

    ss.estimation_tasks = _normalize_estimation_items(ss.get("estimation_tasks", []), hours_per_day)
    est_rows = ss.estimation_tasks

    # ---------- JSON editor state ----------
    if "estimation_json_editor" not in ss:
        ss.estimation_json_editor = json.dumps(est_rows, ensure_ascii=False, indent=2)

    if "estimation_json_editor_pending" not in ss:
        ss.estimation_json_editor_pending = None

    if ss.estimation_json_editor_pending is not None:
        ss.estimation_json_editor = ss.estimation_json_editor_pending
        ss.estimation_json_editor_pending = None

    # ---------- Summary ----------
    evidence_capacity_summary = _compute_capacity_summary_from_rows(
        est_rows,
        ss.slots,
        effort_key="effort_hours",
    )
    ss.capacity_summary_evidence = evidence_capacity_summary

    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    total_effort = sum(_safe_float(t.get("effort_hours", 0.0)) for t in est_rows)
    total_days = sum(_safe_float(t.get("duration_days", 0.0)) for t in est_rows)
    failed_count = sum(
        1 for t in est_rows
        if _is_missing(t.get("optimistic_h"))
        or _is_missing(t.get("most_likely_h"))
        or _is_missing(t.get("pessimistic_h"))
    )

    st.markdown("### 1️⃣ Edit JSON")

    json_text = st.text_area(
        "RAG-assisted Estimate",
        height=420,
        key="estimation_json_editor",
    )

    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("Apply to Preview", use_container_width=True):
            try:
                parsed = json.loads(json_text)
                if not isinstance(parsed, list):
                    raise ValueError("Estimates JSON must be a JSON array.")

                normalized = _normalize_estimation_items(parsed, hours_per_day)
                ss.estimation_tasks = normalized
                ss.estimation_csv = _csv_from_estimates(normalized)
                ss.estimation_json_editor_pending = json.dumps(normalized, ensure_ascii=False, indent=2)
                st.rerun()
            except Exception as ex:
                st.error(f"Invalid JSON: {ex}")

    with b2:
        if st.button("Format JSON", use_container_width=True):
            try:
                parsed = json.loads(json_text)
                if not isinstance(parsed, list):
                    raise ValueError("Estimates JSON must be a JSON array.")

                normalized = _normalize_estimation_items(parsed, hours_per_day)
                ss.estimation_tasks = normalized
                ss.estimation_csv = _csv_from_estimates(normalized)
                ss.estimation_json_editor_pending = json.dumps(normalized, ensure_ascii=False, indent=2)
                st.rerun()
            except Exception as ex:
                st.error(f"Cannot format invalid JSON: {ex}")

    with b3:
        if st.button("Reset to RAG-assisted estimate", use_container_width=True):
            source_rows = ss.get("estimation_tasks_original") or []
            rebuilt = _normalize_estimation_items(source_rows, hours_per_day)

            ss.estimation_tasks = rebuilt
            ss.estimation_csv = _csv_from_estimates(rebuilt)
            ss.estimation_json_editor_pending = json.dumps(rebuilt, ensure_ascii=False, indent=2)
            st.rerun()

    st.markdown("### 2️⃣ Summary")
    st.caption("RAG-assisted estimate")
    st.caption(
        "Formulas: Effort (h) = (O + 4×M + P) / 6, "
        "Duration (days) = Effort / hours_per_day"
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Estimated Effort (h)", round(total_effort, 2))
    m2.metric("Estimated Days", round(total_days, 2))
    m3.metric("Available Capacity (h)", evidence_capacity_summary["available_capacity_hours"])
    m4.metric("Capacity Usage (%)", evidence_capacity_summary["capacity_usage_percent"])
    m5.metric("Failed Tasks", f"{failed_count}/{len(est_rows)}")

    st.markdown("### 3️⃣ Preview")
    st.caption("RAG-assisted estimate")

    df_preview = pd.DataFrame(est_rows)
    if "deps" in df_preview.columns:
        df_preview["deps"] = df_preview["deps"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else ("" if x is None else str(x))
        )

    st.dataframe(df_preview, use_container_width=True, hide_index=True)

    baseline_capacity_summary = _compute_capacity_summary_from_rows(
        ss.get("estimation_det_tasks", []),
        ss.slots,
        effort_key="baseline_h",
    )
    ss.capacity_summary_baseline = baseline_capacity_summary

    with st.expander("Baseline without RAG (comparison)", expanded=False):

        det_rows = ss.get("estimation_det_tasks", []) or []

        if not det_rows:
            st.caption("No baseline estimates available yet.")
        else:

            # ----- baseline summary -----
            det_total_effort = sum(_safe_float(t.get("baseline_h", 0.0)) for t in det_rows)

            det_total_days = sum(
                _safe_float(
                    _pert_days(
                        _safe_float(t.get("O", 0.0)),
                        _safe_float(t.get("M", 0.0)),
                        _safe_float(t.get("P", 0.0)),
                        hours_per_day,
                    )
                )
                for t in det_rows
            )

            baseline_capacity_summary = _compute_capacity_summary_from_rows(
                det_rows,
                ss.slots,
                effort_key="baseline_h",
            )

            st.markdown("**Baseline Summary (context-only)**")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Estimated Effort (h)",
                round(det_total_effort, 2)
            )

            c2.metric(
                "Estimated Duration (days)",
                round(det_total_days, 2)
            )

            c3.metric(
                "Available Capacity (h)",
                baseline_capacity_summary["available_capacity_hours"]
            )

            c4.metric(
                "Capacity Usage (%)",
                baseline_capacity_summary["capacity_usage_percent"]
            )

            st.markdown("---")

            # ----- baseline table -----
            det_df = pd.DataFrame(det_rows)
            st.dataframe(det_df, use_container_width=True, hide_index=True)

    # validation
    issues = []
    for i, row in enumerate(est_rows):
        if not str(row.get("id", "")).strip():
            issues.append(f"Row {i}: empty id")
        if not str(row.get("name", "")).strip():
            issues.append(f"Row {i}: empty name")

        O = row.get("optimistic_h")
        M = row.get("most_likely_h")
        P = row.get("pessimistic_h")

        if O is None or M is None or P is None:
            issues.append(f"Row {i} ({row.get('id','')}): missing O/M/P")
        else:
            try:
                if not (float(O) < float(M) < float(P)):
                    issues.append(f"Row {i} ({row.get('id','')}): must satisfy optimistic < most_likely < pessimistic")
            except Exception:
                issues.append(f"Row {i} ({row.get('id','')}): invalid numeric values")

    if issues:
        st.warning("Estimation JSON issues (fix before continuing):")
        for msg in issues:
            st.write(f"- {msg}")
    else:
        st.success("Estimation JSON looks good.")

    st.markdown("---")
    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("⬅️ Back to WBS Review", width="stretch"):
            go("wbs_review")

    with c2:
        can_continue = len(issues) == 0 and len(est_rows) > 0
        if st.button("Continue ➡️", width="stretch", disabled=not can_continue):
            ss.estimation_tasks = est_rows
            ss.estimation_csv = _csv_from_estimates(est_rows)
            ss.estimation_approved = True
            go("planning_poker")


def render_planning_poker():
    st.header("Planning Poker")
    st.caption("Review and edit the AI-generated planning poker estimates.")

    if not (ss.wbs_approved and ss.estimation_approved):
        st.warning("Planning Poker is locked. Complete WBS Review and PERT Estimation first.")
        if st.button("⬅️ Go to PERT Estimation", use_container_width=True):
            go("estimation")
        return

    wbs_items_local = ss.get("wbs_json", [])
    if not wbs_items_local:
        st.error("No finalized WBS found in session.")
        return

    hours_per_day = _hours_per_day_from_slots(ss.slots)
    sp_to_hours = float(ss.slots.get("estimation_params", {}).get("sp_to_hours", 2))
    leaf_items = _get_leaf_items(wbs_items_local)

    st.info(
        f"Planning Poker estimate = AI-generated story point ranges.  \n"
        f"Current SP → hours mapping = 1 SP ≈ {sp_to_hours} hours."
    )

    ss.setdefault("last_poker_signature", None)

    def _poker_signature(slots: dict, provider: str, model: str, index_dir: str) -> str:
        sig_payload = {
            "wbs_leaf_ids": [str(it.get("id", "")).strip() for it in leaf_items],
            "wbs_leaf_names": [str(it.get("name", "")).strip() for it in leaf_items],
            "use_rag": bool(slots.get("use_rag", True)),
            "k_docs_wbs": slots.get("k_docs_wbs"),
            "min_similarity": slots.get("min_similarity"),
            "sp_to_hours": slots.get("estimation_params", {}).get("sp_to_hours"),
            "novelty_level": slots.get("novelty_level"),
            "risk_level": slots.get("risk_level"),
            "team_experience": slots.get("team_experience"),
            "provider": provider,
            "model": model,
            "index_dir": index_dir,
        }
        return json.dumps(sig_payload, sort_keys=True)

    def _needs_rebuild_poker() -> bool:
        rows = ss.get("poker_tasks")
        if not rows or not isinstance(rows, list):
            return True

        sample = rows[0] if rows else {}
        if "engine_used" not in sample:
            return True

        current_sig = _poker_signature(ss.slots, provider, model, ss.index_dir)
        if ss.get("last_poker_signature") != current_sig:
            return True

        return False

    if _needs_rebuild_poker():
        import copy

        rag_obj = ss.rag if ss.slots.get("use_rag", True) else None
        rag_evidence = _collect_rag_evidence_for_tasks(
            rag=rag_obj,
            leaf_items=leaf_items,
            k=int(2),
            min_sim=float(ss.slots.get("min_similarity", DEFAULT_SLOTS["min_similarity"])),
        )

        poker_result = run_poker_batch_estimation(
            ss.slots,
            wbs_items_local,
            rag_evidence,
            provider,
            model,
        )

        ss.poker_tasks = poker_result["rows"]
        ss.poker_tasks_original = copy.deepcopy(poker_result["rows"])
        ss.poker_raw = poker_result["raw"]
        ss.last_poker_signature = _poker_signature(ss.slots, provider, model, ss.index_dir)

    ss.poker_tasks = _normalize_poker_items(ss.get("poker_tasks", []), sp_to_hours, hours_per_day)
    poker_rows = ss.poker_tasks

    if "poker_json_editor" not in ss:
        ss.poker_json_editor = json.dumps(poker_rows, ensure_ascii=False, indent=2)

    if "poker_json_editor_pending" not in ss:
        ss.poker_json_editor_pending = None

    if ss.poker_json_editor_pending is not None:
        ss.poker_json_editor = ss.poker_json_editor_pending
        ss.poker_json_editor_pending = None

    poker_capacity_summary = _compute_capacity_summary_from_rows(
        poker_rows,
        ss.slots,
        effort_key="hours_estimate",
    )
    ss.capacity_summary_poker = poker_capacity_summary

    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    total_sp_mid = sum(_safe_float(t.get("sp_mid", 0.0)) for t in poker_rows)
    total_hours = sum(_safe_float(t.get("hours_estimate", 0.0)) for t in poker_rows)
    total_days = sum(_safe_float(t.get("duration_days", 0.0)) for t in poker_rows)

    failed_count = sum(
        1 for t in poker_rows
        if not t.get("sp_min") or not t.get("sp_max")
    )

    st.markdown("### 1️⃣ Edit JSON")

    json_text = st.text_area(
        "Planning Poker JSON",
        height=420,
        key="poker_json_editor",
    )

    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("Apply to Preview", use_container_width=True):
            try:
                parsed = json.loads(json_text)
                if not isinstance(parsed, list):
                    raise ValueError("Planning Poker JSON must be a JSON array.")

                normalized = _normalize_poker_items(parsed, sp_to_hours, hours_per_day)
                ss.poker_tasks = normalized
                ss.poker_json_editor_pending = json.dumps(normalized, ensure_ascii=False, indent=2)
                st.rerun()
            except Exception as ex:
                st.error(f"Invalid JSON: {ex}")

    with b2:
        if st.button("Format JSON", use_container_width=True):
            try:
                parsed = json.loads(json_text)
                if not isinstance(parsed, list):
                    raise ValueError("Planning Poker JSON must be a JSON array.")

                normalized = _normalize_poker_items(parsed, sp_to_hours, hours_per_day)
                ss.poker_tasks = normalized
                ss.poker_json_editor_pending = json.dumps(normalized, ensure_ascii=False, indent=2)
                st.rerun()
            except Exception as ex:
                st.error(f"Cannot format invalid JSON: {ex}")

    with b3:
        if st.button("Reset to AI-generated estimate", use_container_width=True):
            source_rows = ss.get("poker_tasks_original") or []
            rebuilt = _normalize_poker_items(source_rows, sp_to_hours, hours_per_day)

            ss.poker_tasks = rebuilt
            ss.poker_json_editor_pending = json.dumps(rebuilt, ensure_ascii=False, indent=2)
            st.rerun()

    st.markdown("### 2️⃣ Summary")
    st.caption("Planning Poker estimate")
    st.caption(
        "Formulas: SP midpoint = (sp_min + sp_max) / 2, "
        "Effort (h) = SP midpoint × sp_to_hours, "
        "Duration (days) = Effort / hours_per_day"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total SP", round(total_sp_mid, 2))
    m2.metric("Estimated Effort (h)", round(total_hours, 2))
    m3.metric("Estimated Days", round(total_days, 2))
    m4.metric("Available Capacity (h)", poker_capacity_summary["available_capacity_hours"])
    m5.metric("Capacity Usage (%)", poker_capacity_summary["capacity_usage_percent"])

    st.markdown("### 3️⃣ Preview")
    st.caption("Planning Poker estimate")

    df_preview = pd.DataFrame(poker_rows)
    st.dataframe(df_preview, use_container_width=True, hide_index=True)

    issues = []
    fib = {1, 2, 3, 5, 8, 13, 21}

    for i, row in enumerate(poker_rows):
        if not str(row.get("id", "")).strip():
            issues.append(f"Row {i}: empty id")
        if not str(row.get("name", "")).strip():
            issues.append(f"Row {i}: empty name")

        sp_min = row.get("sp_min")
        sp_max = row.get("sp_max")

        if sp_min not in fib or sp_max not in fib:
            issues.append(f"Row {i} ({row.get('id','')}): sp_min/sp_max must be Fibonacci values")
        elif sp_max < sp_min:
            issues.append(f"Row {i} ({row.get('id','')}): sp_max must be >= sp_min")

    if issues:
        st.warning("Planning Poker JSON issues (fix before continuing):")
        for msg in issues:
            st.write(f"- {msg}")
    else:
        st.success("Planning Poker JSON looks good.")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("⬅️ Back to PERT Estimation", use_container_width=True):
            go("estimation")

    with c2:
        can_continue = len(issues) == 0 and len(poker_rows) > 0
        if st.button("Continue ➡️", use_container_width=True, disabled=not can_continue):
            ss.poker_tasks = poker_rows
            ss.poker_approved = True
            go("gantt")

# -----------------------------
# Step 4: Gantt (GATED)
# -----------------------------


def render_gantt():
    st.header("Gantt")
    st.caption("This schedule is generated based on the RAG-assisted PERT estimation results.")

    if not (ss.wbs_approved and ss.estimation_approved and ss.poker_approved):
        st.warning("Gantt is locked. Complete Planning Poker first.")
        if st.button("⬅️ Go to Estimation", width="stretch"):
            go("estimation")
        return

    tasks_all = ss.get("estimation_tasks", [])
    tasks = [
        t for t in tasks_all
        if not _is_missing(t.get("optimistic_h"))
        and not _is_missing(t.get("most_likely_h"))
        and not _is_missing(t.get("pessimistic_h"))
    ]

    if not tasks:
        st.error("No valid LLM estimation tasks found for Gantt generation.")
        return

    gantt_docs = []
    try:
        k_docs_wbs = int(ss.slots.get("k_docs_wbs", 5))
        gantt_docs = (
            ss.rag.search("Gantt chart scheduling for student projects", k=k_docs_wbs)
            if hasattr(ss.rag, "search")
            else []
        )
    except Exception:
        gantt_docs = []

    estimates_json_text = json.dumps(tasks, ensure_ascii=False, indent=2)
    gantt_prompt = prompt_gantt(gantt_docs, ss.slots, estimates_json_text)
    ss.gantt_prompt = gantt_prompt
    def _parse_gantt_blocks(raw_text: str):
        mermaid = ""
        csv_text = ""
        cp_items = []

        if not raw_text:
            return mermaid, csv_text, cp_items

        m_mer = re.search(r"```(?:mermaid)?\s*(gantt[\s\S]*?)```", raw_text, re.I)
        if m_mer:
            mermaid = m_mer.group(1).strip()

        m_csv = re.search(r"```csv\s*([\s\S]*?)```", raw_text, re.I)
        if m_csv:
            csv_text = m_csv.group(1).strip()

        cp_text = ""
        m_cp = re.search(
            r"(?:Critical Path|Critical path).*?\n([\s\S]*?)(?:\n#{1,6}|\Z)",
            raw_text,
            re.I,
        )
        if m_cp:
            cp_text = m_cp.group(1).strip()

        for line in cp_text.splitlines():
            line = re.sub(r"^\s*(?:[\-\*]|\d+[\.\)])\s*", "", line.strip())
            if line:
                cp_items.append(line)

        return mermaid, csv_text, cp_items

    # ---- session state init ----
    ss.setdefault("gantt_raw", "")
    ss.setdefault("gantt_raw_original", "")
    ss.setdefault("gantt_raw_editable", "")
    ss.setdefault("gantt_raw_editor_pending", None)


    if ss.gantt_raw_editor_pending is not None:
        ss.gantt_raw_editable = ss.gantt_raw_editor_pending
        ss.gantt_raw_editor_pending = None

    if st.button("Generate Gantt", width="stretch"):
        raw = run_llm(gantt_prompt, provider=provider, model=model)

        ss.gantt_raw_original = raw
        ss.gantt_raw_editable = raw
        ss.gantt_raw = raw

        st.rerun()


    current_gantt_text = ss.get("gantt_raw_editable", "")

    if current_gantt_text:
        mermaid_text, csv_text, cp_items = _parse_gantt_blocks(current_gantt_text)
        ss.gantt_mermaid = mermaid_text
        ss.gantt_csv = csv_text
        ss.gantt_cp = cp_items
        ss.gantt_raw = current_gantt_text

        st.markdown("---")
        st.subheader("LLM Output")

        tab_edit, tab_original = st.tabs(["Editable", "Original"])

        with tab_edit:
            ss.setdefault("gantt_raw_editable", "")
            ss.setdefault("gantt_raw_editor_pending", None)

            # apply pending editor value BEFORE creating the widget
            if ss.gantt_raw_editor_pending is not None:
                ss.gantt_raw_editable = ss.gantt_raw_editor_pending
                ss.gantt_raw_editor_pending = None

            st.text_area(
                "Editable Gantt output",
                height=520,
                key="gantt_raw_editable",
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                if st.button("Apply edits", use_container_width=True):
                    ss.gantt_raw = ss.get("gantt_raw_editable", "")
                    st.rerun()

            with c2:
                if st.button("Format text", use_container_width=True):
                    formatted = (ss.get("gantt_raw_editable", "") or "").strip()
                    ss.gantt_raw = formatted
                    ss.gantt_raw_editor_pending = formatted
                    st.rerun()

            with c3:
                if st.button("Reset editable to original", use_container_width=True):
                    original_text = ss.get("gantt_raw_original", "") or ""
                    ss.gantt_raw = original_text
                    ss.gantt_raw_editor_pending = original_text
                    st.rerun()

        with tab_original:
            st.text_area(
                "Original Gantt output",
                value=ss.get("gantt_raw_original", ""),
                height=520,
                disabled=True,
            )

        st.markdown("---")
        st.subheader("Parsed Preview")

        preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Mermaid", "CSV", "Critical Path"])

        with preview_tab1:
            if ss.get("gantt_mermaid"):
                st.code(ss.gantt_mermaid, language="mermaid")
            else:
                st.caption("No Mermaid block detected.")

        with preview_tab2:
            if ss.get("gantt_csv"):
                st.code(ss.gantt_csv, language="csv")
            else:
                st.caption("No CSV block detected.")

        with preview_tab3:
            if ss.get("gantt_cp"):
                for i, item in enumerate(ss.gantt_cp, start=1):
                    st.markdown(f"{i}. {item}")
            else:
                st.caption("No critical path detected.")

    st.markdown("---")
    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("⬅️ Back to Estimation", width="stretch"):
            go("estimation")

    with c2:
        can_continue = bool(ss.get("gantt_raw_editable"))
        if st.button("Continue ➡️", width="stretch", disabled=not can_continue):
            ss.gantt_approved = True
            go("survey")



# -----------------------------
# Step 5: Survey
# -----------------------------
def render_survey():
    st.header("Feedback Survey")

    if not (ss.wbs_approved and ss.estimation_approved and ss.gantt_approved and ss.poker_approved):
        st.warning("Survey is locked. Complete workflow first.")
        if st.button("⬅️ Go to Gantt", width="stretch"):
            go("gantt")
        return

    st.caption("Quick feedback to support evaluation (Design Science iterations).")

    # ✅ init flag once per session
    ss.setdefault("survey_submitted", False)

    usefulness = st.slider("Usefulness (How useful was the AI-generated output?)", 1, 5, 3)
    clarity = st.slider("Clarity (How clear and understandable were the generated tasks?)", 1, 5, 3)
    trust = st.slider("Trust in estimates (How realistic did the estimates seem?)", 1, 5, 3)
    ease = st.slider("Ease of editing (How easy was it to modify the generated results?)", 1, 5, 3)

    # -----------------------------
    # Part 2 – Method Comparison
    # -----------------------------
    st.subheader("Method Comparison")

    most_accurate_method = st.radio(
        "Which estimation method was the most accurate?",
        options=[
            "RAG-assisted estimation",
            "Baseline without RAG estimation",
            "Planning Poker"
        ]
    )

    # -----------------------------
    # Part 2.5 – Adoption
    # -----------------------------
    st.subheader("Adoption")

    would_use = st.radio(
        "Would you use a tool like this in a real software project?",
        ["Yes", "Maybe", "No"]
    )

    # -----------------------------
    # Part 3 – Open Feedback
    # -----------------------------
    st.subheader("Open Feedback")

    liked_about_tool = st.text_area(
        "What did you like about the tool?",
        height=100
    )

    problems_faced = st.text_area(
        "What problems did you face?",
        height=100
    )

    improvement_suggestions = st.text_area(
        "What improvements do you suggest?",
        height=100
    )


    st.markdown("---")

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        if st.button("⬅️ Back to Gantt", width="stretch"):
            go("gantt")

    with c2:
        if st.button("Submit survey", width="stretch"):
            ss.setdefault("survey_responses", [])
            ss.survey_responses.append({
                "usefulness": usefulness,
                "clarity": clarity,
                "trust": trust,
                "ease": ease,
                "would_use_tool": would_use,

                "most_accurate_method": most_accurate_method,
                "liked_about_tool": liked_about_tool,
                "problems_faced": problems_faced,
                "improvement_suggestions": improvement_suggestions,

                "ts": dt.datetime.now().isoformat()
            })
            ss.survey_submitted = True


    with c3:
        if st.button("🔁 Restart", width="stretch"):
            # optional: reset approval flags too if you want a clean run
            go("setup")

    # ✅ download appears AFTER submit (and stays after rerun)
    if ss.get("survey_submitted", False):
        st.markdown("---")
        st.subheader("Download final submission")

        st.info(
            "After downloading the final submission (.zip), please send the file by email.\n\n"
            "**Email:** leilamo@stud.ntnu.no"
        )

        participant_slug = _safe_slug(ss.get("participant_id", "anon"))
        today_tag = dt.datetime.now().strftime("%Y%m%d")
        zip_bytes = build_submission_zip()

        st.download_button(
            label="⬇️ Download Final Submission (.zip)",
            data=zip_bytes,
            file_name=f"{participant_slug}_submission_{today_tag}.zip",
            mime="application/zip",
            use_container_width=True
        )

        st.success("Thanks!")



# -----------------------------
# Main wizard router
# -----------------------------
step = ss.wizard_step
if not can_enter(step):
    # force back to nearest valid
    if step == "estimation":
        go("wbs_review")
    elif step == "planning_poker":
        go("estimation")
    elif step == "gantt":
        go("estimation")
    else:
        go("setup")

if ss.wizard_step == "setup":
    render_setup()
elif ss.wizard_step == "wbs_draft":
    render_wbs_draft()
elif ss.wizard_step == "wbs_review":
    render_wbs_review()
elif ss.wizard_step == "estimation":
    render_estimation()
elif ss.wizard_step == "planning_poker":
    render_planning_poker()
elif ss.wizard_step == "gantt":
    render_gantt()
elif ss.wizard_step == "survey":
    render_survey()

