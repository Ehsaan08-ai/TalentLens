import streamlit as st
from typing import Any
from icrs.ui.dashboard import (
    transform_response_to_rows,
    format_score,
    BREAKDOWN_LABELS,
)

def render_explainability(st: Any) -> None:
    st.markdown("<h1 class='premium-title'>Explainability Panel</h1>", unsafe_allow_html=True)
    st.caption(
        "Deep dive into matching rationale, skill gaps, and candidate signal breakdowns."
    )
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    if st.session_state.get("ranking_response") is None:
        st.info("No candidate pools have been ranked yet. Run a candidate ranking first to explore explainability.")
        return

    # Use backend ranking directly — no client-side re-weighting.
    rows = transform_response_to_rows(st.session_state.ranking_response, st.session_state.get("uuid_to_name"))

    if not rows:
        st.info("No candidates were ranked. (All may have been excluded before ranking.)")
        return

    # Build UUID -> display_name mapping for the selectbox.
    # Use raw_candidate_id (UUID) as the internal value to avoid ambiguity
    # when multiple candidates share a display name (Bug 3).
    candidate_ids = [r["raw_candidate_id"] for r in rows]
    id_to_display = {r["raw_candidate_id"]: f"#{r['rank']} — {r['display_name']}" for r in rows}

    # Try to select the previously selected candidate from state
    default_index = 0
    selected_id = st.session_state.get("selected_candidate_id")
    if selected_id in candidate_ids:
        default_index = candidate_ids.index(selected_id)

    selected_raw_id = st.selectbox(
        "Select Candidate to Inspect:",
        options=candidate_ids,
        index=default_index,
        format_func=lambda cid: id_to_display.get(cid, cid),
        key="explainability_candidate_select",
    )
    # Update selection in session state (store the UUID, not display name)
    st.session_state.selected_candidate_id = selected_raw_id

    row = next(r for r in rows if r["raw_candidate_id"] == selected_raw_id)
    display_name = row["display_name"]

    st.markdown(f"### Why did we rank {display_name} #{row['rank']}?")

    # ── Confidence Score ──
    conf_label = row["confidence_label"]
    conf_val = row["confidence"]

    overall_match_pct = f"{int(round(float(row['final_score']) * 100))}%" if row['final_score'] is not None else "—"

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            f"""
            <div class="fit-item" style="padding: 16px; margin-bottom: 16px;">
                <div class="fit-label">Overall Match</div>
                <div class="fit-value" style="font-size: 24px; color: #7c3aed;">{overall_match_pct}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_m2:
        badge_color = "#10b981" if conf_label == "High" else "#f59e0b" if conf_label == "Moderate" else "#ef4444"
        st.markdown(
            f"""
            <div class="fit-item" style="padding: 16px; margin-bottom: 16px;">
                <div class="fit-label">Ranking Confidence</div>
                <div class="fit-value" style="font-size: 24px; color: {badge_color};">{conf_label} Confidence ({format_score(conf_val)})</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── Sub-score Breakdown (from real backend data) ──
    semantic = row['breakdown'].get('semantic_fit')
    career = row['breakdown'].get('career_trajectory')
    behavioral = row['breakdown'].get('behavioral')
    hard_filter = row['breakdown'].get('hard_filter_pass')
    penalty = row['breakdown'].get('disqualifying_penalty')

    col1, col2 = st.columns(2)

    with col1:
        # ── Driving Signals & Gaps (real backend data, not fabricated) ──
        st.markdown("### Strengths & Gaps")

        driving_signals = row['driving_signals']
        gaps = row['gaps']
        unmet_must_haves = row['unmet_must_haves']

        if driving_signals:
            matched_html = "".join([
                f'<div style="margin-bottom:6px; color:#10b981; font-weight:600;">✓ {s}</div>'
                for s in driving_signals
            ])
        else:
            matched_html = '<div style="color: #64748b; font-style: italic; font-size: 13px;">No specific strengths identified by the backend</div>'

        all_gaps = list(unmet_must_haves) + list(gaps)
        if all_gaps:
            missing_html = "".join([
                f'<div style="margin-bottom:6px; color:#f59e0b; font-weight:600;">⚠ {m}</div>'
                for m in all_gaps
            ])
        else:
            missing_html = '<div style="color: #64748b; font-style: italic; font-size: 13px;">No significant gaps identified</div>'

        st.markdown(
            f"""
            <div class="glass-card" style="padding: 20px; margin-bottom: 20px;">
                <div style="font-weight:700; color:#334155; margin-bottom:8px;">Driving Signals:</div>
                {matched_html}
                <div style="font-weight:700; color:#334155; margin-top:16px; margin-bottom:8px;">Gaps & Unmet Must-Haves:</div>
                {missing_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Explanation Summary (from backend) ──
        st.markdown("### Explanation Summary")
        summary = row.get("summary", "")
        if summary and summary.strip():
            st.markdown(
                f"""
                <div class="glass-card" style="padding: 20px; margin-bottom: 20px;">
                    <div style="font-weight:600; color:#1e293b; line-height: 1.6;">{summary}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No explanation summary available for this candidate.")

    with col2:
        # ── Signal Breakdown Visualizer (real backend scores) ──
        st.markdown("### Signal Breakdown")

        breakdown_items = [
            ("Semantic Fit", semantic, "#7c3aed"),
            ("Career Trajectory", career, "#2563eb"),
            ("Behavioral", behavioral, "#10b981"),
            ("Hard-Filter Coverage", hard_filter, "#f59e0b"),
        ]

        bars_html = ""
        for label, value, color in breakdown_items:
            if value is not None:
                pct = int(round(float(value) * 100))
                val_display = f"{pct}%"
            else:
                pct = 0
                val_display = "—"
            bars_html += f"""
                <div style="margin-bottom: 12px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:600; color:#475569; margin-bottom:4px;">
                        <span>{label}</span>
                        <span>{val_display}</span>
                    </div>
                    <div style="height:8px; background-color:#e2e8f0; border-radius:9999px; overflow:hidden;">
                        <div style="width:{pct}%; height:100%; background-color:{color}; border-radius:9999px;"></div>
                    </div>
                </div>
            """

        # Penalty (shown as a separate deduction)
        if penalty is not None:
            penalty_pct = int(round(float(penalty) * 100))
            penalty_display = f"-{penalty_pct}%"
        else:
            penalty_pct = 0
            penalty_display = "—"
        bars_html += f"""
            <div style="margin-bottom: 12px;">
                <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:600; color:#475569; margin-bottom:4px;">
                    <span>Disqualifying Penalty</span>
                    <span style="color:#ef4444;">{penalty_display}</span>
                </div>
                <div style="height:8px; background-color:#e2e8f0; border-radius:9999px; overflow:hidden;">
                    <div style="width:{penalty_pct}%; height:100%; background-color:#ef4444; border-radius:9999px;"></div>
                </div>
            </div>
        """

        st.markdown(
            f"""
            <div class="glass-card" style="padding: 20px;">
                {bars_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Explanation Availability Notice ──
        if not row.get("explanation_available", True):
            st.warning(
                "Explanation is unavailable for this candidate. "
                "The score and signal breakdown remain valid, but no "
                "rationale could be generated."
            )

    # ── Candidate Comparison Matrix (uses real backend data) ──
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("### Candidate Comparison Matrix")
    st.caption("Compare the active candidate directly against any other candidate in the pool.")

    comparison_ids = [cid for cid in candidate_ids if cid != selected_raw_id]
    if comparison_ids:
        compare_with_id = st.selectbox(
            "Compare with:",
            options=comparison_ids,
            format_func=lambda cid: id_to_display.get(cid, cid),
            key="compare_select",
        )
        compare_row = next(r for r in rows if r["raw_candidate_id"] == compare_with_id)
        compare_name = compare_row["display_name"]

        skills_active = int(round(float(row['breakdown'].get('semantic_fit') or 0) * 100))
        skills_comp = int(round(float(compare_row['breakdown'].get('semantic_fit') or 0) * 100))

        exp_active = int(round(float(row['breakdown'].get('career_trajectory') or 0) * 100))
        exp_comp = int(round(float(compare_row['breakdown'].get('career_trajectory') or 0) * 100))

        beh_active = int(round(float(row['breakdown'].get('behavioral') or 0) * 100))
        comp_beh = int(round(float(compare_row['breakdown'].get('behavioral') or 0) * 100))

        if row['rank'] < compare_row['rank']:
            better_cand = display_name
            worse_cand = compare_name
            better_rank = row['rank']
            worse_rank = compare_row['rank']
        else:
            better_cand = compare_name
            worse_cand = display_name
            better_rank = compare_row['rank']
            worse_rank = row['rank']

        ai_summary_txt = (
            f"**{better_cand}** is ranked #{better_rank} while **{worse_cand}** is ranked #{worse_rank}. "
            f"{better_cand} has stronger overall alignment in critical dimensions, leading to a higher placement. "
        )
        if skills_active > skills_comp:
            ai_summary_txt += f"Specifically, {display_name} shows stronger skill match ({skills_active}% vs {skills_comp}%). "
        elif skills_comp > skills_active:
            ai_summary_txt += f"Specifically, {compare_name} shows stronger skill match ({skills_comp}% vs {skills_active}%), but {display_name} overrides this due to behavioral or trajectory alignment. "

        if exp_active > exp_comp:
            ai_summary_txt += f"{display_name} also has a higher career trajectory index ({exp_active}% vs {exp_comp}%)."
        elif exp_comp > exp_active:
            ai_summary_txt += f"{compare_name} has a higher experience score ({exp_comp}% vs {exp_active}%), but {display_name} demonstrates stronger behavioral signals."

        st.markdown(
            f"""
            <table style="width:100%; border-collapse: collapse; margin-bottom: 16px;">
                <thead>
                    <tr style="background-color: #f1f5f9;">
                        <th style="padding: 10px; text-align: left; border-bottom: 2px solid #e2e8f0; color:#0f172a;">Factor</th>
                        <th style="padding: 10px; text-align: center; border-bottom: 2px solid #e2e8f0; color:#0f172a;">{display_name} (Rank #{row['rank']})</th>
                        <th style="padding: 10px; text-align: center; border-bottom: 2px solid #e2e8f0; color:#0f172a;">{compare_name} (Rank #{compare_row['rank']})</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight:600;">Skills</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0; color:#10b981; font-weight:700;">{skills_active}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0; color:#3b82f6; font-weight:700;">{skills_comp}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight:600;">Experience</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0; color:#10b981; font-weight:700;">{exp_active}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0; color:#3b82f6; font-weight:700;">{exp_comp}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight:600;">Behavioral</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0; color:#10b981; font-weight:700;">{beh_active}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0; color:#3b82f6; font-weight:700;">{comp_beh}</td>
                    </tr>
                </tbody>
            </table>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 14px; border-radius: 0 8px 8px 0; margin-bottom: 24px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 4px;">Comparison Summary (derived from backend scores):</div>
                <div style="color: #1e293b; font-size:14px; line-height: 1.5;">{ai_summary_txt}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Provide a larger candidate pool to compare candidates.")

    # ── AI Recruiter Chat (keyword-based heuristic, not LLM-backed) ──
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("### AI Recruiter Chat")
    st.caption(
        "Ask questions about the candidate pool to quickly search or compare characteristics. "
        "*(Responses are generated from backend ranking data using keyword matching, not an LLM.)*"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        role_style = "background-color: #eff4ff; border-left: 4px solid #7c3aed;" if chat["role"] == "user" else "background-color: #f1f5f9; border-left: 4px solid #475569;"
        role_label = "Recruiter" if chat["role"] == "user" else "TalentLens AI"
        st.markdown(
            f"""
            <div style="{role_style} padding: 10px; border-radius: 0 8px 8px 0; margin-bottom: 8px;">
                <div style="font-size: 10px; font-weight: 700; color: #64748b; margin-bottom: 2px;">{role_label}:</div>
                <div style="color: #1e293b; font-size: 14px;">{chat["text"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input("Ask about candidates...", placeholder="e.g. Who has the strongest leadership signals?")
        submitted = st.form_submit_button("Send")
        if submitted and user_msg.strip():
            st.session_state.chat_history.append({"role": "user", "text": user_msg.strip()})

            msg_lower = user_msg.lower()
            response_txt = ""

            candidate_details = []
            for r in rows:
                candidate_details.append({
                    "name": r["display_name"],
                    "skills": int(round(float(r["breakdown"].get("semantic_fit") or 0) * 100)),
                    "exp": int(round(float(r["breakdown"].get("career_trajectory") or 0) * 100)),
                    "beh": int(round(float(r["breakdown"].get("behavioral") or 0) * 100)),
                    "gaps": [g.lower() for g in (r["unmet_must_haves"] + r["gaps"])]
                })

            if "leadership" in msg_lower or "ownership" in msg_lower:
                by_beh = sorted(candidate_details, key=lambda x: x["beh"], reverse=True)
                top_3 = [f"{idx}. {item['name']}" for idx, item in enumerate(by_beh[:3], 1)]
                response_txt = "**Top candidate(s) by leadership/behavioral signals:**<br/>" + "<br/>".join(top_3)
            elif "aws" in msg_lower or "missing" in msg_lower:
                matching = []
                for cd in candidate_details:
                    lacks_aws = any("aws" in gap for gap in cd["gaps"]) or "aws" in str(cd["gaps"])
                    if lacks_aws:
                        matching.append(f"- **{cd['name']}** (lacks AWS but has {cd['skills']}% Skills match)")
                if matching:
                    response_txt = "**Candidates lacking AWS but with strong Skills/LLM match:**<br/>" + "<br/>".join(matching)
                else:
                    response_txt = "All candidates in the current pool seem to have AWS skills, or no matching gaps were identified."
            elif "skills" in msg_lower or "strongest" in msg_lower:
                by_skills = sorted(candidate_details, key=lambda x: x["skills"], reverse=True)
                top_3 = [f"{idx}. {item['name']} (Skills: {item['skills']}%)" for idx, item in enumerate(by_skills[:3], 1)]
                response_txt = "**Top candidate(s) by Skills Fit:**<br/>" + "<br/>".join(top_3)
            else:
                response_txt = (
                    f"Based on the analysis, **{rows[0]['display_name']}** is the overall strongest match. "
                    f"Let me know if you would like a breakdown of skills, experience, or behavioral signals!"
                )

            st.session_state.chat_history.append({"role": "ai", "text": response_txt})
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
