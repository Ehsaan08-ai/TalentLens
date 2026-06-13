import html
from typing import Any
from icrs.ui.dashboard import JOB_TYPE_VALUES, decompose_via_api, extract_text_from_file

def render_job_creation(st: Any, base_url: str) -> None:
    st.markdown("<h1 class='premium-title'>Create New Job</h1>", unsafe_allow_html=True)
    st.caption(
        "Use our AI-powered engine to analyze your job description and generate precise candidate matches instantly."
    )
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([7, 5])

    with col1:
        st.subheader("1. Job Details")
        title = st.text_input("Job Title", value=st.session_state.job_title, placeholder="e.g. Lead AI Systems Engineer")
        st.session_state.job_title = title
        
        try:
            jt_index = JOB_TYPE_VALUES.index(st.session_state.job_type)
        except ValueError:
            jt_index = 0
        job_type = st.selectbox("Job Type", options=list(JOB_TYPE_VALUES), index=jt_index, key="jc_job_type")
        st.session_state.job_type = job_type

        jd_file = st.file_uploader("Upload a JD file (.docx / .pdf / .txt / .md)", type=["docx", "pdf", "txt", "md"], key="jc_jd_file")
        
        if jd_file is not None:
            file_key = f"{jd_file.name}_{jd_file.size}"
            if st.session_state.get("last_uploaded_jd_file_key") != file_key:
                st.session_state.last_uploaded_jd_file_key = file_key
                jd_from_file = extract_text_from_file(jd_file)
                st.session_state.raw_jd = jd_from_file
                st.session_state.jc_raw_jd = jd_from_file
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()
        else:
            if "last_uploaded_jd_file_key" in st.session_state:
                del st.session_state.last_uploaded_jd_file_key
        
        if "jc_raw_jd" not in st.session_state:
            st.session_state.jc_raw_jd = st.session_state.raw_jd
            
        raw_jd = st.text_area(
            "Job Description",
            height=250,
            placeholder="Paste the job description here.",
            key="jc_raw_jd"
        )
        st.session_state.raw_jd = raw_jd

        if st.button("Analyze Job", type="primary", use_container_width=True):
            if not raw_jd.strip():
                st.error("Please paste or upload a job description before analyzing.")
            else:
                with st.spinner("Analyzing Job Description..."):
                    try:
                        analysis = decompose_via_api(base_url, raw_jd)
                        st.session_state.analysis_results = analysis
                    except Exception as exc:
                        # Design: surface the actual error instead of fabricating
                        # mock data. The design's error-handling section explicitly
                        # says to show the error, not invent a fallback.
                        st.error(
                            f"Could not reach the backend analysis service. "
                            f"Please ensure the backend is running at **{base_url}** "
                            f"and try again.\n\nError details: `{exc}`"
                        )
                        return
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; color:#0b1c30;'>Job Insights</h3>", unsafe_allow_html=True)
        
        if st.session_state.analysis_results is None:
            st.info("Paste a job description and click **Analyze Job** to extract role intent, requirements, and behavioral signals.")
        else:
            results = st.session_state.analysis_results
            # Use the actual backend data without fallback defaults.
            # Missing keys render as honest "not available" rather than
            # fabricated values.
            role_intent = results.get("role_intent") or "Not available"
            must_haves = results.get("must_have") or []
            nice_to_haves = results.get("nice_to_have") or []
            behavioral_signals = results.get("behavioral_signals") or []

            escaped_role_intent = html.escape(role_intent)
            st.markdown(
                '<div class="insight-section">'
                '<div class="insight-title">Role Intent</div>'
                '<div class="role-intent-box">'
                f'"{escaped_role_intent}"'
                '</div></div>',
                unsafe_allow_html=True,
            )

            if must_haves:
                must_have_html = "".join(
                    f'<div class="badge-must">✓ {html.escape(m)}</div>' for m in must_haves
                )
            else:
                must_have_html = '<div style="color: #64748b; font-style: italic; font-size: 13px;">No must-have requirements extracted</div>'
            st.markdown(
                '<div class="insight-section">'
                '<div class="insight-title">Must Have requirements</div>'
                f'<div class="badge-container">{must_have_html}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            if nice_to_haves:
                nice_have_html = "".join(
                    f'<div class="badge-nice">✓ {html.escape(n)}</div>' for n in nice_to_haves
                )
            else:
                nice_have_html = '<div style="color: #64748b; font-style: italic; font-size: 13px;">No nice-to-have requirements extracted</div>'
            st.markdown(
                '<div class="insight-section">'
                '<div class="insight-title">Nice To Have requirements</div>'
                f'<div class="badge-container">{nice_have_html}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Behavioral signals: display as tags without fabricated percentages.
            # The backend decomposes behavioral signals as text labels; no
            # numeric scores are available, so we show them honestly as badges
            # rather than inventing progress-bar values.
            if behavioral_signals:
                behavioral_html = ""
                icon_map = {
                    "ownership": "rocket_launch",
                    "communication": "forum",
                    "startup": "bolt",
                    "mindset": "bolt",
                    "leadership": "groups",
                    "collaboration": "handshake",
                    "initiative": "trending_up",
                }
                for signal in behavioral_signals:
                    # Pick an icon based on keyword matching, default to "bolt"
                    icon = "bolt"
                    signal_lower = signal.lower()
                    for keyword, icon_name in icon_map.items():
                        if keyword in signal_lower:
                            icon = icon_name
                            break
                    escaped_signal = html.escape(signal)
                    behavioral_html += (
                        '<div class="behavioral-row">'
                        '<div class="behavioral-label">'
                        '<div class="behavior-icon">'
                        f'<span class="material-symbols-outlined" style="font-size:16px;">{icon}</span>'
                        '</div>'
                        f'<span>{escaped_signal}</span>'
                        '</div></div>'
                    )
            else:
                behavioral_html = '<div style="color: #64748b; font-style: italic; font-size: 13px;">No behavioral signals extracted</div>'

            st.markdown(
                '<div class="insight-section">'
                '<div class="insight-title">Behavioral Signals</div>'
                f'<div style="display: flex; flex-direction: column;">{behavioral_html}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.session_state.analysis_results is not None:
            if st.button("Publish & Find Candidates", type="primary", use_container_width=True):
                st.session_state.navigation_page = "Candidate Ranking & Match"
                st.session_state._navigation_source = "button"
                st.rerun()
