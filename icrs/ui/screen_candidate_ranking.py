from typing import Any
from icrs.ui.dashboard import (
    JOB_TYPE_VALUES,
    CandidatePoolError,
    extract_text_from_file,
    parse_candidate_pool,
    parse_candidate_pool_documents,
    parse_source_records,
    parse_source_record_documents,
    build_rank_payload,
    rank_via_api,
    transform_response_to_rows,
    _render_results,
)

def render_candidate_ranking(
    st: Any, 
    base_url: str, 
    timeout_seconds: float,
    rank_client: Any = None
) -> None:
    st.markdown("<h1 class='premium-title'>Candidate Ranking</h1>", unsafe_allow_html=True)
    st.caption(
        "Rank candidate pools against the active job description using high-dimensional semantic match and behavior profiles."
    )
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # --- Inputs: job description --------------------------------------------
    st.subheader("1. Job description")
    jd_file = st.file_uploader("Upload a JD file (.docx / .pdf / .txt / .md)", type=["docx", "pdf", "txt", "md"], key="jd_file")
    
    if jd_file is not None:
        file_key = f"{jd_file.name}_{jd_file.size}"
        if st.session_state.get("last_uploaded_jd_file_key_ranking") != file_key:
            st.session_state.last_uploaded_jd_file_key_ranking = file_key
            jd_from_file = extract_text_from_file(jd_file)
            st.session_state.raw_jd = jd_from_file
            st.session_state.ranking_raw_jd = jd_from_file
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
    else:
        if "last_uploaded_jd_file_key_ranking" in st.session_state:
            del st.session_state.last_uploaded_jd_file_key_ranking
    
    if "ranking_raw_jd" not in st.session_state:
        st.session_state.ranking_raw_jd = st.session_state.raw_jd
        
    raw_jd = st.text_area(
        "...or paste the JD text",
        height=200,
        placeholder="Paste the job description here.",
        key="ranking_raw_jd"
    )
    st.session_state.raw_jd = raw_jd
    
    title = st.text_input("Job title (optional)", value=st.session_state.job_title, key="ranking_title")
    st.session_state.job_title = title
    
    try:
        jt_index = JOB_TYPE_VALUES.index(st.session_state.job_type)
    except ValueError:
        jt_index = 0
    job_type = st.selectbox("Job type", options=list(JOB_TYPE_VALUES), index=jt_index, key="ranking_job_type")
    st.session_state.job_type = job_type

    # --- Inputs: candidate pool ---------------------------------------------
    st.subheader("2. Candidate pool")
    st.caption(
        "Provide a candidate pool as a JSON array, JSON Lines (JSONL), or CSV, "
        "supporting standard or Redrob schemas."
    )
    pool_files = st.file_uploader(
        "Upload candidate pool files or a folder (.json, .jsonl, .csv)",
        type=["json", "jsonl", "csv"],
        key="pool_file",
        accept_multiple_files=True,
    )
    uploaded_documents = []
    if pool_files:
        uploaded_documents = [
            (uploaded_file.name, uploaded_file.getvalue().decode("utf-8", errors="replace"))
            for uploaded_file in pool_files
        ]
    pool_from_file = ""
    if len(uploaded_documents) == 1:
        pool_from_file = uploaded_documents[0][1]
    pool_text = st.text_area(
        "...or paste candidate-pool (JSON, JSONL, or CSV)",
        value=pool_from_file,
        height=200,
        placeholder=(
            '[{"structured_fields": {...}, "free_text": "...", "external_handles": {}}]\n'
            'or JSONL lines, or CSV rows'
        ),
    )

    # --- Trigger ranking -----------------------------------------------------
    if st.button("Rank candidates", type="primary"):
        try:
            if uploaded_documents:
                candidates = parse_candidate_pool_documents(uploaded_documents)
            else:
                candidates = parse_candidate_pool(pool_text)
        except CandidatePoolError as exc:
            st.error(str(exc))
            return

        # Generate deterministic UUIDs and names in UI scope only
        import uuid
        raw_records = []
        if uploaded_documents:
            raw_records = parse_source_record_documents(uploaded_documents)
        else:
            try:
                raw_records = parse_source_records(pool_text)
            except Exception:
                raw_records = []

        uuid_to_name = {}
        for index, c in enumerate(candidates):
            display_name = ""
            if index < len(raw_records):
                item = raw_records[index]
                if isinstance(item, dict):
                    profile_field = item.get("profile")
                    profile_dict = profile_field if isinstance(profile_field, dict) else {}
                    sf_field = item.get("structured_fields")
                    sf_dict = sf_field if isinstance(sf_field, dict) else {}
                    
                    display_name = (
                        profile_dict.get("anonymized_name") or
                        item.get("name") or
                        item.get("candidate_name") or
                        sf_dict.get("display_name") or
                        sf_dict.get("source_candidate_id") or
                        item.get("candidate_id") or
                        item.get("id") or
                        ""
                    )
            display_name = str(display_name).strip()
            if not display_name:
                display_name = f"Candidate #{index + 1}"
            
            client_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{index}_{display_name}"))
            uuid_to_name[client_uuid] = display_name
            c.setdefault("structured_fields", {})["client_assigned_id"] = client_uuid

        try:
            payload = build_rank_payload(raw_jd, job_type, candidates, title=title)
        except CandidatePoolError as exc:
            st.error(str(exc))
            return

        with st.spinner(f"Ranking {len(candidates)} candidate(s)..."):
            try:
                if rank_client is None:
                    response = rank_via_api(
                        base_url, payload, timeout=float(timeout_seconds)
                    )
                else:
                    response = rank_client(base_url, payload)
            except Exception as exc:  # noqa: BLE001
                st.error(
                    f"Could not reach the ranking backend at {base_url} "
                    f"within {int(timeout_seconds)}s. {exc}"
                )
                return
        st.session_state.ranking_response = response
        st.session_state.uuid_to_name = uuid_to_name
        # Set default selected candidate using the raw UUID (not display name)
        # so the explainability panel can look it up unambiguously (Bug 3).
        rows = transform_response_to_rows(response, uuid_to_name)
        if rows:
            st.session_state.selected_candidate_id = rows[0]["raw_candidate_id"]
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    if st.session_state.get("ranking_response") is not None:
        _render_results(st, st.session_state.ranking_response, uuid_to_name=st.session_state.get("uuid_to_name"))
