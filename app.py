from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

from scoring import (
    ScoreSettings,
    make_template_dataframe,
    score_dataframe,
    validate_input_dataframe,
)


st.set_page_config(
    page_title="Mechanistic PK MPO Scorer",
    page_icon="🧪",
    layout="wide",
)

st.title("Mechanistic PK MPO Scorer")
st.caption(
    "Batch-score compounds using a mechanistic PK-inspired workflow based on the manuscript's MPO framework."
)

with st.sidebar:
    st.header("Project settings")
    driver = st.selectbox("Primary PK/PD driver", ["Cavg", "Cmin", "Cmax"], index=0)
    dose_mg_kg = st.number_input("Default dose (mg/kg)", min_value=0.001, value=3.0, step=0.5)
    tau_h = st.number_input("Dosing interval tau (h)", min_value=0.5, value=24.0, step=1.0)
    emax_lower = st.number_input("Emax lower bound (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    emax_upper = st.number_input("Emax upper bound (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
    use_herg = st.checkbox("Calculate TI score when hERG IC50 is available", value=True)

settings = ScoreSettings(
    dose_mg_kg=dose_mg_kg,
    tau_h=tau_h,
    driver=driver,
    emax_lower=emax_lower,
    emax_upper=emax_upper,
    use_herg=use_herg,
)

tab_single, tab_batch, tab_help = st.tabs(["Single compound", "Batch CSV", "Column guide"])

with tab_single:
    st.subheader("Score one compound")
    c1, c2, c3 = st.columns(3)
    with c1:
        compound_id = st.text_input("compound_id", value="Cmpd_001")
        ec50_app_nM = st.number_input("ec50_app_nM", min_value=0.0, value=12.5, step=0.5)
        clint_app = st.number_input("clint_app_mL_min_kg", min_value=0.0, value=18.2, step=0.5)
        fup = st.number_input("fup", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
    with c2:
        fumic = st.number_input("fumic", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
        mdck_papp = st.number_input("mdck_papp_cm_s", min_value=0.0, value=2.1e-6, format="%.2e")
        solubility = st.number_input("solubility_pH7_4_uM", min_value=0.0, value=24.0, step=1.0)
    with c3:
        mw = st.number_input("molecular_weight_g_mol", min_value=1.0, value=500.2, step=1.0)
        pka = st.number_input("pka (optional)", value=7.4, step=0.1)
        logd = st.number_input("logD7_4 (optional)", value=2.8, step=0.1)
        emax_pct = st.number_input("emax_pct", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
    herg_ic50 = st.number_input("hERG_ic50_nM (optional)", min_value=0.0, value=1200.0, step=10.0)

    if st.button("Score compound", type="primary"):
        row = pd.Series(
            {
                "compound_id": compound_id,
                "ec50_app_nM": ec50_app_nM,
                "clint_app_mL_min_kg": clint_app,
                "fup": fup,
                "fumic": fumic,
                "mdck_papp_cm_s": mdck_papp,
                "solubility_pH7_4_uM": solubility,
                "molecular_weight_g_mol": mw,
                "pka": pka,
                "logD7_4": logd,
                "emax_pct": emax_pct,
                "hERG_ic50_nM": herg_ic50,
                "dose_mg_kg": dose_mg_kg,
                "tau_h": tau_h,
                "driver": driver,
            }
        )
        from scoring import calc_scores_for_row

        result = calc_scores_for_row(row, settings)
        result_df = pd.DataFrame([result])
        st.success("Scoring complete")
        st.dataframe(result_df.T, use_container_width=True)
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download single-compound result CSV",
            data=csv_bytes,
            file_name=f"{compound_id}_score.csv",
            mime="text/csv",
        )

with tab_batch:
    st.subheader("Score multiple compounds from a CSV file")
    st.write(
        "Upload a CSV with one row per compound. The app will display the first 10 scored compounds and provide a downloadable results file for the full batch."
    )

    template_df = make_template_dataframe()
    template_csv = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download template CSV",
        data=template_csv,
        file_name="compound_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload compound CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        issues = validate_input_dataframe(df)
        if issues:
            st.error("\n".join(issues))
        else:
            scored = score_dataframe(df, settings)
            st.success(f"Scored {len(scored)} compounds")

            st.markdown("#### Preview (first 10 compounds)")
            preview_cols = [
                "compound_id",
                "primary_score",
                "cmax_score",
                "cmin_score",
                "cavg_score",
                "ti_score",
                "cmax_pred_nM",
                "cmin_pred_nM",
                "cavg_pred_nM",
                "mec_pred_nM",
                "warnings",
            ]
            available_cols = [c for c in preview_cols if c in scored.columns]
            st.dataframe(scored[available_cols].head(10), use_container_width=True)

            out = scored.copy()
            csv_out = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full scored results CSV",
                data=csv_out,
                file_name="mpo_scored_results.csv",
                mime="text/csv",
            )

            st.markdown("#### Ranking summary")
            rank_col = {
                "Cavg": "cavg_score",
                "Cmin": "cmin_score",
                "Cmax": "cmax_score",
            }[driver]
            top = scored.sort_values(rank_col, ascending=False).head(10)
            st.write(f"Top 10 by {rank_col}")
            st.dataframe(top[[c for c in ["compound_id", rank_col, "primary_score", "warnings"] if c in top.columns]], use_container_width=True)

with tab_help:
    st.subheader("Template / column guide")
    st.write("Required columns:")
    st.code(
        "compound_id,ec50_app_nM,clint_app_mL_min_kg,mdck_papp_cm_s,solubility_pH7_4_uM,molecular_weight_g_mol",
        language="text",
    )
    st.write("Optional columns:")
    st.code(
        "fup,fumic,pka,logD7_4,emax_pct,hERG_ic50_nM,dose_mg_kg,tau_h,driver",
        language="text",
    )
    st.info(
        "The app will compute Cmax,score, Cmin,score, Cavg,score for every compound, and uses the selected driver to set the primary ranking column."
    )
    st.markdown(
        """
        **Notes**
        - If only one of `fup` or `fumic` is supplied, the app uses a conservative fallback estimate for the missing value.
        - The PK model is intended for ranking and comparison, not for direct clinical use.
        - You can swap in your own validated submodels later without changing the Streamlit interface.
        """
    )
