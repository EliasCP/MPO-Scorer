from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


HUMAN_LIVER_BLOOD_FLOW_ML_MIN_KG = 20.7
INT_TEST_DILUTION_FACTOR = 10.0
DEFAULT_DOSE_MG_KG = 3.0
DEFAULT_TAU_H = 24.0
DEFAULT_INT_SURFACE_AREA_CM2 = 800.0
DEFAULT_INT_TRANSIT_H = 3.3
DEFAULT_INT_RADIUS_CM = 1.75
DEFAULT_FG = 1.0
DEFAULT_E_MAX_LOWER = 50.0
DEFAULT_E_MAX_UPPER = 90.0
DEFAULT_PAPP_TO_PEFF_SCALE = 26.3
DEFAULT_FA_SATURATION_MG_ML = 0.05


REQUIRED_COLUMNS = [
    "compound_id",
    "ec50_app_nM",
    "clint_app_mL_min_kg",
    "mdck_papp_cm_s",
    "solubility_pH7_4_uM",
    "molecular_weight_g_mol",
]

OPTIONAL_COLUMNS = [
    "fup",
    "fumic",
    "pka",
    "logD7_4",
    "emax_pct",
    "hERG_ic50_nM",
    "dose_mg_kg",
    "tau_h",
    "driver",
]


@dataclass
class ScoreSettings:
    dose_mg_kg: float = DEFAULT_DOSE_MG_KG
    tau_h: float = DEFAULT_TAU_H
    driver: str = "Cavg"
    emax_lower: float = DEFAULT_E_MAX_LOWER
    emax_upper: float = DEFAULT_E_MAX_UPPER
    use_herg: bool = True
    q_ml_min_kg: float = HUMAN_LIVER_BLOOD_FLOW_ML_MIN_KG
    dilution_factor: float = INT_TEST_DILUTION_FACTOR


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or value == ""


def coerce_numeric(value: Any) -> Optional[float]:
    if is_missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def calc_fumedia_from_fup(fup: float, dilution_factor: float = INT_TEST_DILUTION_FACTOR) -> float:
    fup = clamp(float(fup), 1e-6, 1.0 - 1e-6)
    return 1.0 / (1.0 + dilution_factor * (1.0 / fup - 1.0))


def estimate_fup_from_fumic(fumic: float) -> float:
    # Fallback only; used when fup is not available.
    # This is intentionally conservative and should be replaced by assay data when possible.
    fumic = clamp(float(fumic), 1e-6, 1.0 - 1e-6)
    return clamp(0.5 * fumic + 0.25, 1e-3, 0.99)


def estimate_fumic_from_fup(fup: float) -> float:
    # Fallback only; used when fumic is not available.
    fup = clamp(float(fup), 1e-6, 1.0 - 1e-6)
    return clamp(0.8 * fup + 0.15, 1e-3, 0.99)


def calc_mec_nM(ec50_app_nM: float, fup: float, fumedia: float) -> float:
    fup = clamp(float(fup), 1e-6, 1.0 - 1e-6)
    fumedia = clamp(float(fumedia), 1e-6, 1.0 - 1e-6)
    return float(ec50_app_nM) * fumedia / fup


def emax_factor(emax_pct: Optional[float], lower: float = DEFAULT_E_MAX_LOWER, upper: float = DEFAULT_E_MAX_UPPER) -> float:
    if emax_pct is None or np.isnan(emax_pct):
        return 1.0
    lower, upper = float(lower), float(upper)
    if upper <= lower:
        upper = lower + 1.0
    emax_pct = float(emax_pct)
    if emax_pct >= upper:
        return 1.0
    if emax_pct <= lower:
        return 0.0
    return (emax_pct - lower) / (upper - lower)


def calc_clint_in_vivo_mL_min_kg(clint_app_mL_min_kg: float, fumic: float) -> float:
    fumic = clamp(float(fumic), 1e-6, 1.0)
    return float(clint_app_mL_min_kg) / fumic


def calc_hepatic_cl_mL_min_kg(clint_in_vivo_mL_min_kg: float, fup: float, q_ml_min_kg: float = HUMAN_LIVER_BLOOD_FLOW_ML_MIN_KG) -> float:
    fup = clamp(float(fup), 1e-6, 1.0 - 1e-6)
    q = float(q_ml_min_kg)
    clint = float(clint_in_vivo_mL_min_kg)
    return q * fup * clint / (q + fup * clint)


def calc_fh(cl_ml_min_kg: float, q_ml_min_kg: float = HUMAN_LIVER_BLOOD_FLOW_ML_MIN_KG) -> float:
    q = float(q_ml_min_kg)
    cl = clamp(float(cl_ml_min_kg), 0.0, q * 0.999999)
    return clamp(1.0 - cl / q, 1e-6, 1.0)


def mdck_to_peff_cm_s(mdck_papp_cm_s: float, scale: float = DEFAULT_PAPP_TO_PEFF_SCALE) -> float:
    return max(0.0, float(mdck_papp_cm_s) * scale)


def calc_ka_h_inv(peff_cm_s: float, radius_cm: float = DEFAULT_INT_RADIUS_CM) -> float:
    # Convert cm/s to cm/h, then apply the Sinko-style approximation.
    peff_cm_h = float(peff_cm_s) * 3600.0
    ka = 2.0 * peff_cm_h / float(radius_cm)
    return clamp(ka, 0.2, 1.0)


def estimate_fa(
    mdck_papp_cm_s: float,
    solubility_pH7_4_uM: float,
    molecular_weight_g_mol: float,
    peff_scale: float = DEFAULT_PAPP_TO_PEFF_SCALE,
    solubility_saturation_mg_mL: float = DEFAULT_FA_SATURATION_MG_ML,
) -> Tuple[float, Dict[str, float]]:
    peff_cm_s = mdck_to_peff_cm_s(mdck_papp_cm_s, scale=peff_scale)
    ka = calc_ka_h_inv(peff_cm_s)

    sol_mg_mL = max(0.0, float(solubility_pH7_4_uM) * float(molecular_weight_g_mol) / 1e6)
    permeability_component = 1.0 - math.exp(-ka * DEFAULT_INT_TRANSIT_H)
    solubility_component = sol_mg_mL / (sol_mg_mL + solubility_saturation_mg_mL)
    fa = clamp(permeability_component * solubility_component, 0.0, 1.0)
    return fa, {
        "peff_cm_s": peff_cm_s,
        "ka_h_inv": ka,
        "solubility_mg_mL": sol_mg_mL,
        "permeability_component": permeability_component,
        "solubility_component": solubility_component,
    }


def estimate_vdss_l_kg(
    fumic: float,
    fup: float,
    pka: Optional[float] = None,
    logD7_4: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    # A pragmatic, rank-ordering proxy inspired by the manuscript's mechanistic spirit.
    # It is designed to be monotonic with microsomal binding and lipophilicity.
    fumic = clamp(float(fumic), 1e-6, 1.0 - 1e-6)
    fup = clamp(float(fup), 1e-6, 1.0 - 1e-6)

    lipophilicity = 2.0 if logD7_4 is None or np.isnan(logD7_4) else float(logD7_4)
    pka_term = 0.0 if pka is None or np.isnan(pka) else sigmoid((float(pka) - 7.4) / 1.5) - 0.5
    binding_term = math.log10(1.0 / fumic)
    free_fraction_term = math.log10(1.0 / fup)

    raw = 0.65 * binding_term + 0.25 * lipophilicity + 0.10 * free_fraction_term + 0.15 * pka_term
    tissue_multiplier = 0.35 + 3.5 * sigmoid(raw - 0.25)
    vdss = clamp(0.04 + tissue_multiplier, 0.05, 8.0)
    return vdss, {
        "binding_term": binding_term,
        "free_fraction_term": free_fraction_term,
        "pka_term": pka_term,
        "lipophilicity": lipophilicity,
        "raw_vdss_score": raw,
    }


def calc_pk_profile(
    dose_mg_kg: float,
    tau_h: float,
    molecular_weight_g_mol: float,
    cl_ml_min_kg: float,
    vdss_l_kg: float,
    fa: float,
    fh: float,
    fg: float = DEFAULT_FG,
    ka_h_inv: float = 1.0,
) -> Dict[str, float]:
    dose_mg_kg = float(dose_mg_kg)
    tau_h = float(tau_h)
    mw = float(molecular_weight_g_mol)
    cl_l_h_kg = float(cl_ml_min_kg) / 1000.0 * 60.0
    vd = max(1e-9, float(vdss_l_kg))
    ke = clamp(cl_l_h_kg / vd, 1e-6, 10.0)
    ka = clamp(float(ka_h_inv), 0.2, 1.0)
    f = clamp(float(fa) * float(fh) * float(fg), 0.0, 1.0)

    # Convert dose to mol/kg for concentration in mol/L.
    dose_mol_kg = dose_mg_kg / 1000.0 / mw
    factor = f * dose_mol_kg / vd * ka / max(1e-9, abs(ka - ke))
    denom = max(1e-9, 1.0 - math.exp(-ke * tau_h))
    tmax = math.log(ka / ke) / (ka - ke) if abs(ka - ke) > 1e-6 else tau_h / 2.0
    tmax = clamp(tmax, 0.0, tau_h)

    def c_ss(t: float) -> float:
        if abs(ka - ke) < 1e-6:
            # Limit as ka approaches ke.
            return factor * (t * math.exp(-ke * t)) / denom
        return factor * (math.exp(-ke * t) - math.exp(-ka * t)) / denom

    cmax_mol_L = c_ss(tmax)
    cmin_mol_L = c_ss(tau_h)
    auc_mol_h_L = f * dose_mol_kg / cl_l_h_kg
    cavg_mol_L = auc_mol_h_L / tau_h

    return {
        "cl_l_h_kg": cl_l_h_kg,
        "ke_h_inv": ke,
        "ka_h_inv": ka,
        "tmax_h": tmax,
        "cmax_nM": cmax_mol_L * 1e9,
        "cmin_nM": cmin_mol_L * 1e9,
        "cavg_nM": cavg_mol_L * 1e9,
        "auc_mol_h_L": auc_mol_h_L,
        "f_total": f,
    }


def calc_scores_for_row(row: pd.Series, settings: ScoreSettings) -> Dict[str, Any]:
    warnings: List[str] = []

    compound_id = row.get("compound_id", "")
    ec50 = coerce_numeric(row.get("ec50_app_nM"))
    clint_app = coerce_numeric(row.get("clint_app_mL_min_kg"))
    mdck_papp = coerce_numeric(row.get("mdck_papp_cm_s"))
    solubility = coerce_numeric(row.get("solubility_pH7_4_uM"))
    mw = coerce_numeric(row.get("molecular_weight_g_mol"))
    fup = coerce_numeric(row.get("fup"))
    fumic = coerce_numeric(row.get("fumic"))
    pka = coerce_numeric(row.get("pka"))
    logd = coerce_numeric(row.get("logD7_4"))
    emax_pct = coerce_numeric(row.get("emax_pct"))
    herg_ic50 = coerce_numeric(row.get("hERG_ic50_nM"))
    dose = coerce_numeric(row.get("dose_mg_kg")) or settings.dose_mg_kg
    tau = coerce_numeric(row.get("tau_h")) or settings.tau_h
    row_driver = str(row.get("driver", settings.driver) or settings.driver)

    missing_required = [name for name, val in [
        ("compound_id", compound_id),
        ("ec50_app_nM", ec50),
        ("clint_app_mL_min_kg", clint_app),
        ("mdck_papp_cm_s", mdck_papp),
        ("solubility_pH7_4_uM", solubility),
        ("molecular_weight_g_mol", mw),
    ] if is_missing(val)]
    if missing_required:
        return {
            "compound_id": compound_id,
            "status": "missing_required",
            "warnings": "; ".join([f"Missing required: {', '.join(missing_required)}"]),
        }

    if fup is None and fumic is None:
        return {
            "compound_id": compound_id,
            "status": "missing_required",
            "warnings": "Need at least one of fup or fumic.",
        }
    if fup is None:
        fup = estimate_fup_from_fumic(fumic)
        warnings.append("fup missing; estimated from fumic using a fallback.")
    if fumic is None:
        fumic = estimate_fumic_from_fup(fup)
        warnings.append("fumic missing; estimated from fup using a fallback.")

    fumedia = calc_fumedia_from_fup(fup, dilution_factor=settings.dilution_factor)
    mec_nM = calc_mec_nM(ec50, fup, fumedia)
    emax_fac = emax_factor(emax_pct, settings.emax_lower, settings.emax_upper)

    clint_in_vivo = calc_clint_in_vivo_mL_min_kg(clint_app, fumic)
    cl = calc_hepatic_cl_mL_min_kg(clint_in_vivo, fup, settings.q_ml_min_kg)
    fh = calc_fh(cl, settings.q_ml_min_kg)

    fa, fa_details = estimate_fa(mdck_papp, solubility, mw)
    vdss, vd_details = estimate_vdss_l_kg(fumic, fup, pka=pka, logD7_4=logd)

    profile = calc_pk_profile(
        dose_mg_kg=dose,
        tau_h=tau,
        molecular_weight_g_mol=mw,
        cl_ml_min_kg=cl,
        vdss_l_kg=vdss,
        fa=fa,
        fh=fh,
        fg=DEFAULT_FG,
        ka_h_inv=fa_details["ka_h_inv"],
    )

    cmax_score = (profile["cmax_nM"] / mec_nM) * emax_fac if mec_nM > 0 else np.nan
    cmin_score = (profile["cmin_nM"] / mec_nM) * emax_fac if mec_nM > 0 else np.nan
    cavg_score = (profile["cavg_nM"] / mec_nM) * emax_fac if mec_nM > 0 else np.nan

    herg_score = np.nan
    ti_score = np.nan
    if settings.use_herg and herg_ic50 is not None and herg_ic50 > 0:
        # Simple surrogate aligned with the manuscript concept.
        herg_score = herg_ic50 / max(1e-9, fup * profile["cmax_nM"])
        ti_score = herg_score * cmax_score if np.isfinite(cmax_score) else np.nan

    primary_map = {
        "Cmax": cmax_score,
        "Cmin": cmin_score,
        "Cavg": cavg_score,
    }
    primary_score = primary_map.get(row_driver, cavg_score)

    if row_driver not in primary_map:
        warnings.append(f"driver='{row_driver}' not recognized; using Cavg.")

    return {
        "compound_id": compound_id,
        "driver": row_driver,
        "status": "ok",
        "warnings": "; ".join(warnings),
        "dose_mg_kg": dose,
        "tau_h": tau,
        "ec50_app_nM": ec50,
        "emax_pct": emax_pct,
        "emax_factor": emax_fac,
        "fup": fup,
        "fumic": fumic,
        "fumedia": fumedia,
        "clint_app_mL_min_kg": clint_app,
        "clint_in_vivo_mL_min_kg": clint_in_vivo,
        "cl_pred_mL_min_kg": cl,
        "fh_pred": fh,
        "mdck_papp_cm_s": mdck_papp,
        "peff_pred_cm_s": fa_details["peff_cm_s"],
        "ka_pred_h_inv": profile["ka_h_inv"],
        "fa_pred": fa,
        "vdss_pred_L_kg": vdss,
        "ke_pred_h_inv": profile["ke_h_inv"],
        "tmax_h": profile["tmax_h"],
        "cmax_pred_nM": profile["cmax_nM"],
        "cmin_pred_nM": profile["cmin_nM"],
        "cavg_pred_nM": profile["cavg_nM"],
        "mec_pred_nM": mec_nM,
        "cmax_score": cmax_score,
        "cmin_score": cmin_score,
        "cavg_score": cavg_score,
        "primary_score": primary_score,
        "hERG_ic50_nM": herg_ic50,
        "hERG_score": herg_score,
        "ti_score": ti_score,
    }


def validate_input_dataframe(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {', '.join(missing)}")
    return issues


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def score_dataframe(df: pd.DataFrame, settings: ScoreSettings) -> pd.DataFrame:
    df = prepare_dataframe(df)
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(calc_scores_for_row(row, settings))
    out = pd.DataFrame(rows)
    if not out.empty and "primary_score" in out.columns:
        out = out.sort_values(by=["primary_score", "compound_id"], ascending=[False, True], na_position="last")
    return out


def make_template_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "compound_id": "Cmpd_001",
                "ec50_app_nM": 12.5,
                "clint_app_mL_min_kg": 18.2,
                "fup": 0.08,
                "fumic": 0.65,
                "mdck_papp_cm_s": 2.1e-6,
                "solubility_pH7_4_uM": 24.0,
                "molecular_weight_g_mol": 500.2,
                "pka": 7.4,
                "logD7_4": 2.8,
                "emax_pct": 100,
                "hERG_ic50_nM": 1200,
                "dose_mg_kg": DEFAULT_DOSE_MG_KG,
                "tau_h": DEFAULT_TAU_H,
                "driver": "Cavg",
            },
            {
                "compound_id": "Cmpd_002",
                "ec50_app_nM": 45.0,
                "clint_app_mL_min_kg": 8.9,
                "fup": 0.15,
                "fumic": 0.72,
                "mdck_papp_cm_s": 7.5e-6,
                "solubility_pH7_4_uM": 10.0,
                "molecular_weight_g_mol": 430.1,
                "pka": 8.1,
                "logD7_4": 3.4,
                "emax_pct": 85,
                "hERG_ic50_nM": 2000,
                "dose_mg_kg": DEFAULT_DOSE_MG_KG,
                "tau_h": DEFAULT_TAU_H,
                "driver": "Cmin",
            },
            {
                "compound_id": "Cmpd_003",
                "ec50_app_nM": 3.2,
                "clint_app_mL_min_kg": 31.0,
                "fup": 0.03,
                "fumic": 0.55,
                "mdck_papp_cm_s": 1.2e-5,
                "solubility_pH7_4_uM": 5.0,
                "molecular_weight_g_mol": 610.8,
                "pka": 6.7,
                "logD7_4": 1.9,
                "emax_pct": 100,
                "hERG_ic50_nM": 950,
                "dose_mg_kg": DEFAULT_DOSE_MG_KG,
                "tau_h": DEFAULT_TAU_H,
                "driver": "Cmax",
            },
        ]
    )
