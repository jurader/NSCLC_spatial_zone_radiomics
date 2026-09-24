#!/usr/bin/env python3
"""Strict reanalysis using Le et al. supplementary-table original radiomics.

The common LUNG1/LUNG2 cohorts are fixed before feature selection and model
development.  Imputation and standardization for Elastic-Net CV are fitted
inside each training fold only.  This script does not overwrite earlier
results.
"""
from pathlib import Path
import os, re, warnings, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter

MAX_FEATURES = int(os.environ.get("MAX_FEATURES", "20"))
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("REANALYSIS_DATA", str(PACKAGE_ROOT / "data")))
OUT = Path(os.environ.get("STRICT_OUT", str(PACKAGE_ROOT / "results" / f"max{MAX_FEATURES}")))
OUT.mkdir(parents=True, exist_ok=True)
CLIN = DATA_ROOT / "clinical_outcome"
ZROOT = DATA_ROOT / "spatial_features"
SEED = 20260830


def load_outcomes():
    files = {
        "LUNG1_training": "Supplementary-Table-S1-Training-set-LUNG1-Radiomics.csv",
        "LUNG1_testing": "Supplementary-Table-S2-Testing-set-LUNG1-Radiomics.csv",
        "LUNG2_validation": "Supplementary-Table-S3-Validation-set-LUNG2-Radiogenomics.csv",
    }
    out = {}
    for name, filename in files.items():
        d = pd.read_csv(CLIN / filename)
        id_col = "ID" if "ID" in d else "PatientID"
        d = d.rename(columns={id_col: "PatientID", "Survival.time": "time",
                              "deadstatus.event": "event", "gender": "sex"})
        d["PatientID"] = d["PatientID"].astype(str)
        d["age"] = pd.to_numeric(d["age"], errors="coerce")
        d["male"] = d["sex"].astype(str).str.lower().map(
            {"male": 1, "female": 0, "m": 1, "f": 0}
        )
        out[name] = d[["PatientID", "age", "male", "time", "event"]]
    age_median = float(out["LUNG1_training"]["age"].median())
    for name in out:
        out[name] = out[name].copy()
        out[name]["age"] = out[name]["age"].fillna(age_median)
    return out, age_median


def canon(value):
    value = value.replace("ori-", "original_").replace("ori_", "original_")
    return re.sub(r"[^a-z0-9]", "", value.lower())


def baseline_features():
    frames = {}
    for name, filename in [
        ("LUNG1_training", "Supplementary-Table-S1-Training-set-LUNG1-Radiomics.csv"),
        ("LUNG1_testing", "Supplementary-Table-S2-Testing-set-LUNG1-Radiomics.csv"),
    ]:
        d = pd.read_csv(CLIN / filename)
        id_col = "ID" if "ID" in d else "PatientID"
        cols = d.columns[10:]
        frames[name] = pd.DataFrame(
            {"PatientID": d[id_col].astype(str),
             **{f"base__{canon(c)}": d[c] for c in cols}}
        )
    d = pd.read_csv(
        CLIN / "Supplementary-Table-S3-Validation-set-LUNG2-Radiogenomics.csv"
    )
    id_col = "ID" if "ID" in d else "PatientID"
    cols = d.columns[10:]
    frames["LUNG2_validation"] = pd.DataFrame(
        {"PatientID": d[id_col].astype(str),
         **{f"base__{canon(c)}": d[c] for c in cols}}
    )
    common = set.intersection(*(set(frame.columns[1:]) for frame in frames.values()))
    return {k: frame[["PatientID"] + sorted(common)] for k, frame in frames.items()}


def survival_target(frame):
    return np.array(
        list(zip(frame["event"].astype(bool), frame["time"].astype(float))),
        dtype=[("event", "?"), ("time", "<f8")],
    )


def c_index(frame, risk):
    return float(concordance_index_censored(
        frame["event"].astype(bool).to_numpy(),
        frame["time"].astype(float).to_numpy(),
        np.asarray(risk),
    )[0])


def transform_fold(x_train, x_valid):
    imputer = SimpleImputer(strategy="median").fit(x_train)
    scaler = StandardScaler().fit(imputer.transform(x_train))
    train_z = np.clip(scaler.transform(imputer.transform(x_train)), -10.0, 10.0)
    valid_z = np.clip(scaler.transform(imputer.transform(x_valid)), -10.0, 10.0)
    return train_z, valid_z


def select_enet(train, columns):
    raw = train[columns].to_numpy(float)
    target = survival_target(train)
    event = train["event"].astype(int).to_numpy()
    records = []
    alphas = np.logspace(0, -2, 12)
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for l1_ratio in (0.1, 0.5, 0.9, 1.0):
        for train_idx, valid_idx in splitter.split(raw, event):
            x_train, x_valid = transform_fold(raw[train_idx], raw[valid_idx])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = CoxnetSurvivalAnalysis(
                    l1_ratio=l1_ratio, alphas=alphas,
                    max_iter=100000, tol=1e-7,
                ).fit(x_train, target[train_idx])
            for alpha in model.alphas_:
                records.append({
                    "l1_ratio": l1_ratio,
                    "alpha": float(alpha),
                    "c_index": c_index(train.iloc[valid_idx], model.predict(x_valid, alpha=float(alpha))),
                })
    tuning = pd.DataFrame(records)
    means = (tuning.groupby(["l1_ratio", "alpha"], as_index=False)["c_index"]
             .mean().sort_values("c_index", ascending=False))
    best = means.iloc[0]
    l1_ratio, alpha = float(best["l1_ratio"]), float(best["alpha"])

    # Final preprocessing is fit on the fixed common LUNG1 development cohort.
    imputer = SimpleImputer(strategy="median").fit(train[columns])
    scaler = StandardScaler().fit(imputer.transform(train[columns]))
    x = np.clip(scaler.transform(imputer.transform(train[columns])), -10.0, 10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio, alphas=[alpha],
            max_iter=100000, tol=1e-7,
        ).fit(x, target)
    beta = final.coef_.ravel()
    nonzero = [c for c, b in zip(columns, beta) if abs(b) > 1e-8]
    if not nonzero:
        nonzero = list(columns[: min(10, len(columns))])
    chosen = sorted(nonzero, key=lambda c: -abs(beta[columns.index(c)]))[:MAX_FEATURES]
    return chosen, {c: float(beta[columns.index(c)]) for c in chosen}, tuning, {
        "l1_ratio": l1_ratio, "alpha": alpha, "n_nonzero": len(nonzero),
    }


def main():
    outcomes, age_median = load_outcomes()
    baseline = baseline_features()
    manifest = pd.read_csv(
        DATA_ROOT / "common_cohort_manifest/aligned_case_manifest.csv",
        encoding="utf-8-sig",
    )
    manifest["PatientID"] = manifest["PatientID"].astype(str)
    lung1_ids = set(manifest.loc[manifest["cohort"].eq("LUNG1_all"), "PatientID"])
    lung2_ids = set(manifest.loc[manifest["cohort"].eq("LUNG2_validation"), "PatientID"])
    expected_split_n = {
        "LUNG1_training": len(set(outcomes["LUNG1_training"]["PatientID"]) & lung1_ids),
        "LUNG1_testing": len(set(outcomes["LUNG1_testing"]["PatientID"]) & lung1_ids),
        "LUNG2_validation": len(lung2_ids),
    }
    methods = ["original_radiomics"] + [
        f"{zone}_physical{distance}mm"
        for distance in (2, 3, 5)
        for zone in ("inner_ring", "inner_core", "outer_ring")
    ]
    all_performance, all_predictions, all_selected, all_tuning = [], [], [], []
    all_exclusions = []

    for condition in methods:
        if condition == "original_radiomics":
            datasets = {k: outcomes[k].merge(baseline[k], on="PatientID") for k in outcomes}
            feature_columns = [c for c in datasets["LUNG1_training"] if c.startswith("base__")]
        else:
            zone, distance = condition.rsplit("_", 1)
            spatial = pd.read_csv(ZROOT / f"features_{distance}.csv")
            feature_columns = [c for c in spatial if c.startswith(zone + "__")]
            datasets = {}
            for cohort, outcome in outcomes.items():
                source_cohort = "LUNG1" if cohort.startswith("LUNG1") else "LUNG2"
                datasets[cohort] = outcome.merge(
                    spatial.loc[spatial["cohort"].eq(source_cohort), ["PatientID"] + feature_columns],
                    on="PatientID",
                )

        # Fixed before any feature selection, CV, preprocessing, or fitting.
        datasets["LUNG1_training"] = datasets["LUNG1_training"].loc[
            datasets["LUNG1_training"]["PatientID"].isin(lung1_ids)].copy()
        datasets["LUNG1_testing"] = datasets["LUNG1_testing"].loc[
            datasets["LUNG1_testing"]["PatientID"].isin(lung1_ids)].copy()
        datasets["LUNG2_validation"] = datasets["LUNG2_validation"].loc[
            datasets["LUNG2_validation"]["PatientID"].isin(lung2_ids)].copy()
        for cohort, data in datasets.items():
            all_exclusions.append({
                "condition": condition, "cohort": cohort,
                "n_after_common_filter": len(data),
                "n_expected_common": expected_split_n[cohort],
                "missing_after_merge": expected_split_n[cohort] - len(data),
            })

        development = pd.concat(
            [datasets["LUNG1_training"], datasets["LUNG1_testing"]], ignore_index=True
        ).dropna(subset=["time", "event"]).copy()
        usable = [c for c in feature_columns if development[c].notna().sum() >= 30 and development[c].nunique(dropna=True) > 1]
        chosen, beta, tuning, setting = select_enet(development, usable)
        tuning.assign(condition=condition).to_csv(OUT / f"tuning_{condition}.csv", index=False, encoding="utf-8-sig")
        all_tuning.append({"condition": condition, **setting, "n_candidates": len(usable)})
        for rank, feature in enumerate(chosen, 1):
            all_selected.append({
                "condition": condition, "rank": rank, "feature": feature,
                "elastic_net_coefficient": beta[feature],
            })

        # Fit final imputer/scaler and Cox model only on fixed common LUNG1.
        imputer = SimpleImputer(strategy="median").fit(development[chosen])
        scaler = StandardScaler().fit(imputer.transform(development[chosen]))
        z_development = np.clip(scaler.transform(imputer.transform(development[chosen])), -10.0, 10.0)
        feature_cols = [f"feature_{i}" for i in range(len(chosen))]
        development_model = development[["PatientID", "time", "event", "age", "male"]].copy()
        for i, col in enumerate(feature_cols):
            development_model[col] = z_development[:, i]
        fitted = {}
        for model_name, covariates in [("radiomics", feature_cols), ("age_sex", ["age", "male"] + feature_cols)]:
            fitted[model_name] = CoxPHFitter(penalizer=0.01).fit(
                development_model[["time", "event"] + covariates], "time", "event", robust=True
            )

        for cohort_name, data in [("LUNG1_all", development), ("LUNG2_validation", datasets["LUNG2_validation"])]:
            data = data.dropna(subset=["time", "event", "age", "male"]).copy()
            z = np.clip(scaler.transform(imputer.transform(data[chosen])), -10.0, 10.0)
            for i, col in enumerate(feature_cols):
                data[col] = z[:, i]
            for model_name, covariates in [("radiomics", feature_cols), ("age_sex", ["age", "male"] + feature_cols)]:
                risk = fitted[model_name].predict_log_partial_hazard(data[covariates]).to_numpy().ravel()
                all_performance.append({
                    "condition": condition, "model": model_name, "cohort": cohort_name,
                    "n": len(data), "events": int(data["event"].sum()),
                    "c_index": c_index(data, risk), "selected_features": ";".join(chosen),
                })
                all_predictions.append(pd.DataFrame({
                    "condition": condition, "model": model_name, "cohort": cohort_name,
                    "PatientID": data["PatientID"], "time": data["time"],
                    "event": data["event"], "risk_score": risk,
                }))

    pd.DataFrame(all_performance).to_csv(OUT / "performance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_selected).to_csv(OUT / "selected_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_tuning).to_csv(OUT / "elastic_net_tuning_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT / "fixed_model_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_exclusions).to_csv(OUT / "common_cohort_check.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"cohort": ["LUNG1_all", "LUNG2_validation"], "n": [len(lung1_ids), len(lung2_ids)]}).to_csv(
        OUT / "common_cohort_manifest_summary.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "metadata.json").write_text(json.dumps({
        "analysis": "strict common cohort + fold-specific preprocessing (main analysis A)",
        "max_features": MAX_FEATURES, "seed": SEED,
        "lung2_original_radiomics_source": "Le et al. Supplementary Table S3",
        "lung1_common_n": len(lung1_ids), "lung2_common_n": len(lung2_ids),
        "age_training_median": age_median, "conditions": methods,
        "models": ["radiomics", "age_sex"],
        "development_cohort": "fixed common LUNG1 397 cases (training + testing)",
        "validation_cohort": "fixed common LUNG2 113 cases",
        "fold_preprocessing": "median imputation and StandardScaler fit on each CV training fold",
    }, indent=2), encoding="utf-8")
    (OUT / "strict_common_cohort_reanalysis_procedure.md").write_text(
        "# Strict common-cohort reanalysis of main analysis A\n\n"
        "The common LUNG1 cohort (397 cases) and common LUNG2 cohort (113 cases) were fixed before condition-specific feature selection, CV preprocessing, and model fitting. LUNG1 training and testing were combined as the fixed LUNG1 development cohort; therefore LUNG1_all is not an independent held-out test.\n\n"
        "For each of 10 conditions (original radiomics; inner ring, inner core, and outer ring at physical 2, 3, and 5 mm), Elastic-Net Cox hyperparameters were selected with 5-fold stratified CV. In every fold, median imputation and standardization were fit using only the fold training subset and then applied to the validation subset. Final preprocessing and Cox coefficients were fit on the fixed common LUNG1 development cohort and applied to the fixed common LUNG2 validation cohort.\n\n"
        "Original-GTV radiomics were obtained from Le et al. Supplementary Tables S1 and S2 for LUNG1 and Supplementary Table S3 for LUNG2.\n\n"
        "This is a new strict reanalysis and does not overwrite results from 13_recalculated_analysis_A.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
