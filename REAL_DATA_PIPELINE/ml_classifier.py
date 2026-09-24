#!/usr/bin/env python3
"""
06_ml_pipeline.py
=================
Machine learning classification pipeline for blood-based T2DM and AD diagnosis.

LEAKAGE FIX (v2):
    Feature selection (top N genes by variance) is now performed on the
    TRAINING SET ONLY, then applied to the test set. Previously, variance
    was computed on all samples before the train/test split, which allowed
    test-set signal to influence feature selection — a standard form of
    data leakage that inflates AUC.

Models trained:
    - Random Forest (RF)
    - Support Vector Machine (SVM, RBF kernel)
    - Gradient Boosting Machine (GBM)
    - Multilayer Perceptron (MLP)
    - Logistic Regression with LASSO (L1)

Features: Top N variable genes (selected on training set only) + clinical
Evaluation: 5-fold stratified CV + independent held-out test set (80:20)
Tracking: MLflow experiment logging
Interpretability: SHAP feature importance

Author: Usama Manzoor (usama.manzoor1121@gmail.com)
"""

import yaml
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, classification_report,
                              roc_curve, confusion_matrix)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PARAMS   = yaml.safe_load(open("params.yaml"))
PROC_DIR = Path(PARAMS["paths"]["processed_data"])
MOD_DIR  = Path(PARAMS["paths"]["models"])
FIG_DIR  = Path(PARAMS["paths"]["figures"]) / "ml"
TAB_DIR  = Path(PARAMS["paths"]["tables"])
MOD_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RS       = PARAMS["ml"]["random_state"]
CV       = PARAMS["ml"]["cv_folds"]
TEST     = PARAMS["ml"]["test_size"]
N_GENES  = PARAMS["ml"].get("n_top_genes", 500)   # new param — see params.yaml


def build_models() -> dict:
    """Instantiate all classifiers with params from params.yaml."""
    hp = PARAMS["ml"]["hyperparams"]
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=hp["random_forest"]["n_estimators"],
            min_samples_leaf=hp["random_forest"]["min_samples_leaf"],
            n_jobs=-1, random_state=RS
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=hp["gradient_boosting"]["n_estimators"],
            learning_rate=hp["gradient_boosting"]["learning_rate"],
            max_depth=hp["gradient_boosting"]["max_depth"],
            subsample=hp["gradient_boosting"]["subsample"],
            random_state=RS
        ),
        "SVM (RBF)": SVC(
            C=hp["svm_rbf"]["C"],
            kernel="rbf",
            probability=True,
            random_state=RS
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=tuple(hp["mlp"]["hidden_layers"]),
            max_iter=hp["mlp"]["epochs"],
            random_state=RS
        ),
        "LASSO Logistic": LogisticRegression(
            penalty="l1", solver="saga", C=1.0, random_state=RS, max_iter=3000
        ),
    }


def prepare_features(disease: str) -> tuple:
    """
    Load processed expression + metadata.
    Returns (X_all, y, sample_ids) — WITHOUT feature selection.
    Feature selection is done AFTER train/test split in run_pipeline().

    This separation is critical: selecting features on all samples before
    splitting leaks test-set information into the training process.
    """
    expr_path = PROC_DIR / f"{disease}_expression_normalised.csv"
    meta_path = PROC_DIR / f"{disease}_metadata_clean.csv"

    if not expr_path.exists():
        log.warning(f"Processed data not found for {disease}. Running preprocessing.")
        import subprocess
        subprocess.run(["python", "scripts/02_preprocessing/preprocess_blood.py"])

    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path, index_col=0)
    common = expr.columns.intersection(meta.index)
    expr = expr[common]
    meta = meta.loc[common]

    # Full gene matrix — NO feature selection here
    X_all = expr.T   # samples × genes

    # Labels
    diag_col = "diagnosis" if "diagnosis" in meta.columns else meta.columns[0]
    le = LabelEncoder()
    y  = le.fit_transform(meta[diag_col])

    log.info(f"  Full feature matrix: {X_all.shape[0]} samples × {X_all.shape[1]} genes")
    log.info(f"  Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    # Add clinical features if available
    clinical_cols = [c for c in ["HbA1c", "HOMA_IR", "FastingGlucose", "CRP",
                                  "MMSE", "NfL", "GFAP"]
                     if c in meta.columns]
    if clinical_cols:
        log.info(f"  Clinical features available: {clinical_cols}")

    return X_all, y, le.classes_, meta[diag_col], meta, clinical_cols


def select_features_on_train(X_train_df: pd.DataFrame,
                              X_test_df: pd.DataFrame,
                              n_top: int = 500) -> tuple:
    """
    Select top N genes by variance computed on TRAINING SET ONLY.
    Apply the same gene selection to the test set.

    This is the correct approach — the test set has no influence on
    which features are selected.

    Returns (X_train_selected, X_test_selected, selected_gene_names)
    """
    train_var  = X_train_df.var(axis=0)
    top_genes  = train_var.nlargest(n_top).index
    return (
        X_train_df[top_genes].values,
        X_test_df[top_genes].values,
        list(top_genes)
    )


def evaluate_model(name: str, model, X_train, X_test,
                   y_train, y_test, feature_names: list) -> dict:
    """Train, evaluate, and log a single model with MLflow."""
    with mlflow.start_run(run_name=name, nested=True):

        # Scale features (fit on train, transform both)
        scaler  = StandardScaler()
        Xtr_sc  = scaler.fit_transform(X_train)
        Xte_sc  = scaler.transform(X_test)

        # Cross-validation on training set
        cv_aucs = cross_val_score(
            model, Xtr_sc, y_train,
            cv=StratifiedKFold(n_splits=CV, shuffle=True, random_state=RS),
            scoring="roc_auc_ovr"
        )
        log.info(f"  {name}: CV AUC = {cv_aucs.mean():.4f} +/- {cv_aucs.std():.4f}")

        # Fit on full training set
        model.fit(Xtr_sc, y_train)

        # Test set evaluation
        y_prob   = model.predict_proba(Xte_sc)
        y_pred   = model.predict(Xte_sc)
        test_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        test_acc = accuracy_score(y_test, y_pred)

        log.info(f"  {name}: Test AUC = {test_auc:.4f} | Acc = {test_acc:.4f}")
        log.info(f"\n{classification_report(y_test, y_pred)}")

        mlflow.log_params({
            "model":      name,
            "n_features": X_train.shape[1],
            "n_train":    X_train.shape[0],
            "n_test":     X_test.shape[0],
        })
        mlflow.log_metrics({
            "cv_auc_mean": cv_aucs.mean(),
            "cv_auc_std":  cv_aucs.std(),
            "test_auc":    test_auc,
            "test_acc":    test_acc,
        })
        try:
            mlflow.sklearn.log_model(model, name=name.replace(" ", "_"))
        except Exception as e:
            log.warning(f"  MLflow model artifact logging skipped ({name}): {e}")

        return {
            "model":        name,
            "cv_auc_mean":  round(cv_aucs.mean(), 4),
            "cv_auc_std":   round(cv_aucs.std(),  4),
            "test_auc":     round(test_auc, 4),
            "test_acc":     round(test_acc, 4),
            "fitted_model": model,
            "scaler":       scaler,
            "y_prob":       y_prob,
            "y_pred":       y_pred,
        }


def compute_shap(model, X_train_sc, X_test_sc, feature_names, outdir, model_name):
    """Compute and save SHAP feature importance plot."""
    if not PARAMS["ml"]["shap"]["enabled"]:
        return
    try:
        n_bg       = min(PARAMS["ml"]["shap"]["n_background"], X_train_sc.shape[0])
        bg         = shap.sample(X_train_sc, n_bg)
        explainer  = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") \
                     else shap.KernelExplainer(model.predict_proba, bg)
        shap_vals  = explainer.shap_values(X_test_sc[:100])
        if isinstance(shap_vals, list):
            shap_vals = np.abs(np.array(shap_vals)).mean(axis=0)
        max_disp = min(PARAMS["ml"]["shap"]["max_display"], len(feature_names))
        shap.summary_plot(shap_vals, X_test_sc[:100],
                          feature_names=feature_names,
                          max_display=max_disp,
                          show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(outdir / f"shap_{model_name.replace(' ', '_')}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info(f"  SHAP plot saved for {model_name}")
    except Exception as e:
        log.warning(f"  SHAP failed for {model_name}: {e}")


def plot_roc_curves(results, y_test, outdir, disease):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    colors = ["#02C39A", "#E74C3C", "#F4A261", "#9B59B6", "#3498DB"]
    ax.plot([0, 1], [0, 1], "w--", alpha=0.4, label="Random (AUC=0.5)")

    for res, color in zip(results, colors):
        y_prob = res["y_prob"]
        if y_prob.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        else:
            fpr, tpr, _ = roc_curve(
                (y_test > 0).astype(int), y_prob.max(axis=1)
            )
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{res['model']} (AUC={res['test_auc']:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curves — {disease.upper()} Blood Classification", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#2A2A4A", labelcolor="white", fontsize=9)
    ax.grid(alpha=0.15, color="white")
    plt.tight_layout()
    plt.savefig(outdir / f"{disease}_roc_curves.png", dpi=300,
                bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()
    log.info(f"  ROC curves saved: {disease}_roc_curves.png")


def run_pipeline(disease: str):
    """Run full ML pipeline for one disease."""
    log.info(f"\n{'='*60}")
    log.info(f"ML Pipeline — {disease.upper()}")
    log.info(f"{'='*60}")

    mlflow.set_experiment(f"blood_multiomics_{disease}")

    # 1. Load all data — no feature selection yet
    X_all, y, classes, labels, meta, clinical_cols = prepare_features(disease)

    # 2. Train/test split BEFORE feature selection
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_all, y, test_size=TEST, stratify=y, random_state=RS
    )
    log.info(f"Train: {X_train_df.shape[0]} | Test: {X_test_df.shape[0]}")

    # 3. Feature selection on TRAINING SET ONLY
    X_train, X_test, selected_genes = select_features_on_train(
        X_train_df, X_test_df, n_top=N_GENES
    )
    log.info(f"Selected {len(selected_genes)} genes (variance on training set only)")

    # 4. Add clinical features if available
    if clinical_cols:
        train_idx = X_train_df.index
        test_idx  = X_test_df.index
        clin_train = meta.loc[train_idx, clinical_cols].values
        clin_test  = meta.loc[test_idx,  clinical_cols].values
        X_train = np.hstack([X_train, clin_train])
        X_test  = np.hstack([X_test,  clin_test])
        feature_names = selected_genes + clinical_cols
    else:
        feature_names = selected_genes

    log.info(f"Final feature matrix: {X_train.shape[1]} features")

    models  = build_models()
    results = []

    with mlflow.start_run(run_name=f"{disease}_full_pipeline"):
        mlflow.log_params({
            "disease":    disease,
            "n_features": X_train.shape[1],
            "n_genes":    N_GENES,
            "n_samples":  len(y),
            "cv_folds":   CV,
            "test_size":  TEST,
            "feature_selection": "train_variance_only",  # log the fix
        })

        for name, model in models.items():
            log.info(f"\n  Training: {name}")
            res = evaluate_model(
                name, model, X_train, X_test, y_train, y_test, feature_names
            )
            results.append(res)

            scaler  = res.pop("scaler")
            fitted  = res.pop("fitted_model")
            joblib.dump({"model": fitted, "scaler": scaler,
                         "selected_genes": selected_genes},
                        MOD_DIR / f"{disease}_{name.replace(' ', '_')}.pkl")

            if "Forest" in name or "Boosting" in name:
                Xtr_sc = scaler.transform(X_train)
                Xte_sc = scaler.transform(X_test)
                compute_shap(fitted, Xtr_sc, Xte_sc,
                             feature_names, FIG_DIR, f"{disease}_{name}")

    # Save results
    results_df = pd.DataFrame([{k: v for k, v in r.items()
                                 if k not in ("y_prob", "y_pred")}
                                for r in results])
    results_df.to_csv(TAB_DIR / f"{disease}_ml_results.csv", index=False)
    pd.Series(selected_genes).to_csv(
        TAB_DIR / f"{disease}_selected_genes.csv", index=False, header=["gene"]
    )

    plot_roc_curves(results, y_test, FIG_DIR, disease)

    best = max(results, key=lambda x: x.get("test_auc", 0))
    log.info(f"\n  Best model: {best['model']} (Test AUC={best.get('test_auc', 'N/A'):.4f})")

    return results_df


if __name__ == "__main__":
    log.info("Blood Multi-Omics ML Pipeline (v2 — leakage-corrected)")
    log.info("=" * 60)
    log.info("Feature selection: training set variance only (correct)")
    log.info("=" * 60)

    mlflow.set_tracking_uri(PARAMS["paths"]["mlflow_uri"])

    for disease in ["t2d", "ad"]:
        results = run_pipeline(disease)
        log.info(f"\n{disease.upper()} Results:")
        log.info(results[["model", "cv_auc_mean", "cv_auc_std", "test_auc", "test_acc"]].to_string(index=False))

    log.info("\nAll models trained and logged to MLflow.")
