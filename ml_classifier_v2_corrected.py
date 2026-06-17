"""
ML runner v3 — AD fold-changes adjusted to published range
S100A8/S100A9/C1QB in blood AD: 0.8-1.2 log2FC (Lunnon et al 2012, AddNeuroMed)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

def simulate_ad_v3():
    np.random.seed(43)
    n_ad, n_mci, n_ctrl = 288, 238, 394
    n_total = n_ad + n_mci + n_ctrl
    n_genes = 22000
    baseline = np.random.normal(8.0, 1.5, (n_genes, n_total))
    baseline = baseline + np.random.normal(0, 0.2, (1, n_total))

    # Adjusted to published AddNeuroMed/GSE63060 blood effect sizes
    # Strong markers: 0.8-1.2 log2FC; weaker markers: 0.4-0.6 log2FC
    up = {"S100A8":1.10,"S100A9":1.00,"C1QB":0.95,"C1QC":0.85,
          "PF4":0.75,"MX1":0.80,"OAS1":0.65,"TXNIP":0.55,"IL1B":0.70}
    dn = {"CD3D":-0.90,"CD3E":-0.80,"PRF1":-0.72,"NKG7":-0.65,
          "IRS1":-0.55,"PPARGC1A":-0.68,"ADIPOR1":-0.48}
    gene_fcs = {**up, **dn}

    for i, (g, fc) in enumerate(gene_fcs.items()):
        # Per-sample noise: 35% of fold-change (realistic heterogeneity)
        noise    = np.random.normal(0, abs(fc)*0.35, n_ad)
        noise_m  = np.random.normal(0, abs(fc)*0.35, n_mci)
        if fc > 0:
            baseline[i,:n_ad]            += (fc + noise)
            baseline[i,n_ad:n_ad+n_mci] += (fc*0.55 + noise_m)
        else:
            baseline[i,:n_ad]            += (fc + noise)
            baseline[i,n_ad:n_ad+n_mci] += (fc*0.45 + noise_m)

    return baseline.T, np.array([0]*n_ad + [1]*n_mci + [2]*n_ctrl)

def simulate_t2d():
    np.random.seed(42)
    n_t2d, n_pre, n_ctrl = 509, 180, 501
    n_total = n_t2d + n_pre + n_ctrl
    n_genes = 18000
    baseline = np.random.negative_binomial(20, 0.5, size=(n_genes, n_total)).astype(float)
    baseline = baseline * (1 + np.random.normal(0, 0.15, (1, n_total)))
    gene_fcs = [0.75,0.68,0.80,0.62,0.55,0.48,0.42,0.38,0.35,0.32,
                -0.70,-0.75,-0.55,-0.48,-0.45,-0.40,-0.50,-0.38]
    for i, fc in enumerate(gene_fcs):
        noise   = np.random.normal(0, abs(fc)*0.4, n_t2d)
        noise_p = np.random.normal(0, abs(fc)*0.4, n_pre)
        if fc > 0:
            baseline[i,:n_t2d]            *= 2**(fc+noise)
            baseline[i,n_t2d:n_t2d+n_pre] *= 2**(fc*0.5+noise_p)
        else:
            baseline[i,:n_t2d]            /= 2**(abs(fc)+noise)
            baseline[i,n_t2d:n_t2d+n_pre] /= 2**(abs(fc)*0.5+noise_p)
    return np.log2(baseline+1).T, np.array([0]*n_t2d+[1]*n_pre+[2]*n_ctrl)

def run_ml(X_all, y_all, disease, n_top=500):
    X_df = pd.DataFrame(X_all)
    Xtr_df,Xte_df,ytr,yte = train_test_split(
        X_df, y_all, test_size=0.20, stratify=y_all, random_state=42)
    top = Xtr_df.var(axis=0).nlargest(n_top).index
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr_df[top])
    Xte = sc.transform(Xte_df[top])

    models = {
        "Random Forest":    RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "Gradient Boosting":GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        "SVM (RBF)":        SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
        "MLP":              MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42),
        "LASSO Logistic":   OneVsRestClassifier(LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42)),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\n{'='*62}\n{disease}\n{'='*62}")
    print(f"{'Model':<22} {'CV AUC':>10} {'Test AUC':>10} {'Test Acc':>10}")
    print("-"*55)
    results = []
    for name, model in models.items():
        print(f"  Training {name}...", flush=True)
        cv_auc   = cross_val_score(model, Xtr, ytr, cv=cv, scoring="roc_auc_ovr").mean()
        model.fit(Xtr, ytr)
        test_auc = roc_auc_score(yte, model.predict_proba(Xte), multi_class="ovr")
        test_acc = accuracy_score(yte, model.predict(Xte))
        print(f"  {name:<20} CV={cv_auc:.4f}  Test={test_auc:.4f}  Acc={test_acc:.4f}")
        results.append({"Model":name,"CV AUC":round(cv_auc,4),
                        "Test AUC":round(test_auc,4),"Test Acc":round(test_acc,4)})
    df = pd.DataFrame(results)
    print(f"\nSummary:\n{df.to_string(index=False)}")
    return df

print("Blood Multi-Omics ML Pipeline v3")
print("T2DM: 0.3-0.8 log2FC | AD: 0.5-1.1 log2FC (AddNeuroMed range)")

print("\nSimulating T2DM...")
r_t2d = run_ml(*simulate_t2d(), "T2DM (T2D vs Pre-DM vs Control)")

print("\nSimulating AD...")
r_ad = run_ml(*simulate_ad_v3(), "AD (AD vs MCI vs Control)")

r_t2d.to_csv("t2d_ml_results_v3.csv", index=False)
r_ad.to_csv("ad_ml_results_v3.csv",  index=False)
print("\nSaved: t2d_ml_results_v3.csv and ad_ml_results_v3.csv")
