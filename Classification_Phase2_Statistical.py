#!/usr/bin/env python3
"""
================================================================================
FASE 2 — CLASSIFICAÇÃO SUPERVISIONADA DE PERFIS DE EMPREGADOS
Pipeline Completo com Validação Estatística Rigorosa

Configuração:
  - K-Means k=8, 5 variáveis (alinhado Fase 1 / Dissertação)
  - Solver: newton-cg (equivalente R nnet::multinom)
  - Cross-Validation: 10-fold estratificado
  - Seed: 123 (reprodutibilidade — mesmo da dissertação)
  - Intervalos de Confiança: 95%
  - Testes estatísticos: McNemar (pairwise) + Cochran's Q (omnibus)
  - ROC comparativa: todos os modelos
================================================================================
"""

import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import os; os.makedirs("MDM", exist_ok=True)

# === ML ===
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# === Stats ===
from scipy import stats
from itertools import combinations

# === Plots ===
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ============================================================================
# PARAMETERS
# ============================================================================
SEED = 123
N_FOLDS = 10
K_CLUSTERS = 8
TEST_SIZE = 0.3
CI_LEVEL = 0.95
CI_Z = stats.norm.ppf(1 - (1 - CI_LEVEL) / 2)  # 1.96

np.random.seed(SEED)

print("=" * 75)
print("  PHASE 2: Supervised Classification — Full Statistical Pipeline")
print(f"  Config: k={K_CLUSTERS}, {N_FOLDS}-fold CV, seed={SEED}, CI={CI_LEVEL*100:.0f}%")
print("=" * 75)

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================
print("\n[1/10] Loading and preparing data...")

df = pd.read_excel("MDM/BD_APEX_SUM.xlsx")
dados = df[['IDADE', 'TEMPO_DE_CASA', 'TOTAL_DEPENDENTES', 'GENERO', 'TOTAL_MINUTOS']].copy()
dados['GENERO'] = dados['GENERO'].map({'M': 2, 'F': 1})
dados = dados.dropna().drop_duplicates()
print(f"  Dataset: {dados.shape[0]} employees, {dados.shape[1]} variables")

# ============================================================================
# 2. CLUSTERING (Phase 1 replication)
# ============================================================================
print("\n[2/10] Replicating Phase 1 clustering (K-Means k=8)...")

feature_cols = ['IDADE', 'TEMPO_DE_CASA', 'TOTAL_DEPENDENTES', 'GENERO', 'TOTAL_MINUTOS']
scaler = StandardScaler()
dados_norm = scaler.fit_transform(dados[feature_cols])

kmeans = KMeans(n_clusters=K_CLUSTERS, n_init=25, random_state=SEED)
clusters = kmeans.fit_predict(dados_norm)
dados['cluster'] = clusters + 1

sil = silhouette_score(dados_norm, clusters)
print(f"  Silhouette Index: {sil:.4f} (article: 0.34) {'✅' if round(sil, 2) == 0.34 else '⚠️'}")

# Cluster distribution
dist = dados.groupby('cluster').agg(
    N=('cluster', 'count'),
    Age=('IDADE', 'mean'),
    Tenure=('TEMPO_DE_CASA', 'mean'),
    Dep=('TOTAL_DEPENDENTES', 'mean'),
    Absent_h=('TOTAL_MINUTOS', lambda x: (x/60).mean()),
).reset_index()
dist['%'] = (dist['N'] / dist['N'].sum() * 100).round(1)
dist['%Fem'] = dados.groupby('cluster')['GENERO'].apply(lambda x: (x==1).mean()*100).round(0).values
print(f"\n  Cluster Distribution:")
print(f"  {'C':>2} {'N':>4} {'%':>5} {'Age':>5} {'Ten':>5} {'Dep':>5} {'Abs_h':>7} {'%F':>4}")
print(f"  " + "-"*42)
for _, r in dist.iterrows():
    print(f"  {int(r['cluster']):>2} {int(r['N']):>4} {r['%']:>5.1f} {r['Age']:>5.0f} {r['Tenure']:>5.0f} {r['Dep']:>5.1f} {r['Absent_h']:>7.0f} {r['%Fem']:>4.0f}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n[3/10] Train-test split (70/30 stratified)...")

X = dados[feature_cols].values
y = dados['cluster'].values
classes = sorted(np.unique(y))
n_classes = len(classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================================
# 4. FIND BEST K FOR KNN
# ============================================================================
print("\n[4/10] Optimizing KNN (finding best k)...")

cv_strat = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
knn_k_scores = {}
for k in range(1, 21):
    scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=k), X_train, y_train,
        cv=cv_strat, scoring='accuracy'
    )
    knn_k_scores[k] = {'mean': scores.mean(), 'std': scores.std()}

best_k = max(knn_k_scores, key=lambda k: knn_k_scores[k]['mean'])
print(f"  Best k={best_k} (CV Accuracy={knn_k_scores[best_k]['mean']:.4f})")

# ============================================================================
# 5. DEFINE MODELS
# ============================================================================
models = {
    'Multinomial Logistic Regression': LogisticRegression(
        solver='newton-cg', max_iter=10000, random_state=SEED
    ),
    'Decision Tree': DecisionTreeClassifier(random_state=SEED),
    'Random Forest': RandomForestClassifier(
        n_estimators=500, random_state=SEED
    ),
    f'KNN (k={best_k})': KNeighborsClassifier(n_neighbors=best_k),
}

# ============================================================================
# 6. 10-FOLD STRATIFIED CROSS-VALIDATION WITH FULL METRICS
# ============================================================================
print(f"\n[5/10] Running {N_FOLDS}-fold stratified cross-validation...")

cv_results = {}

for name, model in models.items():
    print(f"\n  --- {name} ---")
    
    # Per-fold metrics
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    
    fold_idx = 0
    for train_idx, val_idx in cv_strat.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Clone model for each fold
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_fold_train, y_fold_train)
        y_fold_pred = fold_model.predict(X_fold_val)
        
        fold_accuracies.append(accuracy_score(y_fold_val, y_fold_pred))
        fold_precisions.append(precision_score(y_fold_val, y_fold_pred, average='weighted', zero_division=0))
        fold_recalls.append(recall_score(y_fold_val, y_fold_pred, average='weighted', zero_division=0))
        fold_f1s.append(f1_score(y_fold_val, y_fold_pred, average='weighted', zero_division=0))
        fold_idx += 1
    
    fold_accuracies = np.array(fold_accuracies)
    fold_precisions = np.array(fold_precisions)
    fold_recalls = np.array(fold_recalls)
    fold_f1s = np.array(fold_f1s)
    
    # CI calculation (t-distribution for small samples)
    t_crit = stats.t.ppf(1 - (1 - CI_LEVEL) / 2, df=N_FOLDS - 1)
    
    def ci(arr):
        m, se = arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr))
        return m, m - t_crit * se, m + t_crit * se, se
    
    acc_m, acc_lo, acc_hi, acc_se = ci(fold_accuracies)
    prec_m, prec_lo, prec_hi, _ = ci(fold_precisions)
    rec_m, rec_lo, rec_hi, _ = ci(fold_recalls)
    f1_m, f1_lo, f1_hi, _ = ci(fold_f1s)
    
    # Train on full training set → evaluate on held-out test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    test_rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    cv_results[name] = {
        'fold_accuracies': fold_accuracies,
        'cv_acc_mean': acc_m, 'cv_acc_lo': acc_lo, 'cv_acc_hi': acc_hi, 'cv_acc_se': acc_se,
        'cv_prec_mean': prec_m, 'cv_prec_lo': prec_lo, 'cv_prec_hi': prec_hi,
        'cv_rec_mean': rec_m, 'cv_rec_lo': rec_lo, 'cv_rec_hi': rec_hi,
        'cv_f1_mean': f1_m, 'cv_f1_lo': f1_lo, 'cv_f1_hi': f1_hi,
        'test_acc': test_acc, 'test_prec': test_prec, 'test_rec': test_rec, 'test_f1': test_f1,
        'y_pred': y_pred,
    }
    
    print(f"    CV Accuracy:  {acc_m:.4f} ± {acc_se:.4f}  [{acc_lo:.4f}, {acc_hi:.4f}]")
    print(f"    CV Precision: {prec_m:.4f}  [{prec_lo:.4f}, {prec_hi:.4f}]")
    print(f"    CV Recall:    {rec_m:.4f}  [{rec_lo:.4f}, {rec_hi:.4f}]")
    print(f"    CV F1:        {f1_m:.4f}  [{f1_lo:.4f}, {f1_hi:.4f}]")
    print(f"    Test Accuracy: {test_acc:.4f}")

# ============================================================================
# 7. TABLE 3: COMPREHENSIVE METRICS WITH CONFIDENCE INTERVALS
# ============================================================================
print(f"\n[6/10] Building Table 3...")

table3_rows = []
for name in models:
    r = cv_results[name]
    table3_rows.append({
        'Model': name,
        'CV_Accuracy': f"{r['cv_acc_mean']:.4f}",
        'CV_Accuracy_CI95': f"[{r['cv_acc_lo']:.4f}, {r['cv_acc_hi']:.4f}]",
        'CV_Precision': f"{r['cv_prec_mean']:.4f}",
        'CV_Precision_CI95': f"[{r['cv_prec_lo']:.4f}, {r['cv_prec_hi']:.4f}]",
        'CV_Recall': f"{r['cv_rec_mean']:.4f}",
        'CV_Recall_CI95': f"[{r['cv_rec_lo']:.4f}, {r['cv_rec_hi']:.4f}]",
        'CV_F1': f"{r['cv_f1_mean']:.4f}",
        'CV_F1_CI95': f"[{r['cv_f1_lo']:.4f}, {r['cv_f1_hi']:.4f}]",
        'Test_Accuracy': f"{r['test_acc']:.4f}",
        'Test_F1': f"{r['test_f1']:.4f}",
    })

table3 = pd.DataFrame(table3_rows)
print("\n=== TABLE 3: Classification Metrics with 95% Confidence Intervals ===")
print(f"    (10-fold Stratified CV, seed={SEED})\n")
for _, row in table3.iterrows():
    print(f"  {row['Model']}")
    print(f"    CV Accuracy:  {row['CV_Accuracy']}  {row['CV_Accuracy_CI95']}")
    print(f"    CV F1 (wtd):  {row['CV_F1']}  {row['CV_F1_CI95']}")
    print(f"    Test Accuracy: {row['Test_Accuracy']}  |  Test F1: {row['Test_F1']}")
    print()

table3.to_excel("MDM/Table3_Metrics_with_CI.xlsx", index=False)

# Also export a numeric-only version for the article
table3_numeric = pd.DataFrame()
for name in models:
    r = cv_results[name]
    table3_numeric = pd.concat([table3_numeric, pd.DataFrame([{
        'Model': name,
        'CV_Acc': round(r['cv_acc_mean'], 4),
        'CV_Acc_SE': round(r['cv_acc_se'], 4),
        'CV_Acc_CI_Lo': round(r['cv_acc_lo'], 4),
        'CV_Acc_CI_Hi': round(r['cv_acc_hi'], 4),
        'CV_Prec': round(r['cv_prec_mean'], 4),
        'CV_Rec': round(r['cv_rec_mean'], 4),
        'CV_F1': round(r['cv_f1_mean'], 4),
        'Test_Acc': round(r['test_acc'], 4),
        'Test_Prec': round(r['test_prec'], 4),
        'Test_Rec': round(r['test_rec'], 4),
        'Test_F1': round(r['test_f1'], 4),
    }])], ignore_index=True)
table3_numeric.to_excel("MDM/Table3_Numeric.xlsx", index=False)

# ============================================================================
# 8. STATISTICAL TESTS
# ============================================================================
print(f"\n[7/10] Statistical tests for model comparison...")

model_names = list(models.keys())
n_models = len(model_names)

# --- 8.1 Cochran's Q Test (omnibus: are all models equally accurate?) ---
print("\n  === 8.1 Cochran's Q Test (omnibus) ===")
print("  H0: All models have the same error rate on the test set")
print("  H1: At least one model differs\n")

# Build binary correctness matrix (n_test × n_models)
correctness = np.zeros((len(y_test), n_models), dtype=int)
for j, name in enumerate(model_names):
    correctness[:, j] = (cv_results[name]['y_pred'] == y_test).astype(int)

n = correctness.shape[0]
k = correctness.shape[1]

# Cochran's Q statistic
row_sums = correctness.sum(axis=1)  # T_i
col_sums = correctness.sum(axis=0)  # C_j
T = row_sums.sum()

numerator = (k - 1) * (k * np.sum(col_sums**2) - T**2)
denominator = k * T - np.sum(row_sums**2)

if denominator > 0:
    Q_stat = numerator / denominator
    Q_df = k - 1
    Q_pvalue = 1 - stats.chi2.cdf(Q_stat, Q_df)
    print(f"  Q = {Q_stat:.4f}, df = {Q_df}, p-value = {Q_pvalue:.6f}")
    if Q_pvalue < 0.05:
        print(f"  → REJECT H0: Models differ significantly (p < 0.05)")
        print(f"  → Proceed to pairwise McNemar tests")
    else:
        print(f"  → FAIL TO REJECT H0: No significant difference")
else:
    print("  → Cannot compute (denominator = 0)")
    Q_stat, Q_pvalue = np.nan, np.nan

# --- 8.2 McNemar's Test (pairwise) ---
print("\n  === 8.2 McNemar's Tests (pairwise) ===")
print("  H0: Models i and j have the same error rate")
print("  Bonferroni correction for", int(n_models * (n_models - 1) / 2), "comparisons\n")

n_comparisons = int(n_models * (n_models - 1) / 2)
bonferroni_alpha = 0.05 / n_comparisons

mcnemar_results = []
print(f"  {'Model A':<35} {'Model B':<35} {'χ²':>8} {'p-value':>10} {'Bonf.':>8} {'Sig.':>5}")
print("  " + "-" * 105)

for i, j in combinations(range(n_models), 2):
    name_i, name_j = model_names[i], model_names[j]
    correct_i = correctness[:, i]
    correct_j = correctness[:, j]
    
    # Contingency: b = i correct & j wrong, c = i wrong & j correct
    b = np.sum((correct_i == 1) & (correct_j == 0))  # i right, j wrong
    c = np.sum((correct_i == 0) & (correct_j == 1))  # i wrong, j right
    
    # McNemar with continuity correction
    if (b + c) > 0:
        chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) >= 25 else (b - c)**2 / (b + c)
        p_val = 1 - stats.chi2.cdf(chi2, 1)
    else:
        chi2, p_val = 0.0, 1.0
    
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < bonferroni_alpha else "ns"
    
    mcnemar_results.append({
        'Model_A': name_i, 'Model_B': name_j,
        'b (A✓B✗)': b, 'c (A✗B✓)': c,
        'Chi2': round(chi2, 4), 'p_value': p_val,
        'Bonferroni_sig': sig
    })
    
    print(f"  {name_i:<35} {name_j:<35} {chi2:>8.4f} {p_val:>10.6f} {bonferroni_alpha:>8.4f} {sig:>5}")

mcnemar_df = pd.DataFrame(mcnemar_results)
mcnemar_df.to_excel("MDM/Table4_McNemar_Tests.xlsx", index=False)

# --- 8.3 Paired t-test on CV folds (Dietterich, 1998) ---
print("\n  === 8.3 Paired t-tests on CV fold accuracies ===")
print("  (Corrected for dependence between folds — Nadeau & Bengio, 2003)\n")

paired_t_results = []
print(f"  {'Model A':<35} {'Model B':<35} {'t-stat':>8} {'p-value':>10} {'Sig.':>5}")
print("  " + "-" * 98)

for i, j in combinations(range(n_models), 2):
    name_i, name_j = model_names[i], model_names[j]
    acc_i = cv_results[name_i]['fold_accuracies']
    acc_j = cv_results[name_j]['fold_accuracies']
    
    # Corrected repeated cross-validation t-test
    # (Nadeau & Bengio, 2003): accounts for non-independence of folds
    diffs = acc_i - acc_j
    d_bar = diffs.mean()
    n_folds = len(diffs)
    
    # Variance with correction for test/train ratio
    n_test_fold = len(X_train) / N_FOLDS
    n_train_fold = len(X_train) - n_test_fold
    
    var_d = diffs.var(ddof=1)
    # Corrected variance (Nadeau & Bengio)
    corrected_var = (1/n_folds + n_test_fold/n_train_fold) * var_d
    
    if corrected_var > 0:
        t_stat = d_bar / np.sqrt(corrected_var)
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_folds - 1))
    else:
        t_stat, p_val = 0.0, 1.0
    
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < bonferroni_alpha else "ns"
    
    paired_t_results.append({
        'Model_A': name_i, 'Model_B': name_j,
        'Mean_Diff': round(d_bar, 4),
        't_stat': round(t_stat, 4), 'p_value': p_val,
        'Bonferroni_sig': sig
    })
    
    print(f"  {name_i:<35} {name_j:<35} {t_stat:>8.4f} {p_val:>10.6f} {sig:>5}")

paired_t_df = pd.DataFrame(paired_t_results)
paired_t_df.to_excel("MDM/Table5_Paired_tTests.xlsx", index=False)

# ============================================================================
# 9. FIGURES
# ============================================================================
print(f"\n[8/10] Generating figures...")

# Color palette
colors = {
    'Multinomial Logistic Regression': '#2196F3',  # blue
    'Decision Tree': '#FF9800',                     # orange
    'Random Forest': '#4CAF50',                     # green
    f'KNN (k={best_k})': '#F44336',                # red
}

# --- 9.1 Confusion Matrices ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes_flat = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    cm = confusion_matrix(y_test, cv_results[name]['y_pred'], labels=classes)
    ax = axes_flat[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu_r',
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('Actual Class', fontsize=11)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')

plt.suptitle('Confusion Matrices — All Models (Test Set)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("MDM/Fig1_ConfusionMatrices_All.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig1_ConfusionMatrices_All.png")

# --- 9.2 COMPARATIVE ROC CURVES (all models, all classes) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Mean ROC per model (macro-averaged across classes)
ax = axes[0]
y_test_bin = label_binarize(y_test, classes=classes)

model_mean_aucs = {}
for name, model in models.items():
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)
    else:
        continue
    
    # Compute per-class ROC and average
    fpr_grid = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(fpr_grid)
    class_aucs = []
    
    for c_idx, cls in enumerate(classes):
        if c_idx < probs.shape[1]:
            fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, c_idx], probs[:, c_idx])
            mean_tpr += np.interp(fpr_grid, fpr_c, tpr_c)
            class_aucs.append(auc(fpr_c, tpr_c))
    
    mean_tpr /= n_classes
    mean_auc = np.mean(class_aucs)
    std_auc = np.std(class_aucs)
    model_mean_aucs[name] = {'mean': mean_auc, 'std': std_auc}
    
    ax.plot(fpr_grid, mean_tpr, color=colors[name], lw=2,
            label=f'{name}\n(AUC={mean_auc:.3f}±{std_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('(A) Macro-Average ROC — All Models', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

# Panel B: Per-class ROC for best model (Random Forest)
ax = axes[1]
rf_model = models['Random Forest']
rf_probs = rf_model.predict_proba(X_test)
class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

for c_idx, cls in enumerate(classes):
    fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, c_idx], rf_probs[:, c_idx])
    auc_c = auc(fpr_c, tpr_c)
    ax.plot(fpr_c, tpr_c, color=class_colors[c_idx], lw=1.5,
            label=f'Class {cls} (AUC={auc_c:.3f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('(B) Per-Class ROC — Random Forest', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig("MDM/Fig2_ROC_Comparative.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig2_ROC_Comparative.png")

# --- 9.3 CV ACCURACY WITH CONFIDENCE INTERVALS (Forest plot style) ---
fig, ax = plt.subplots(figsize=(10, 5))

y_positions = np.arange(n_models)
model_order = sorted(model_names, key=lambda n: cv_results[n]['cv_acc_mean'], reverse=True)

for idx, name in enumerate(model_order):
    r = cv_results[name]
    mean_acc = r['cv_acc_mean']
    lo = r['cv_acc_lo']
    hi = r['cv_acc_hi']
    
    # Point + CI bar
    ax.errorbar(mean_acc, idx, xerr=[[mean_acc - lo], [hi - mean_acc]],
                fmt='o', color=colors[name], markersize=10, capsize=6, capthick=2, lw=2)
    
    # Test accuracy as diamond
    ax.scatter(r['test_acc'], idx, marker='D', s=80, color=colors[name],
               edgecolors='black', linewidth=1, zorder=5)
    
    # Label
    ax.text(hi + 0.01, idx, f"CV: {mean_acc:.3f} [{lo:.3f}, {hi:.3f}]",
            va='center', fontsize=9)

ax.set_yticks(y_positions)
ax.set_yticklabels(model_order, fontsize=11)
ax.set_xlabel('Accuracy', fontsize=12)
ax.set_title(f'Cross-Validation Accuracy with {CI_LEVEL*100:.0f}% CI\n(●=CV mean, ◆=Test accuracy)',
             fontsize=13, fontweight='bold')
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlim([0, 1.15])
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("MDM/Fig3_CV_Accuracy_CI.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig3_CV_Accuracy_CI.png")

# --- 9.4 BOXPLOT OF CV FOLD ACCURACIES ---
fig, ax = plt.subplots(figsize=(10, 5))

fold_data = []
fold_labels = []
for name in model_order:
    fold_data.append(cv_results[name]['fold_accuracies'])
    fold_labels.append(name)

bp = ax.boxplot(fold_data, labels=fold_labels, patch_artist=True,
                medianprops=dict(color='black', lw=2),
                whiskerprops=dict(lw=1.5),
                capprops=dict(lw=1.5))

for patch, name in zip(bp['boxes'], model_order):
    patch.set_facecolor(colors[name])
    patch.set_alpha(0.7)

# Add individual fold points
for idx, (data, name) in enumerate(zip(fold_data, model_order)):
    x = np.random.normal(idx + 1, 0.04, len(data))
    ax.scatter(x, data, alpha=0.6, s=30, color=colors[name], edgecolors='black', linewidth=0.5)

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title(f'Distribution of {N_FOLDS}-Fold CV Accuracies', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig("MDM/Fig4_Boxplot_CV_Folds.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig4_Boxplot_CV_Folds.png")

# --- 9.5 VARIABLE IMPORTANCE ---
rf = models['Random Forest']
imp = pd.DataFrame({
    'Variable': feature_cols,
    'Gini': rf.feature_importances_
}).sort_values('Gini', ascending=True)  # ascending for horizontal bars

label_map = {'IDADE': 'Age', 'TEMPO_DE_CASA': 'Hired Time',
             'TOTAL_DEPENDENTES': 'Dependents', 'GENERO': 'Gender',
             'TOTAL_MINUTOS': 'Absenteeism (Min)'}
imp['Label'] = imp['Variable'].map(label_map)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(imp['Label'], imp['Gini'], color='#3575b5', edgecolor='white')
ax.set_xlabel('Mean Decrease Gini', fontsize=12)
ax.set_title('Variable Importance — Random Forest', fontsize=13, fontweight='bold')
for i, (v, g) in enumerate(zip(imp['Label'], imp['Gini'])):
    ax.text(g + 0.003, i, f'{g:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig("MDM/Fig5_Variable_Importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig5_Variable_Importance.png")

# --- 9.6 KNN k-optimization ---
fig, ax = plt.subplots(figsize=(8, 4))
ks = list(knn_k_scores.keys())
means = [knn_k_scores[k]['mean'] for k in ks]
stds = [knn_k_scores[k]['std'] for k in ks]

ax.errorbar(ks, means, yerr=[1.96*s/np.sqrt(N_FOLDS) for s in stds],
            fmt='o-', color='#3575b5', capsize=3, lw=1.5, markersize=5)
ax.axvline(x=best_k, color='red', ls='--', alpha=0.7, label=f'Best k={best_k}')
ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
ax.set_ylabel('CV Accuracy', fontsize=12)
ax.set_title('KNN: Accuracy vs k (with 95% CI)', fontsize=13, fontweight='bold')
ax.set_xticks(ks)
ax.legend()
plt.tight_layout()
plt.savefig("MDM/Fig6_KNN_Optimization.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: Fig6_KNN_Optimization.png")

# ============================================================================
# 10. PER-CLASS DETAILED REPORTS
# ============================================================================
print(f"\n[9/10] Generating detailed per-class reports...")

for name in ['Multinomial Logistic Regression', 'Random Forest']:
    report = classification_report(y_test, cv_results[name]['y_pred'], digits=4, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_excel(f"MDM/PerClass_{name.replace(' ', '_')}.xlsx")
    print(f"\n  === {name} — Per-Class Report ===")
    print(classification_report(y_test, cv_results[name]['y_pred'], digits=4))

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================
print(f"\n[10/10] Final Summary")
print("=" * 75)

print(f"""
  EXPERIMENTAL SETUP
  ─────────────────────────────────────────────────
  Dataset:           {len(dados)} employees, {len(feature_cols)} features
  Target variable:   {K_CLUSTERS} clusters (K-Means, Silhouette={sil:.4f})
  Train/Test split:  70/30 stratified (n_train={len(X_train)}, n_test={len(X_test)})
  Cross-validation:  {N_FOLDS}-fold stratified
  Seed:              {SEED}
  Confidence level:  {CI_LEVEL*100:.0f}%
  
  RESULTS (ranked by CV Accuracy)
  ─────────────────────────────────────────────────""")

for name in model_order:
    r = cv_results[name]
    print(f"  {name}")
    print(f"    CV:   Acc={r['cv_acc_mean']:.4f} [{r['cv_acc_lo']:.4f}, {r['cv_acc_hi']:.4f}]  F1={r['cv_f1_mean']:.4f}")
    print(f"    Test: Acc={r['test_acc']:.4f}  F1={r['test_f1']:.4f}")
    print()

print(f"""  STATISTICAL TESTS
  ─────────────────────────────────────────────────
  Cochran's Q:       Q={Q_stat:.4f}, df={Q_df}, p={Q_pvalue:.6f}
                     {'→ Models differ significantly' if Q_pvalue < 0.05 else '→ No significant difference'}
  McNemar (pairwise): See Table 4 (Bonferroni α={bonferroni_alpha:.4f})
  Paired t-test:      See Table 5 (Nadeau & Bengio correction)
  
  MEAN AUC (macro-averaged ROC)
  ─────────────────────────────────────────────────""")

for name in model_order:
    if name in model_mean_aucs:
        a = model_mean_aucs[name]
        print(f"  {name}: {a['mean']:.4f} ± {a['std']:.4f}")

print(f"""
  EXPORTED FILES
  ─────────────────────────────────────────────────""")
for f in sorted(os.listdir("MDM")):
    if f.endswith(('.xlsx', '.png')) and f != 'BD_APEX_SUM.xlsx':
        print(f"  MDM/{f}")

print(f"\n{'=' * 75}")
print(f"  Analysis complete. All results reproducible with seed={SEED}.")
print(f"{'=' * 75}")
