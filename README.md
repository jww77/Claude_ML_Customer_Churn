[ChurnScope_Guide.docx](https://github.com/user-attachments/files/25412609/ChurnScope_Guide.docx)
[ChurnScope_Guide.docx](https://github.com/user-attachments/files/25412609/ChurnScope_Guide.docx)
[ChurnScope_Sample_Report.html](https://github.com/user-attachments/files/25412611/ChurnScope_Sample_Report.html)
[ChurnScope_Business (1).ipynb](https://github.com/user-attachments/files/25412651/ChurnScope_Business.1.ipynb){
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  },
  "colab": {
   "provenance": []
  }
 },
 "cells": [
  {
   "cell_type": "markdown",
   "id": "title",
   "metadata": {},
   "source": "# 📊 ChurnScope — Customer Churn Analysis\n### Business Edition · Google Colab · No installation required\n\n---\n\n**What this notebook does in 12 steps:**\n- Uploads your customer CSV directly in the browser\n- Explores and visualises your churn data\n- Trains 3 machine learning models and picks the best\n- Ranks every customer by churn probability (0–100%)\n- Shows which factors are driving churn in your data\n- Calculates the ROI of ML-guided vs random outreach\n- Downloads a ranked action list and executive summary\n\n**Total time: ~5–10 minutes**\n\n---\n### ▶ How to run\n1. Click **Runtime → Run all** (or `Ctrl+F9`)\n2. When Step 2 prompts you, upload your CSV\n3. Edit the 6 settings in **Step 3** (the only cell you need to change)\n4. Wait ~5 minutes — all results appear automatically\n\n> **Tip:** If you only want to re-run from a specific step, click the ▶ button on the left of that cell."
  },
  {
   "cell_type": "markdown",
   "id": "s1md",
   "metadata": {},
   "source": "---\n## ⚙️ Step 1 — Setup\n*Runs automatically. Nothing to change here.*"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s1code",
   "metadata": {},
   "outputs": [],
   "source": "# Install and import everything needed\nimport subprocess, sys\nsubprocess.run([sys.executable, \"-m\", \"pip\", \"install\",\n                \"scikit-learn\", \"seaborn\", \"matplotlib\", \"pandas\", \"numpy\", \"--quiet\"],\n               capture_output=True)\n\nimport pandas as pd\nimport numpy as np\nimport warnings\nwarnings.filterwarnings(\"ignore\")\n\nimport matplotlib\nmatplotlib.use(\"Agg\")\nimport matplotlib.pyplot as plt\nimport matplotlib.patches as mpatches\nimport seaborn as sns\n\nfrom IPython.display import display, HTML\nfrom google.colab import files as colab_files\n\nfrom sklearn.model_selection  import train_test_split, cross_val_score\nfrom sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier\nfrom sklearn.linear_model     import LogisticRegression\nfrom sklearn.preprocessing    import LabelEncoder, StandardScaler\nfrom sklearn.metrics          import (confusion_matrix, roc_auc_score,\n                                      roc_curve, ConfusionMatrixDisplay)\nfrom sklearn.impute           import SimpleImputer\n\nsns.set_theme(style=\"whitegrid\", palette=\"muted\")\nplt.rcParams.update({\"figure.dpi\": 130, \"figure.facecolor\": \"white\"})\nCOLORS = dict(stayed=\"#2E86AB\", churned=\"#E84855\", highlight=\"#F4A261\",\n              neutral=\"#A8DADC\", dark=\"#1D3557\", green=\"#27AE60\")\n\nprint(\"✅  Setup complete — proceed to Step 2.\")"
  },
  {
   "cell_type": "markdown",
   "id": "s2md",
   "metadata": {},
   "source": "---\n## 📤 Step 2 — Upload Your CSV\n*A file picker will appear below. Click it to select your customer data file.*\n\n**Your CSV needs:**\n- A column showing whether each customer churned (`Yes`/`No` or `1`/`0`)\n- At least 3 columns describing customers (tenure, charges, contract type, etc.)\n- At least 200 rows"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s2code",
   "metadata": {},
   "outputs": [],
   "source": "print(\"📂  Click 'Choose Files' below and select your CSV...\")\nuploaded = colab_files.upload()\n\nif not uploaded:\n    raise ValueError(\"No file uploaded. Re-run this cell and select a CSV file.\")\n\nCSV_FILENAME = list(uploaded.keys())[0]\ntry:\n    df_raw = pd.read_csv(CSV_FILENAME, encoding=\"utf-8\")\nexcept UnicodeDecodeError:\n    df_raw = pd.read_csv(CSV_FILENAME, encoding=\"latin-1\")\n\nprint(f\"\\n✅  Loaded: {CSV_FILENAME}\")\nprint(f\"   Rows   : {df_raw.shape[0]:,}\")\nprint(f\"   Columns: {df_raw.shape[1]}\")\nprint(\"\\n📋  Column names in your file:\")\nfor i, col in enumerate(df_raw.columns, 1):\n    sample = str(df_raw[col].dropna().iloc[0]) if df_raw[col].dropna().size > 0 else \"—\"\n    print(f\"   {i:>2}. {col:<30}  (sample value: {sample})\")"
  },
  {
   "cell_type": "markdown",
   "id": "s3md",
   "metadata": {},
   "source": "---\n## ✏️ Step 3 — Configure Settings\n**This is the only cell you need to edit.**\n\n1. Look at the column list printed above\n2. Set `CHURN_COLUMN` to the exact name of your churn column\n3. Adjust the business parameters to match your situation"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s3code",
   "metadata": {},
   "outputs": [],
   "source": "# ════════════════════════════════════════════════════════════════\n#  ✏️  EDIT THESE SETTINGS — then press the ▶ button to run\n# ════════════════════════════════════════════════════════════════\n\nCHURN_COLUMN        = \"Churn\"       # Exact column name for churn (case-sensitive)\nCUSTOMER_ID_COLUMN  = \"customerID\"  # ID column to exclude from model (or None)\n\n# ── Business parameters for the ROI calculator ───────────────────\nCUSTOMER_LTV        = 500   # $ revenue lost when one customer churns\nOFFER_COST          = 30    # $ cost of one retention offer or discount\nRETENTION_RATE      = 0.40  # fraction of contacted at-risk customers who stay (40% = 0.40)\nMONTHLY_CAPACITY    = 100   # how many customers your team can contact per month\nRISK_THRESHOLD      = 0.50  # flag customers with churn probability above this value\n\n# ════════════════════════════════════════════════════════════════\n#  Do not edit below this line\n# ════════════════════════════════════════════════════════════════\nif CHURN_COLUMN not in df_raw.columns:\n    print(f\"❌  Column '{CHURN_COLUMN}' not found.\")\n    print(f\"   Available columns: {list(df_raw.columns)}\")\n    raise KeyError(f\"Set CHURN_COLUMN to one of the column names listed above.\")\n\nPOSITIVE_LABELS = {\"yes\", \"1\", \"true\", \"churned\", \"churn\", \"exited\", \"1.0\"}\ndf_raw[\"__churn\"] = (df_raw[CHURN_COLUMN].astype(str).str.strip().str.lower()\n                     .apply(lambda x: 1 if x in POSITIVE_LABELS else 0))\n\nn_stayed  = (df_raw[\"__churn\"] == 0).sum()\nn_churned = (df_raw[\"__churn\"] == 1).sum()\nchurn_rate = df_raw[\"__churn\"].mean() * 100\n\nprint(f\"✅  Churn column '{CHURN_COLUMN}' recognised.\")\nprint(f\"   Unique values found: {sorted(df_raw[CHURN_COLUMN].dropna().astype(str).unique().tolist())}\")\nprint()\nprint(f\"   Total customers : {len(df_raw):,}\")\nprint(f\"   Stayed          : {n_stayed:,}  ({100 - churn_rate:.1f}%)\")\nprint(f\"   Churned         : {n_churned:,}  ({churn_rate:.1f}%)\")\nprint()\nif churn_rate < 1:\n    print(\"⚠️  Very low churn rate — double-check CHURN_COLUMN.\")\nif len(df_raw) < 200:\n    print(\"⚠️  Small dataset — results may be less reliable.\")\nprint(\"✅  Configuration complete — proceeding to Step 4.\")"
  },
  {
   "cell_type": "markdown",
   "id": "s4md",
   "metadata": {},
   "source": "---\n## 🔍 Step 4 — Explore Your Data\n*Auto-generated overview of data types, missing values, and key statistics.*"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s4code",
   "metadata": {},
   "outputs": [],
   "source": "drop_cols = [\"__churn\", CHURN_COLUMN]\nif CUSTOMER_ID_COLUMN and CUSTOMER_ID_COLUMN in df_raw.columns:\n    drop_cols.append(CUSTOMER_ID_COLUMN)\ndf_explore = df_raw.drop(columns=drop_cols, errors=\"ignore\").copy()\n\nsummary = pd.DataFrame({\n    \"Type\"    : df_explore.dtypes.astype(str),\n    \"Non-Null\": df_explore.count(),\n    \"Missing\" : df_explore.isnull().sum(),\n    \"Missing%\": (df_explore.isnull().mean() * 100).round(1).astype(str) + \"%\",\n    \"Unique\"  : df_explore.nunique(),\n    \"Sample\"  : df_explore.apply(lambda c: str(c.dropna().iloc[0]) if c.dropna().size else \"—\"),\n})\n\nprint(\"COLUMN SUMMARY\")\nprint(\"=\" * 60)\ndisplay(summary)\n\nprint(\"\\nFIRST 5 ROWS\")\nprint(\"=\" * 60)\ndisplay(df_raw.drop(columns=[\"__churn\"], errors=\"ignore\").head())\n\nnum_cols = df_explore.select_dtypes(include=\"number\").columns\nif len(num_cols):\n    print(\"\\nNUMERIC STATISTICS\")\n    print(\"=\" * 60)\n    display(df_explore[num_cols].describe().round(2))"
  },
  {
   "cell_type": "markdown",
   "id": "s5md",
   "metadata": {},
   "source": "---\n## 📊 Step 5 — Visualise Churn Patterns\n*Four auto-generated charts. Nothing to edit.*"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s5code",
   "metadata": {},
   "outputs": [],
   "source": "df_viz = df_raw.drop(columns=drop_cols, errors=\"ignore\").copy()\ndf_viz[\"Churn\"] = df_raw[\"__churn\"].values\nnum_feats = df_viz.select_dtypes(include=\"number\").columns.drop(\"Churn\", errors=\"ignore\").tolist()\ncat_feats  = [c for c in df_viz.select_dtypes(include=\"object\").columns\n              if 2 <= df_viz[c].nunique() <= 12]\n\n# Chart 1 — Churn distribution\nfig, ax = plt.subplots(figsize=(6, 4))\nbars = ax.bar([\"Stayed\", \"Churned\"], [n_stayed, n_churned],\n              color=[COLORS[\"stayed\"], COLORS[\"churned\"]], edgecolor=\"white\", width=0.5)\nfor bar, n in zip(bars, [n_stayed, n_churned]):\n    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df_raw) * 0.01,\n            f\"{n:,}\\n({n / len(df_raw) * 100:.1f}%)\", ha=\"center\", fontweight=\"bold\", fontsize=11)\nax.set_title(\"Chart 1 — Churn Distribution\", fontweight=\"bold\", fontsize=13)\nax.set_ylabel(\"Customers\")\nax.set_ylim(0, max(n_stayed, n_churned) * 1.22)\nsns.despine(); plt.tight_layout(); plt.show()\n\n# Chart 2 — Numeric features by churn\nif num_feats:\n    n = min(len(num_feats), 6)\n    ncols = min(3, n); nrows = -(-n // ncols)\n    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))\n    axf = np.array(axes).flatten() if n > 1 else [axes]\n    for i, col in enumerate(num_feats[:n]):\n        d0 = df_viz.loc[df_viz[\"Churn\"] == 0, col].dropna()\n        d1 = df_viz.loc[df_viz[\"Churn\"] == 1, col].dropna()\n        bp = axf[i].boxplot([d0, d1], labels=[\"Stayed\", \"Churned\"], patch_artist=True)\n        bp[\"boxes\"][0].set_facecolor(COLORS[\"stayed\"])\n        bp[\"boxes\"][1].set_facecolor(COLORS[\"churned\"])\n        for el in [\"whiskers\", \"caps\", \"medians\"]:\n            for item in bp[el]: item.set_color(COLORS[\"dark\"])\n        axf[i].set_title(col, fontweight=\"bold\")\n    for j in range(n, len(axf)): axf[j].set_visible(False)\n    fig.suptitle(\"Chart 2 — Numeric Features: Stayed vs Churned\", fontweight=\"bold\", y=1.01)\n    plt.tight_layout(); plt.show()\n    print(\"📌 Boxes at different heights = that feature predicts churn\\n\")\n\n# Chart 3 — Categorical churn rates\nif cat_feats:\n    n = min(len(cat_feats), 6)\n    ncols = min(3, n); nrows = -(-n // ncols)\n    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))\n    axf = np.array(axes).flatten() if n > 1 else [axes]\n    for i, col in enumerate(cat_feats[:n]):\n        rates = df_viz.groupby(col)[\"Churn\"].mean() * 100\n        rates.sort_values().plot(kind=\"bar\", ax=axf[i], color=COLORS[\"churned\"], edgecolor=\"white\")\n        axf[i].set_title(f\"{col}\\nChurn Rate %\", fontweight=\"bold\", fontsize=10)\n        axf[i].set_ylabel(\"%\"); axf[i].tick_params(axis=\"x\", rotation=30)\n        for p in axf[i].patches:\n            axf[i].annotate(f\"{p.get_height():.1f}%\",\n                            (p.get_x() + p.get_width() / 2, p.get_height()),\n                            ha=\"center\", va=\"bottom\", fontsize=8)\n    for j in range(n, len(axf)): axf[j].set_visible(False)\n    fig.suptitle(\"Chart 3 — Churn Rate by Category\", fontweight=\"bold\", y=1.01)\n    plt.tight_layout(); plt.show()\n    print(\"📌 Taller bars = higher churn rate in that category\\n\")\n\n# Chart 4 — Correlation heatmap\nnum_all = df_viz.select_dtypes(include=\"number\")\nif num_all.shape[1] >= 3:\n    corr = num_all.corr()\n    mask = np.triu(np.ones_like(corr, dtype=bool))\n    sz = min(10, 1.2 * num_all.shape[1])\n    fig, ax = plt.subplots(figsize=(sz, sz * 0.8))\n    sns.heatmap(corr, mask=mask, annot=True, fmt=\".2f\", cmap=\"RdYlBu_r\",\n                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={\"size\": 9})\n    ax.set_title(\"Chart 4 — Correlation Heatmap\", fontweight=\"bold\", fontsize=13)\n    plt.tight_layout(); plt.show()\n    print(\"📌 Check the 'Churn' row — values far from 0 are most related to churn\")"
  },
  {
   "cell_type": "markdown",
   "id": "s6md",
   "metadata": {},
   "source": "---\n## 🧹 Step 6 — Prepare Data for Modelling\n*Encodes text columns, fills missing values, removes useless columns. Runs automatically.*"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s6code",
   "metadata": {},
   "outputs": [],
   "source": "df_model = df_viz.copy()\nlog = []\n\n# Drop columns that add no information\nfor col in [c for c in df_model.columns if c != \"Churn\"]:\n    nu = df_model[col].nunique()\n    if nu <= 1:\n        df_model.drop(columns=[col], inplace=True)\n        log.append(f\"  Dropped '{col}' — constant (no variation)\")\n    elif nu > 0.95 * len(df_model) and df_model[col].dtype == object:\n        df_model.drop(columns=[col], inplace=True)\n        log.append(f\"  Dropped '{col}' — near-unique text (likely ID or free text)\")\n\n# Encode text columns to numbers\nle = LabelEncoder()\nfor col in df_model.select_dtypes(include=\"object\").columns:\n    df_model[col] = le.fit_transform(df_model[col].astype(str).str.strip())\n    log.append(f\"  Encoded  '{col}' — text → numbers\")\n\n# Fill missing values\nfeat_cols = [c for c in df_model.columns if c != \"Churn\"]\ndf_model[feat_cols] = SimpleImputer(strategy=\"median\").fit_transform(df_model[feat_cols])\nlog.append(f\"  Filled all missing values with column medians\")\n\nX = df_model[feat_cols]\ny = df_model[\"Churn\"].values\n\nprint(\"DATA PREPARATION LOG\")\nprint(\"=\" * 55)\nfor line in log: print(line)\nprint(f\"\\n✅  Ready: {X.shape[0]:,} rows  ×  {X.shape[1]} features\")\nprint(f\"   Churn rate in model data: {y.mean() * 100:.1f}%\")\nprint(f\"\\n   Features the model will use:\")\nfor col in X.columns:\n    print(f\"     • {col}\")"
  },
  {
   "cell_type": "markdown",
   "id": "s7md",
   "metadata": {},
   "source": "---\n## 🤖 Step 7 — Train & Compare Three Models\n*Trains Logistic Regression, Random Forest, and Gradient Boosting. Automatically picks the best.*\n\n> ⏱️ Takes 60–90 seconds. Watch the spinning ▶ — wait for ✅ before continuing."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s7code",
   "metadata": {},
   "outputs": [],
   "source": "X_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y)\n\nscaler  = StandardScaler()\nXtr_sc  = scaler.fit_transform(X_train)\nXte_sc  = scaler.transform(X_test)\n\nMODEL_DEFS = {\n    \"Logistic Regression\": LogisticRegression(max_iter=1000, random_state=42,\n                                               class_weight=\"balanced\"),\n    \"Random Forest\"      : RandomForestClassifier(n_estimators=300, random_state=42,\n                                                   class_weight=\"balanced\", n_jobs=-1),\n    \"Gradient Boosting\"  : GradientBoostingClassifier(n_estimators=300, random_state=42),\n}\n\nmodel_results = {}\ntrained_models = {}\n\nprint(f\"  {'Model':<24} {'Accuracy':>10} {'AUC':>8} {'Recall':>9}\")\nprint(f\"  {'─' * 54}\")\n\nfor name, mdl in MODEL_DEFS.items():\n    print(f\"  Training {name}...\", end=\" \", flush=True)\n    Xtr = Xtr_sc if name == \"Logistic Regression\" else X_train\n    Xte = Xte_sc if name == \"Logistic Regression\" else X_test\n    mdl.fit(Xtr, y_train)\n    yp  = mdl.predict(Xte)\n    ypr = mdl.predict_proba(Xte)[:, 1]\n    acc = (yp == y_test).mean() * 100\n    auc = roc_auc_score(y_test, ypr)\n    tn_, fp_, fn_, tp_ = confusion_matrix(y_test, yp).ravel()\n    rec = tp_ / (tp_ + fn_) * 100 if (tp_ + fn_) > 0 else 0\n    model_results[name] = dict(acc=round(acc, 1), auc=round(auc, 3), rec=round(rec, 1))\n    trained_models[name] = dict(model=mdl, yp=yp, ypr=ypr, Xte=Xte)\n    print(f\"done  →  Acc {acc:.1f}%   AUC {auc:.3f}   Recall {rec:.1f}%\")\n\nBEST_MODEL = max(model_results, key=lambda k: model_results[k][\"auc\"])\nprint(f\"\\n🏆  Best model: {BEST_MODEL}  (AUC {model_results[BEST_MODEL]['auc']})\")\n\n# Comparison chart\nfig, axes = plt.subplots(1, 2, figsize=(12, 4))\nnames = list(model_results.keys())\nclrs  = [COLORS[\"stayed\"], COLORS[\"highlight\"], COLORS[\"churned\"]]\nfor i, (metric, label) in enumerate([(\"acc\", \"Accuracy (%)\"), (\"auc\", \"AUC Score\")]):\n    vals = [model_results[m][metric] for m in names]\n    bs   = axes[i].bar(names, vals, color=clrs, edgecolor=\"white\", linewidth=1.5)\n    for b, v in zip(bs, vals):\n        axes[i].text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02,\n                     str(v), ha=\"center\", fontweight=\"bold\", fontsize=11)\n    axes[i].set_title(label, fontweight=\"bold\")\n    axes[i].tick_params(axis=\"x\", rotation=15)\n    sns.despine(ax=axes[i])\nplt.suptitle(\"Chart 5 — Model Comparison\", fontsize=14, fontweight=\"bold\")\nplt.tight_layout(); plt.show()"
  },
  {
   "cell_type": "markdown",
   "id": "s8md",
   "metadata": {},
   "source": "---\n## 🔬 Step 8 — Evaluate the Best Model\n*Confusion matrix, ROC curve, and plain-English performance breakdown.*"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s8code",
   "metadata": {},
   "outputs": [],
   "source": "best_yp  = trained_models[BEST_MODEL][\"yp\"]\nbest_ypr = trained_models[BEST_MODEL][\"ypr\"]\n\ncm_vals  = confusion_matrix(y_test, best_yp)\nTN, FP, FN, TP = cm_vals.ravel()\nRECALL_PCT = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0\nPREC_PCT   = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0\nAUC_VAL    = roc_auc_score(y_test, best_ypr)\n\nprint(f\"BEST MODEL: {BEST_MODEL}\")\nprint(\"=\" * 55)\nprint(f\"  Accuracy  : {model_results[BEST_MODEL]['acc']:.1f}%\")\nprint(f\"  AUC       : {AUC_VAL:.3f}  (0.5 = random, 1.0 = perfect)\")\nprint(f\"  Recall    : {RECALL_PCT:.1f}%  ← % of actual churners the model catches\")\nprint(f\"  Precision : {PREC_PCT:.1f}%  ← % of flagged customers who actually churn\")\nprint()\nprint(\"  CONFUSION MATRIX\")\nprint(f\"    True Negatives  {TN:>5}  — correctly predicted to stay\")\nprint(f\"    False Positives {FP:>5}  — flagged as churner, actually stayed (wasted offer)\")\nprint(f\"    False Negatives {FN:>5}  — predicted stay, actually left  ← ${FN * CUSTOMER_LTV:,.0f} revenue at risk\")\nprint(f\"    True Positives  {TP:>5}  — correctly caught churners\")\n\nfig, axes = plt.subplots(1, 2, figsize=(13, 5))\n\n# Confusion matrix heatmap\nConfusionMatrixDisplay(cm_vals, display_labels=[\"Stayed\", \"Churned\"]).plot(\n    ax=axes[0], cmap=\"Blues\", colorbar=False)\nfor (r_, c_), lbl in [((0,0), f\"Correct\\n{TN}\"), ((0,1), f\"False Alarm\\n{FP}\"),\n                       ((1,0), f\"MISSED\\n{FN}\"),  ((1,1), f\"Caught\\n{TP}\")]:\n    axes[0].text(c_, r_ + 0.35, lbl, ha=\"center\", fontsize=9, color=\"white\", fontweight=\"bold\")\naxes[0].set_title(f\"Chart 6A — Confusion Matrix\\n{BEST_MODEL}\", fontweight=\"bold\")\n\n# ROC curve\nfpr_, tpr_, _ = roc_curve(y_test, best_ypr)\naxes[1].plot(fpr_, tpr_, color=COLORS[\"churned\"], lw=2, label=f\"Model (AUC = {AUC_VAL:.3f})\")\naxes[1].plot([0,1],[0,1], \"k--\", lw=1, label=\"Random (AUC = 0.500)\")\naxes[1].fill_between(fpr_, tpr_, alpha=0.08, color=COLORS[\"churned\"])\naxes[1].set_xlabel(\"False Positive Rate\"); axes[1].set_ylabel(\"True Positive Rate\")\naxes[1].set_title(\"Chart 6B — ROC Curve\", fontweight=\"bold\"); axes[1].legend(loc=\"lower right\")\nplt.tight_layout(); plt.show()\n\nif RECALL_PCT < 30:\n    print(f\"\\n⚠️  Recall is {RECALL_PCT:.1f}% — the model is missing many churners.\")\n    print(f\"   Try lowering RISK_THRESHOLD in Step 3 to 0.30, then re-run from Step 10.\")"
  },
  {
   "cell_type": "markdown",
   "id": "s9md",
   "metadata": {},
   "source": "---\n## 🔎 Step 9 — What Drives Churn?\n*Feature importance — which factors most influence the model's predictions.*"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s9code",
   "metadata": {},
   "outputs": [],
   "source": "best_mdl = trained_models[BEST_MODEL][\"model\"]\nif hasattr(best_mdl, \"feature_importances_\"):\n    imps     = best_mdl.feature_importances_\n    imp_type = \"Gini Importance\"\nelse:\n    imps     = np.abs(best_mdl.coef_[0])\n    imp_type = \"|Coefficient|\"\n\nfeat_imp = pd.Series(imps, index=X.columns).sort_values(ascending=False)\ntop_n    = min(15, len(feat_imp))\ntop_feat = feat_imp.head(top_n)\n\nfig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))\nbar_colors = [COLORS[\"churned\"] if i < 3 else COLORS[\"highlight\"] if i < 6 else COLORS[\"neutral\"]\n              for i in range(top_n)]\nax.barh(top_feat.index[::-1], top_feat.values[::-1], color=bar_colors[::-1], edgecolor=\"white\")\nfor bar in ax.patches:\n    ax.text(bar.get_width() + top_feat.max() * 0.01,\n            bar.get_y() + bar.get_height() / 2,\n            f\"{bar.get_width():.4f}\", va=\"center\", fontsize=8)\nax.legend(handles=[\n    mpatches.Patch(color=COLORS[\"churned\"],   label=\"Top 3 drivers\"),\n    mpatches.Patch(color=COLORS[\"highlight\"], label=\"Rank 4–6\"),\n    mpatches.Patch(color=COLORS[\"neutral\"],   label=\"Other\"),\n], loc=\"lower right\", fontsize=9)\nax.set_xlabel(imp_type)\nax.set_title(f\"Chart 7 — Feature Importance ({BEST_MODEL})\", fontweight=\"bold\", fontsize=13)\nsns.despine(); plt.tight_layout(); plt.show()\n\nprint(\"\\nFEATURE IMPORTANCE RANKING\")\nprint(f\"  {'Rank':<6} {'Feature':<35} {'Score':>10}\")\nprint(\"  \" + \"─\" * 53)\nfor rank, (feat, val) in enumerate(feat_imp.items(), 1):\n    tag = \"  ← #1 driver — design your retention offer around this\" if rank == 1 else \"\"\n    print(f\"  {rank:<6} {feat:<35} {val:>10.4f}{tag}\")"
  },
  {
   "cell_type": "markdown",
   "id": "s10md",
   "metadata": {},
   "source": "---\n## 🎯 Step 10 — Score & Rank Every Customer\n*Every customer in the test set gets a 0–100% churn probability. Highest-risk listed first.*"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s10code",
   "metadata": {},
   "outputs": [],
   "source": "risk_scores = X_test.copy().reset_index(drop=True)\nrisk_scores[\"Churn_Probability_%\"] = (best_ypr * 100).round(1)\nrisk_scores[\"Churn_Flag\"]          = (best_ypr >= RISK_THRESHOLD).astype(int)\nrisk_scores[\"Actual_Churn\"]        = y_test\nrisk_scores[\"Risk_Tier\"] = pd.cut(\n    risk_scores[\"Churn_Probability_%\"],\n    bins=[0, 30, 55, 75, 101],\n    labels=[\"🟢 Low (<30%)\", \"🟡 Medium (30–55%)\", \"🟠 High (55–75%)\", \"🔴 Critical (>75%)\"])\n\n# Re-attach customer IDs if available\nif CUSTOMER_ID_COLUMN and CUSTOMER_ID_COLUMN in df_raw.columns:\n    all_idx = np.arange(len(df_raw))\n    _, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42,\n                                    stratify=df_raw[\"__churn\"].values)\n    risk_scores.insert(0, CUSTOMER_ID_COLUMN, df_raw[CUSTOMER_ID_COLUMN].values[test_idx])\n\nrisk_scores = risk_scores.sort_values(\"Churn_Probability_%\", ascending=False)\nrisk_scores.insert(0, \"Rank\", range(1, len(risk_scores) + 1))\n\n# Tier summary table\ntier_summary = risk_scores.groupby(\"Risk_Tier\", observed=True).agg(\n    Customers       = (\"Churn_Probability_%\", \"count\"),\n    Actual_Churners = (\"Actual_Churn\",        \"sum\"),\n    Avg_Prob        = (\"Churn_Probability_%\",  \"mean\"),\n).round(1)\ntier_summary[\"Actual_Churn_Rate_%\"] = (\n    tier_summary[\"Actual_Churners\"] / tier_summary[\"Customers\"] * 100).round(1)\n\nprint(\"RISK TIER SUMMARY\"); print(\"=\" * 55)\ndisplay(tier_summary)\n\nprint(\"\\nTOP 20 HIGHEST-RISK CUSTOMERS\"); print(\"=\" * 55)\nid_cols = ([CUSTOMER_ID_COLUMN] if CUSTOMER_ID_COLUMN and CUSTOMER_ID_COLUMN in risk_scores.columns else [])\ndisplay(risk_scores[[\"Rank\"] + id_cols + [\"Churn_Probability_%\", \"Risk_Tier\", \"Actual_Churn\"]].head(20))\n\n# Charts\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\nd0 = risk_scores.loc[risk_scores[\"Actual_Churn\"] == 0, \"Churn_Probability_%\"]\nd1 = risk_scores.loc[risk_scores[\"Actual_Churn\"] == 1, \"Churn_Probability_%\"]\naxes[0].hist(d0, bins=20, alpha=0.6, color=COLORS[\"stayed\"],  label=\"Stayed\",  edgecolor=\"white\")\naxes[0].hist(d1, bins=20, alpha=0.6, color=COLORS[\"churned\"], label=\"Churned\", edgecolor=\"white\")\naxes[0].set_xlabel(\"Predicted Churn Probability (%)\"); axes[0].set_ylabel(\"Customers\")\naxes[0].set_title(\"Chart 8A — Score Distribution\", fontweight=\"bold\"); axes[0].legend()\nsns.despine(ax=axes[0])\n\ntc = risk_scores[\"Risk_Tier\"].value_counts().sort_index()\naxes[1].bar(range(len(tc)), tc.values,\n            color=[COLORS[\"stayed\"], COLORS[\"highlight\"], \"#E07C24\", COLORS[\"churned\"]],\n            edgecolor=\"white\")\naxes[1].set_xticks(range(len(tc))); axes[1].set_xticklabels(tc.index, rotation=20, ha=\"right\")\nfor b, cnt in zip(axes[1].patches, tc.values):\n    axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,\n                 str(cnt), ha=\"center\", fontweight=\"bold\")\naxes[1].set_ylabel(\"Customers\"); axes[1].set_title(\"Chart 8B — Customers by Risk Tier\", fontweight=\"bold\")\nsns.despine(ax=axes[1])\nplt.suptitle(\"Chart 8 — Customer Risk Profile\", fontsize=14, fontweight=\"bold\")\nplt.tight_layout(); plt.show()\n\nn_critical = (risk_scores[\"Risk_Tier\"] == \"🔴 Critical (>75%)\").sum()\nprint(f\"\\n⚡  Critical-risk customers: {n_critical} — contact these first.\")"
  },
  {
   "cell_type": "markdown",
   "id": "s11md",
   "metadata": {},
   "source": "---\n## 💰 Step 11 — Business ROI Calculator\n*Compares the dollar value of three outreach strategies.*\n\n> ✏️ Adjust `CUSTOMER_LTV`, `OFFER_COST`, `RETENTION_RATE`, and `MONTHLY_CAPACITY` in **Step 3** to match your business."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s11code",
   "metadata": {},
   "outputs": [],
   "source": "total_churners_test = int(y_test.sum())\nrev_no_action       = total_churners_test * CUSTOMER_LTV\n\n# Scenario B — random outreach\nrand_hits   = int(MONTHLY_CAPACITY * y_test.mean())\nrand_saved  = int(rand_hits * RETENTION_RATE)\nrand_net    = rand_saved * CUSTOMER_LTV - MONTHLY_CAPACITY * OFFER_COST\n\n# Scenario C — ML-guided outreach (contact top-ranked customers)\ntop_contacts = risk_scores.head(min(MONTHLY_CAPACITY, len(risk_scores)))\nml_hits      = int(top_contacts[\"Actual_Churn\"].sum())\nml_saved     = int(ml_hits * RETENTION_RATE)\nml_net       = ml_saved * CUSTOMER_LTV - len(top_contacts) * OFFER_COST\nml_lift      = ml_net - rand_net\nml_efficiency = ml_hits / max(rand_hits, 1)\n\nprint(\"=\" * 65)\nprint(\"  BUSINESS ROI ANALYSIS\")\nprint(\"=\" * 65)\nprint(f\"  Parameters used\")\nprint(f\"    Customer LTV          : ${CUSTOMER_LTV:,.0f}\")\nprint(f\"    Cost per offer        : ${OFFER_COST:,.0f}\")\nprint(f\"    Retention success     : {RETENTION_RATE * 100:.0f}%\")\nprint(f\"    Monthly capacity      : {MONTHLY_CAPACITY} contacts\")\nprint()\nprint(f\"  Scenario A — No action\")\nprint(f\"    Churners lost         : {total_churners_test}\")\nprint(f\"    Revenue at risk       : -${rev_no_action:,.0f}\")\nprint()\nprint(f\"  Scenario B — Random outreach  ({MONTHLY_CAPACITY} contacts)\")\nprint(f\"    Churners in list      : {rand_hits}\")\nprint(f\"    Customers saved       : {rand_saved}\")\nprint(f\"    Net value             : ${rand_net:,.0f}\")\nprint()\nprint(f\"  Scenario C — ML-guided outreach  ({MONTHLY_CAPACITY} contacts)\")\nprint(f\"    Churners in list      : {ml_hits}  ({ml_efficiency:.1f}x more than random)\")\nprint(f\"    Customers saved       : {ml_saved}\")\nprint(f\"    Net value             : ${ml_net:,.0f}\")\nprint()\nprint(f\"  ══════════════════════════════════════\")\nprint(f\"  🚀  ML advantage over random : +${ml_lift:,.0f}\")\nprint(\"=\" * 65)\n\n# ROI chart\nfig, ax = plt.subplots(figsize=(9, 5))\nscenario_vals  = [-rev_no_action, rand_net, ml_net]\nscenario_names = [\"No Action\", \"Random Outreach\", \"ML-Guided\"]\nbar_clrs       = [COLORS[\"churned\"], COLORS[\"neutral\"], COLORS[\"stayed\"]]\nbs = ax.bar(scenario_names, scenario_vals, color=bar_clrs, edgecolor=\"white\", width=0.5)\nax.axhline(0, color=\"black\", lw=0.8, linestyle=\"--\")\nspan = max(abs(v) for v in scenario_vals)\nfor b, v in zip(bs, scenario_vals):\n    ax.text(b.get_x() + b.get_width() / 2,\n            b.get_height() + (span * 0.03 if v >= 0 else -span * 0.11),\n            f\"${v:,.0f}\", ha=\"center\", fontweight=\"bold\", fontsize=12)\nax.set_ylabel(\"Net Financial Impact ($)\")\nax.set_title(\n    f\"Chart 9 — Business ROI  \"\n    f\"(LTV=${CUSTOMER_LTV:,} | Offer=${OFFER_COST} | \"\n    f\"Success={RETENTION_RATE*100:.0f}% | Cap={MONTHLY_CAPACITY})\",\n    fontweight=\"bold\", fontsize=12)\nsns.despine(); plt.tight_layout(); plt.show()"
  },
  {
   "cell_type": "markdown",
   "id": "s12md",
   "metadata": {},
   "source": "---\n## 📥 Step 12 — Download Your Results\n*Three files download automatically. Check your browser's Downloads folder.*"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "s12code",
   "metadata": {},
   "outputs": [],
   "source": "from datetime import datetime\ndate_tag = datetime.now().strftime(\"%Y%m%d\")\n\n# ── File 1: Full risk scores ─────────────────────────────────────────────────\nscores_file = f\"{date_tag}_churn_risk_scores.csv\"\nrisk_scores.to_csv(scores_file, index=False)\ncolab_files.download(scores_file)\nprint(f\"✅  Downloaded: {scores_file}  ({len(risk_scores):,} customers ranked by risk)\")\n\n# ── File 2: High-risk action list ────────────────────────────────────────────\naction_file = f\"{date_tag}_high_risk_action_list.csv\"\nhigh_risk = risk_scores[risk_scores[\"Churn_Flag\"] == 1].copy()\nhigh_risk.to_csv(action_file, index=False)\ncolab_files.download(action_file)\nprint(f\"✅  Downloaded: {action_file}  ({len(high_risk)} customers above {RISK_THRESHOLD*100:.0f}% threshold)\")\n\n# ── File 3: Executive summary ────────────────────────────────────────────────\nsummary_file = f\"{date_tag}_executive_summary.txt\"\ntier_counts = risk_scores[\"Risk_Tier\"].value_counts().sort_index()\nlines = [\n    \"=\" * 65,\n    \"  CHURNSCOPE — EXECUTIVE SUMMARY\",\n    f\"  Run date : {datetime.now().strftime('%Y-%m-%d %H:%M')}\",\n    f\"  Dataset  : {CSV_FILENAME}\",\n    \"=\" * 65, \"\",\n    \"DATASET\",\n    f\"  Total customers : {n_stayed + n_churned:,}\",\n    f\"  Stayed          : {n_stayed:,}  ({100 - churn_rate:.1f}%)\",\n    f\"  Churned         : {n_churned:,}  ({churn_rate:.1f}%)\", \"\",\n    \"BEST MODEL\",\n    f\"  {BEST_MODEL}\",\n    f\"  AUC       : {AUC_VAL:.3f}\",\n    f\"  Accuracy  : {model_results[BEST_MODEL]['acc']:.1f}%\",\n    f\"  Recall    : {RECALL_PCT:.1f}%\",\n    f\"  Precision : {PREC_PCT:.1f}%\", \"\",\n    \"CONFUSION MATRIX\",\n    f\"  True Negatives  : {TN}  (correctly predicted to stay)\",\n    f\"  False Positives : {FP}  (flagged — wasted offers)\",\n    f\"  False Negatives : {FN}  (missed churners — ${FN * CUSTOMER_LTV:,.0f} revenue at risk)\",\n    f\"  True Positives  : {TP}  (correctly caught)\", \"\",\n    \"TOP 5 CHURN DRIVERS\",\n] + [f\"  {i+1}. {f}  ({v:.4f})\" for i, (f, v) in enumerate(feat_imp.head(5).items())] + [\n    \"\",\n    \"RISK TIERS\",\n] + [f\"  {t}: {c}\" for t, c in tier_counts.items()] + [\n    \"\",\n    \"BUSINESS ROI\",\n    f\"  No action            : -${rev_no_action:,.0f}\",\n    f\"  Random outreach      :  ${rand_net:,.0f}\",\n    f\"  ML-guided outreach   :  ${ml_net:,.0f}\",\n    f\"  ML advantage         :  +${ml_lift:,.0f}  ({ml_efficiency:.1f}x efficiency)\",\n    \"\",\n    \"=\" * 65,\n]\nwith open(summary_file, \"w\") as f:\n    f.write(\"\\n\".join(lines))\ncolab_files.download(summary_file)\nprint(f\"✅  Downloaded: {summary_file}\")\nprint()\nprint(\"\\n\".join(lines))"
  },
  {
   "cell_type": "markdown",
   "id": "donecell",
   "metadata": {},
   "source": "---\n## ✅ Analysis Complete!\n\n| File downloaded | What to do with it |\n|---|---|\n| `YYYYMMDD_churn_risk_scores.csv` | Full ranked list — every customer with their churn score |\n| `YYYYMMDD_high_risk_action_list.csv` | Give this to your retention team — sorted by highest risk |\n| `YYYYMMDD_executive_summary.txt` | Paste key numbers into a management presentation |\n\n---\n### 📌 Recommended next steps\n\n1. **Open `high_risk_action_list.csv`** — sort by `Churn_Probability_%` descending and start with Critical-tier customers\n2. **Act on the top churn driver** (Step 9 chart) — design your retention offer specifically around it\n3. **Re-run monthly** — upload a fresh CSV each month; the model retrains automatically\n4. **Tune the threshold** — if recall is low, lower `RISK_THRESHOLD` in Step 3 to catch more churners\n5. **Update ROI parameters** — after your first campaign, update `RETENTION_RATE` with your actual results\n\n---\n*ChurnScope Business Edition · Google Colab · Compatible with MIS204_Churn_Dataset.csv*"
  }
 ]
}
[ChurnScope_Business (1).ipynb](https://github.com/user-attachments/files/25412651/ChurnScope_Business.1.ipynb)

