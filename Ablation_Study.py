#############################################################################################
#Spring 2026
#COSC 702: Advanced AI-Driven Software Engineering
#Project: Benchmarking the Detection of Cloud-based User Authentication Log Anomalies Using Machine Learning Techniques
#Code Component: Ablation Study: Feature Importance Analysis for Anomaly Detection
#Dataset: 13k Datasets (Labeled & Unlabeled)
#Submitted to: Dr. Jamal Bentahar
#Done by: Aisha AlQahtani, & Salwa Mohammed Razaulla
#############################################################################################

#############################################################################################
#Part 0: Configuration
#############################################################################################

#Importing the configuration file containing library imports, file paths, feature columns, model lists and colors, & shared helper functions,

from config import *

#############################################################################################
#Part 1: Ablation-Study-Specific Text Pre-processing
#############################################################################################

def combine_and_vectorize(df, columns, max_features=500):
    """Combine multiple feature columns and vectorize with TF-IDF."""
    valid_cols = [c for c in columns if c in df.columns] #Only use columns that actually exist in the dataframe.
    if not valid_cols:
        raise ValueError(f"None of the requested columns {columns} exist in the dataframe.")
    #Fills nulls with empty string, converts all values to string, then joins.
    #each row's values with a space, e.g. "Failure Moscow 45.33.32.156 AzureAD".

    combined   = df[valid_cols].fillna('').astype(str).apply(
        lambda row: ' '.join(row.values), axis=1
    )
    cleaned    = combined.apply(preprocess_text) #Apply text cleaning to every combined row.
    vectorizer = TfidfVectorizer(max_features=max_features) #Limits vocabulary to top 500 most informative words.
    matrix     = vectorizer.fit_transform(cleaned) #Fit vocabulary and transform text into a numerical sparse matrix.
    return matrix, vectorizer #Both returned, matrix for training, vectorizer to inspect feature names in Experiment 2.


#############################################################################################
#Part 2: Model Training Helpers
#############################################################################################

#Important Note: These functions are kept here and NOT in config.py because they are
#ablation-specific, they train fresh models per feature combination without
#saving .pkl files, unlike Main_AD_Code.py which trains the final production models.

def train_unsupervised(tfidf_matrix, model_name):
    """Train a single unsupervised model and return predictions + scores."""
    reduced = get_reduced(tfidf_matrix) #Compress TF-IDF matrix into 50 dense dimensions once, shared by Elliptic Envelope and One-Class SVM.

    if model_name == 'Isolation Forest':
        m = IsolationForest(contamination=0.3, random_state=42) #Contamination=0.3 expects ~30% of records to be attacks.
        m.fit(tfidf_matrix)
        preds  = m.predict(tfidf_matrix) #Returns -1 (anomaly) or 1 (normal).
        scores = m.decision_function(tfidf_matrix) #More negative = more anomalous.

    elif model_name == 'K-Means Clustering':
        m = KMeans(n_clusters=2, random_state=42, n_init=10) #2 clusters = normal vs attack.
        m.fit(tfidf_matrix)
        _, distances = pairwise_distances_argmin_min(tfidf_matrix, m.cluster_centers_) #Memory-efficient distance to nearest cluster centre.
        threshold = np.percentile(distances, 70) #Top 30% most distant points flagged as anomalies.
        preds  = np.where(distances >= threshold, -1, 1) #-1 = anomaly, 1 = normal.
        scores = -distances #Negate so higher score = more anomalous, consistent with other models.

    elif model_name == 'Elliptic Envelope':
        m = EllipticEnvelope(contamination=0.3, support_fraction=0.99, random_state=42)  #support_fraction=0.9 prevents singular covariance matrix errors.
        m.fit(reduced) #Uses SVD-reduced matrix — Elliptic Envelope requires dense input.
        preds  = m.predict(reduced)
        scores = m.decision_function(reduced)

    elif model_name == 'Local Outlier Factor':
        m = LocalOutlierFactor(n_neighbors=20, contamination=0.3, novelty=True) #novelty=True enables predict() on new unseen data.
        m.fit(tfidf_matrix)
        preds  = m.predict(tfidf_matrix)
        scores = m.decision_function(tfidf_matrix)

    elif model_name == 'One-Class SVM':
        m = OneClassSVM(nu=0.3, kernel='rbf', gamma='scale') #nu=0.3 equivalent to contamination, gamma='scale' auto-adjusts to the feature space.
        m.fit(reduced) #Uses SVD-reduced matrix — One-Class SVM requires dense input.
        preds  = m.predict(reduced)
        scores = m.decision_function(reduced)

    else:
        raise ValueError(f"Unknown unsupervised model: {model_name}") #Catches any typos in model names.

   #No .pkl saving here; ablation trains temporary models per experiment, not production models.
    return preds, scores


def train_supervised(tfidf_matrix, y, model_name):
    """Train a single supervised model with 80/20 split."""
    #stratify=y preserves the 79/21 Normal/Anomaly ratio in both train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_name == 'Random Forest':
        m = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') #class_weight='balanced' corrects for the 79/21 label imbalance.
    elif model_name == 'SVM':
        m = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42) #probability=True enables predict_proba() for metric computation.
    elif model_name == 'Logistic Regression':
        m = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42) #max_iter=1000 ensures convergence on high-dimensional TF-IDF space.
    elif model_name == 'Gradient Boosting':
        m = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42) #learning_rate=0.1 controls how aggressively each tree corrects the previous.
    elif model_name == 'KNN':
        m = KNeighborsClassifier(n_neighbors=5) #Classifies by majority vote of 5 nearest neighbours.
    else:
        raise ValueError(f"Unknown supervised model: {model_name}")

    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    proba = m.predict_proba(X_test)[:, 1] if hasattr(m, 'predict_proba') else preds #Probability of being an attack, falls back to binary if not supported.

    #Returns metrics dict and trained model object, model is discarded after evaluation.
    return compute_metrics_sup(y_test, preds, proba), m


#############################################################################################
#Part 3: Metric Helpers
#############################################################################################

#Important Note: This function is kept in Ablation_Study.py and NOT in config.py because it is
# specific to the ablation study's cybersecurity interpretation logic, and it has no use
# in other components and would add unnecessary clutter to config.py.

def interpret_feature_importance(removed_col, f1_drop):
    """
    Translate F1 drop into a brute force cybersecurity finding.
    """
    #Severity thresholds: based on how much the F1 score drops.
    #when this feature column is removed from the model input.
    if f1_drop > 0.3:
        severity = "CRITICAL"
        meaning  = f"Removing '{removed_col}' severely degrades brute force detection"
    elif f1_drop > 0.1:
        severity = "HIGH"
        meaning  = f"'{removed_col}' is a strong secondary brute force indicator"
    elif f1_drop > 0.02:
        severity = "MODERATE"
        meaning  = f"'{removed_col}' adds useful context beyond login status alone"
    elif f1_drop > 0:
        severity = "LOW"
        meaning  = f"'{removed_col}' provides marginal improvement for brute force detection"
    else:
        severity = "MINIMAL"
        meaning  = f"'{removed_col}' has little independent impact on brute force detection"

    #Pull the cybersecurity role of this column from FEATURE_CONTEXT in config.py.
    #e.g. 'Status' → 'Primary brute force indicator, repeated failures signal attack'
    #Returns empty string if the column isn't in FEATURE_CONTEXT.
    context = FEATURE_CONTEXT.get(removed_col, '')
    return f"[{severity}] {meaning} — {context}" #e.g. [CRITICAL] Removing 'Status' severely degrades brute force detection, Primary brute force indicator.


#############################################################################################
#Part 4: Experiment 1: Feature Column Ablation (Leave-One-Out)
#############################################################################################

def run_column_ablation(df_labeled, feature_columns):
    """
    Leave-One-Out Column Ablation for brute force detection.

    Tests:
    - ALL features combined (baseline)
    - Remove one feature at a time (leave-one-out)
    - Each feature in isolation (single-feature)

    The F1 drop when a feature is removed = its importance
    for brute force attack detection.
    """
    all_results = []  #Collects results from every experiment as a list of dicts, converted to DataFrame at the end.

    #Build the full experiment list — 3 types of experiments:
    # 1. Baseline: all 4 columns together.
    # 2. Leave-one-out: remove one column at a time (4 sub-experiments).
    # 3. Single-feature: only one column at a time (4 sub-experiments)
    # Total = 9 experiments × 10 models = 90 rows in the results DataFrame.
    experiments = {'ALL (Baseline)': feature_columns}
    for col in feature_columns:
        remaining = [c for c in feature_columns if c != col]  #All columns except the one being removed.
        experiments[f'Without {col}'] = remaining
    for col in feature_columns:
        experiments[f'Only {col}'] = [col]  #Just one column in isolation.

    total = len(experiments)
    for idx, (exp_name, cols) in enumerate(experiments.items(), 1):  #Loop through each experiment.
        print(f"\n[{idx}/{total}] Experiment: '{exp_name}'")
        print(f"       Columns used: {cols}")

        #For single-feature experiments, print the cybersecurity role of that column so it's clear what signal we're testing in isolation.
        if exp_name.startswith('Only '):
            col = exp_name.replace('Only ', '')
            print(f"       Context: {FEATURE_CONTEXT.get(col, '')}")
        print("  " + "─" * 53)

        try:
            #Fit a NEW TF-IDF vectorizer on just the selected columns for this experiment.
            #This is intentional — each feature combination produces a different vocabulary.
            matrix, _ = combine_and_vectorize(df_labeled, cols)
        except ValueError as e:
            print(f"    [!] Skipping — {e}")
            continue  #Skip this experiment if the columns don't exist and move to the next one.

        true_unsup = df_labeled['label_unsup'].values  #Ground truth in -1/1 format for unsupervised models.
        true_sup   = df_labeled['label_sup'].values    #Ground truth in 0/1 format for supervised models.

        #Unsupervised
        for model_name in UNSUPERVISED_MODELS:
            try:
                preds, scores = train_unsupervised(matrix, model_name)  #Train and get predictions.
                metrics = compute_metrics_unsup(true_unsup, preds, scores)  #Evaluate against ground truth.
                all_results.append({
                    'experiment':   exp_name,
                    'columns_used': ', '.join(cols),  #Record which columns were used for this experiment.
                    'model':        model_name,
                    'type':         'Unsupervised',
                    **metrics  #Unpack the metrics dictionary directly into this row.
                })
                print(f"    [SUCCESS] {model_name:<25} F1={metrics['f1']:.4f}  AUC={metrics['auc_roc']:.4f}")
            except Exception as e:
                print(f"    [FAILURE] {model_name} failed: {e}")  #Log the failure but continue with remaining models.

        #Supervised
        for model_name in SUPERVISED_MODELS:
            try:
                metrics, _ = train_supervised(matrix, true_sup, model_name)  # _ discards the model object, not needed here.
                all_results.append({
                    'experiment':   exp_name,
                    'columns_used': ', '.join(cols),
                    'model':        model_name,
                    'type':         'Supervised',
                    **metrics
                })
                print(f"    [SUCCESS] {model_name:<25} F1={metrics['f1']:.4f}  AUC={metrics['auc_roc']:.4f}")
            except Exception as e:
                print(f"    [FAILURE] {model_name} failed: {e}")

    return pd.DataFrame(all_results)  #Convert list of dicts into a DataFrame for plotting and saving to CSV.


#############################################################################################
#Part 5: Experiment 2: TF-IDF Term Importance
#############################################################################################

def run_term_importance(df_labeled, feature_columns, top_n=20):
    """
    Identifies which specific terms are most predictive of
    brute force attacks using Random Forest + Logistic Regression.

    In the brute force context:
    - High-importance terms = strongest attack indicators
    - Positive LR coefficients = anomaly-indicative terms
    - Negative LR coefficients = normal behavior terms
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: TF-IDF Term Importance")
    print("Context: Which specific terms best identify brute force attacks?")
    print("=" * 60)

    #Vectorize all 4 feature columns combined, gives the full vocabulary including location names, IP fragments, application names and status words.
    matrix, vectorizer = combine_and_vectorize(df_labeled, feature_columns, max_features=500)
    y = df_labeled['label_sup'].values  #Supervised labels (0/1) needed for both RF and LR.

    #80/20 stratified split, same approach as supervised models in Main_AD_Code.py.
    X_train, X_test, y_train, y_test = train_test_split(
        matrix, y, test_size=0.2, random_state=42, stratify=y
    )

    #Random Forest importance
    #n_estimators=200 (more trees than usual) for more stable feature importance scores.
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    feature_names = vectorizer.get_feature_names_out()  #The actual word/term for each TF-IDF column.
    importances   = rf.feature_importances_             #Score per term — how much it reduces impurity across all trees.
    indices       = np.argsort(importances)[::-1][:top_n]  #Sort descending and take top N indices.

    #Build a clean DataFrame of the top N terms and their importance scores.
    importance_df = pd.DataFrame({
        'term':       feature_names[indices],
        'importance': importances[indices]
    })
    print(f"\n  Top {top_n} most important terms for brute force detection (Random Forest):")
    print(f"  {'─'*45}")
    for _, row in importance_df.iterrows():
        bar = '█' * int(row['importance'] * 1000)  # Visual bar scaled by importance score
        print(f"  {row['term']:<20} {row['importance']:.4f}  {bar}")

    #Logistic Regression coefficients
    #LR gives directional importance — positive coefficient = pushes towards attack, negative coefficient = pushes towards normal. RF only gives magnitude, not direction.
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    lr_coefs       = lr.coef_[0]                        #coef_[0] because binary classification has one row of coefficients.
    lr_indices_pos = np.argsort(lr_coefs)[::-1][:top_n] #Top N most positive — strongest attack indicators.
    lr_indices_neg = np.argsort(lr_coefs)[:top_n]       #Top N most negative — strongest normal indicators.

    #Combine attack and normal indicators into one DataFrame with a direction column.
    lr_importance_df = pd.DataFrame({
        'term':        np.concatenate([feature_names[lr_indices_pos], feature_names[lr_indices_neg]]),
        'coefficient': np.concatenate([lr_coefs[lr_indices_pos], lr_coefs[lr_indices_neg]]),
        'direction':   ['Anomaly'] * top_n + ['Normal'] * top_n  #Label each term as attack or normal indicator.
    })

    print(f"\n  Top {top_n} brute force attack indicators (Logistic Regression):")
    print(f"  {'─'*45}")
    for _, row in lr_importance_df[lr_importance_df['direction'] == 'Anomaly'].iterrows():
        print(f"  {row['term']:<20} coef={row['coefficient']:+.4f}  → ATTACK indicator")  # + in :+.4f forces sign display.

    print(f"\n  Top {top_n} normal behavior indicators (Logistic Regression):")
    print(f"  {'─'*45}")
    for _, row in lr_importance_df[lr_importance_df['direction'] == 'Normal'].iterrows():
        print(f"  {row['term']:<20} coef={row['coefficient']:+.4f}  → NORMAL indicator")

    return importance_df, lr_importance_df  #Both returned for plotting and saving to CSV in main().


#############################################################################################
#Part 6: Experiment 3: Contamination Rate Sensitivity
#############################################################################################

def run_contamination_sensitivity(df_labeled):
    """
    Tests how sensitive unsupervised models are to the assumed
    contamination rate — i.e. how much of the data we assume
    is brute force attack traffic.

    In the brute force context:
    - Low contamination (0.1) = assume 10% of traffic is attack
    - High contamination (0.5) = assume 50% of traffic is attack
    - The actual rate in this dataset is ~20.6%
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Contamination Rate Sensitivity")
    print("Context: How does the assumed attack rate affect detection?")
    print(f"Note: Actual brute force rate in dataset = ~20.6%")
    print("=" * 60)

    #Test 9 contamination rates from 10% to 50% in steps of 5%.
    #This covers under-estimating (0.1) and over-estimating (0.5) the actual attack rate.
    contamination_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    #Only testing the 3 unsupervised models that use a contamination parameter.
    #K-Means uses a distance threshold instead, so it's excluded here.
    models_to_test = ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM']

    #Vectorize once using all 4 feature columns, same input for all contamination rates.
    matrix, _  = combine_and_vectorize(df_labeled, ALL_FEATURE_COLUMNS)
    reduced    = get_reduced(matrix)               # SVD reduction for One-Class SVM
    true_unsup = df_labeled['label_unsup'].values  # Ground truth in -1/1 format

    results = []
    for rate in contamination_rates:
        #Add a marker in the console output when we're near the actual attack rate (~20.6%) so it's easy to spot which contamination value aligns with reality.
        marker = " ← actual brute force rate" if abs(rate - 0.206) < 0.03 else ""
        print(f"\n  Contamination = {rate:.2f}{marker}")

        for model_name in models_to_test:
            try:
                if model_name == 'Isolation Forest':
                    m = IsolationForest(contamination=rate, random_state=42)  #Rate passed directly as contamination.
                    m.fit(matrix)
                    preds  = m.predict(matrix)
                    scores = m.decision_function(matrix)

                elif model_name == 'Local Outlier Factor':
                    m = LocalOutlierFactor(n_neighbors=20, contamination=rate, novelty=True)  #novelty=True required for predict().
                    m.fit(matrix)
                    preds  = m.predict(matrix)
                    scores = m.decision_function(matrix)

                elif model_name == 'One-Class SVM':
                    m = OneClassSVM(nu=rate, kernel='rbf', gamma='scale')  #nu is the One-Class SVM equivalent of contamination.
                    m.fit(reduced)   #Uses SVD-reduced matrix, One-Class SVM requires dense input.
                    preds  = m.predict(reduced)
                    scores = m.decision_function(reduced)

                metrics = compute_metrics_unsup(true_unsup, preds, scores)
                results.append({
                    'contamination':  rate,
                    'model':          model_name,
                    'is_actual_rate': abs(rate - 0.206) < 0.03,  #Flag rows where rate ≈ actual attack rate; used for chart annotation.
                    **metrics  #Unpack all 5 metrics directly into the row.
                })
                print(f"    [SUCCESS] {model_name:<25} F1={metrics['f1']:.4f}  Recall={metrics['recall']:.4f}")

            except Exception as e:
                print(f"    [FAILURE] {model_name} @ {rate}: {e}")  #Log failure but continue with remaining models and rates.

    return pd.DataFrame(results)  #Converted to DataFrame for plotting in plot_contamination_sensitivity().


#############################################################################################
#Part 7: Plotting
#############################################################################################

def plot_column_ablation(ablation_df):
    """
    Plot 1: F1 heatmap — models × feature experiments
    Plot 2: Average F1 per experiment
    Plot 3: F1 drop heatmap — feature importance for brute force detection
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)  #Create outputs folder if it doesn't exist.

    #Plot 1: F1 Heatmap
    #Pivot the results DataFrame so rows = models, columns = experiments, values = F1 score.
    #This gives an at-a-glance view of how every model performs across every feature combination.
    pivot = ablation_df.pivot_table(
        index='model', columns='experiment', values='f1', aggfunc='mean'
    )
    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 1.4), 7))  #Width scales with number of experiments.
    sns.heatmap(
        pivot, annot=True, fmt='.3f', cmap='RdYlGn',  #Red = low F1, green = high F1.
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'F1-Score (Brute Force Detection)'},
        ax=ax, vmin=0, vmax=1  #Fix scale to 0-1 so colours are consistent.
    )
    ax.set_title('Ablation Study — Brute Force Detection F1-Score\n(Models × Feature Combinations)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Feature Combination', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}ablation_heatmap_f1.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: ablation_heatmap_f1.png")

    #Plot 2: Average F1 per experiment
    #Average F1 across all 10 models for each experiment.
    #Shows which feature combination works best overall regardless of model choice.
    avg_f1 = ablation_df.groupby('experiment')['f1'].mean().sort_values(ascending=False)

    #Color bars by experiment type: blue=baseline, red=leave-one-out, green=single feature.
    colors = ['#2C7BB6' if 'Baseline' in e else
              '#D7191C' if 'Without'  in e else '#1A9641'
              for e in avg_f1.index]

    fig2, ax2 = plt.subplots(figsize=(13, 5))
    bars = ax2.bar(avg_f1.index, avg_f1.values, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_title('Average Brute Force Detection F1 per Feature Combination\n(Across All 10 Models)',
                  fontsize=13, fontweight='bold')
    ax2.set_ylabel('Mean F1-Score', fontweight='bold')
    ax2.set_ylim(0, 1.15)  #Extra headroom for value labels.
    ax2.tick_params(axis='x', rotation=35)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=8.5)  #Value label above each bar.

    patch1 = mpatches.Patch(color='#2C7BB6', label='All Features (Baseline)')
    patch2 = mpatches.Patch(color='#D7191C', label='Leave-One-Out')
    patch3 = mpatches.Patch(color='#1A9641', label='Single Feature Only')
    ax2.legend(handles=[patch1, patch2, patch3], fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}ablation_avg_f1_per_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: ablation_avg_f1_per_experiment.png")

    #Plot 3: F1 drop, feature importance
    #For each model, compute how much F1 drops when each feature is removed.
    #Drop = baseline F1 - F1 without that feature.
    #Higher drop = that feature is more critical for brute force detection.
    baseline_f1   = ablation_df[ablation_df['experiment'] == 'ALL (Baseline)'].groupby('model')['f1'].mean()
    leave_one_out = ablation_df[ablation_df['experiment'].str.startswith('Without')]  #Filter to leave-one-out rows only.

    drop_records = []
    for exp_name, group in leave_one_out.groupby('experiment'):
        removed_col = exp_name.replace('Without ', '')  #Extract the column name that was removed.
        model_f1    = group.groupby('model')['f1'].mean()
        for model, f1_val in model_f1.items():
            drop = baseline_f1.get(model, np.nan) - f1_val  #Positive drop = performance got worse without this feature.
            drop_records.append({
                'removed_column': removed_col,
                'model':          model,
                'f1_drop':        drop
            })

    drop_df    = pd.DataFrame(drop_records)
    pivot_drop = drop_df.pivot_table(
        index='model', columns='removed_column', values='f1_drop'
    )  #Rows = models, columns = removed features, values = F1 drop.

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot_drop, annot=True, fmt='.3f',
        cmap='Reds',  #Darker red = bigger F1 drop = more important feature.
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'F1 Drop when feature removed\n(Higher = more critical for brute force detection)'},
        ax=ax3
    )
    ax3.set_title('Brute Force Feature Importance — F1 Drop Analysis\n'
                  '(Higher = Removing this feature hurts attack detection more)',
                  fontsize=13, fontweight='bold', pad=15)
    ax3.set_xlabel('Removed Feature', fontweight='bold')
    ax3.set_ylabel('Model', fontweight='bold')
    ax3.tick_params(axis='x', rotation=20)
    ax3.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}ablation_f1_drop_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: ablation_f1_drop_heatmap.png")

    return drop_df  #Returned to main() for use in print_ablation_summary().


def plot_term_importance(importance_df, lr_importance_df, top_n=20):
    """
    Plot TF-IDF term importance — highlighting which specific
    terms are the strongest brute force attack indicators.
    """

    #RF Term Importance
    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importance_df)))[::-1]  #Gradient from red (low) to green (high importance).
    bars    = ax.barh(
        importance_df['term'][::-1],        #Reverse so highest importance appears at the top.
        importance_df['importance'][::-1],
        color=colors, edgecolor='white'
    )
    ax.set_title(f'Top {top_n} Brute Force Detection Terms\n(Random Forest Feature Importances)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature Importance', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.0002, bar.get_y() + bar.get_height()/2,
                f'{w:.4f}', va='center', fontsize=8)  #Value label to the right of each bar.
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}ablation_term_importance_rf.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: ablation_term_importance_rf.png")

    #LR Attack vs Normal indicators
    #Two side-by-side charts: left = attack indicators, right = normal indicators.
    #This makes it visually clear which terms push a record towards being flagged as an attack.
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Brute Force Attack vs Normal Behavior Indicators\n(Logistic Regression Coefficients)',
                  fontsize=13, fontweight='bold')

    #Loop through both directions (Anomaly and Normal) and their corresponding subplot.
    for ax_i, direction, color, label in zip(
        axes,
        ['Anomaly', 'Normal'],
        ['#D7191C', '#2C7BB6'],
        ['Attack Indicators →', '← Normal Behavior Indicators']
    ):
        subset = lr_importance_df[lr_importance_df['direction'] == direction].head(top_n)
        ax_i.barh(
            subset['term'][::-1].values,        #Reverse so the strongest indicator appears at top.
            subset['coefficient'][::-1].values,
            color=color, alpha=0.85, edgecolor='white'
        )
        ax_i.set_title(label, fontweight='bold', fontsize=11, color=color)
        ax_i.set_xlabel('LR Coefficient', fontweight='bold')
        ax_i.spines[['top', 'right']].set_visible(False)
        ax_i.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}ablation_term_importance_lr.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: ablation_term_importance_lr.png")


def plot_contamination_sensitivity(sens_df):
    """
    Line chart: F1 vs contamination rate.
    Highlights the actual brute force rate in the dataset (~20.6%).
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    #Consistent colors and markers per model — same palette used in other ablation plots.
    colors  = {
        'Isolation Forest':     '#2C7BB6',
        'Local Outlier Factor': '#D7191C',
        'One-Class SVM':        '#1A9641'
    }
    markers = {
        'Isolation Forest':     'o',
        'Local Outlier Factor': 's',
        'One-Class SVM':        '^'
    }

    for model_name, group in sens_df.groupby('model'):
        group = group.sort_values('contamination')  #Ensure points are connected left to right.
        ax.plot(
            group['contamination'], group['f1'],
            marker=markers.get(model_name, 'o'),
            color=colors.get(model_name, 'grey'),
            linewidth=2, markersize=7, label=model_name
        )

    #Vertical dashed line marking the actual attack rate, ideal contamination setting.
    #Models should perform best when contamination ≈ actual attack rate.
    ax.axvline(x=0.206, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.text(0.21, ax.get_ylim()[0] + 0.02,
            'Actual attack\nrate (~20.6%)',
            fontsize=8, color='black', alpha=0.7)

    ax.set_xlabel('Assumed Contamination Rate (% of traffic assumed to be attacks)',
                  fontweight='bold', fontsize=10)
    ax.set_ylabel('F1-Score (Brute Force Detection)', fontweight='bold', fontsize=10)
    ax.set_title('Contamination Rate Sensitivity for Brute Force Detection\n'
                 '(How does the assumed attack rate affect model performance?)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}ablation_contamination_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: ablation_contamination_sensitivity.png")


#############################################################################################
#Part 8: Summary
#############################################################################################

def print_ablation_summary(ablation_df, drop_df):
    """Print a brute force focused summary of all ablation findings."""
    print("\n")
    print("#" * 70)
    print("       ABLATION STUDY SUMMARY — BRUTE FORCE DETECTION")
    print("#" * 70)

    #Feature importance ranking
    #Average F1 drop across all 10 models for each removed feature.
    #Higher drop = that feature is more critical for brute force detection.
    avg_drop = drop_df.groupby('removed_column')['f1_drop'].mean().sort_values(ascending=False)
    print("\n  Feature Importance for Brute Force Detection")
    print("  (Ranked by average F1 drop when feature is removed):")
    print("  " + "─" * 60)
    for rank, (col, drop) in enumerate(avg_drop.items(), 1):
        bar     = '█' * int(abs(drop) * 100)  #Visual bar scaled by drop magnitude.
        finding = interpret_feature_importance(col, drop)  #Translate drop into a severity label.
        print(f"\n  {rank}. {col:<15} Avg F1 Drop: {drop:+.4f}  {bar}")
        print(f"     {finding}")

    #Best and worst feature combinations
    #Average F1 across all 10 models per experiment to find the strongest and weakest combinations.
    avg_by_exp = ablation_df.groupby('experiment')['f1'].mean().sort_values(ascending=False)
    print(f"\n\n  Best feature combination  : {avg_by_exp.index[0]:<30} F1={avg_by_exp.iloc[0]:.4f}")
    print(f"  Worst feature combination : {avg_by_exp.index[-1]:<30} F1={avg_by_exp.iloc[-1]:.4f}")

    #Key cybersecurity takeaway
    most_important  = avg_drop.index[0]   #Feature whose removal hurts the most.
    least_important = avg_drop.index[-1]  #Feature whose removal hurts the least.
    print(f"\n  Key Finding:")
    print(f"  '((({most_important})))' is the most critical feature for brute force detection through ML.")
    print(f"  '((({least_important})))' has the smallest impact on detection performance when all other features are already present in the model.")
    print("-" * 70)

    #Export feature importance to IEEE-style LaTeX table
    #Produces a .tex file that can be pasted directly into the ablation section of the IEEE paper to be submitted later.
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\caption{Feature Importance for Brute Force Detection (Ablation Study)}')
    lines.append(r'\label{tab:ablation}')
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{lccl}')
    lines.append(r'\hline')  #Top rule.
    lines.append(r'\textbf{Rank} & \textbf{Feature} & \textbf{Avg F1 Drop} & \textbf{Severity} \\')
    lines.append(r'\hline')  #Header rule.

    for rank, (col, drop) in enumerate(avg_drop.items(), 1):
        #Get just the severity label (CRITICAL, HIGH, etc.) from interpret_feature_importance.
        severity = interpret_feature_importance(col, drop).split(']')[0].replace('[', '')
        lines.append(f"{rank} & {col} & {drop:+.4f} & {severity} \\\\")

    lines.append(r'\hline')  #Bottom rule.
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex_path = f'{OUTPUTS_DIR}table_ablation_summary.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n[SAVED] Saved: table_ablation_summary.tex")

#############################################################################################
#Part 9: Data Loading
#############################################################################################

def load_datasets():
    """Load and return both the main and labeled datasets."""
    df_main = load_data(MAIN_DATASET_PATH)        #Uses load_data() from config.py.
    if df_main is None:
        return None, None

    df_labeled = load_labeled_data(LABELED_DATASET_PATH)  #Uses load_labeled_data() from config.py.
    if df_labeled is None:
        return None, None

    #Print the cybersecurity role of each feature column for context.
    print(f"\n  Brute Force Context:")
    print(f"  {'─'*55}")
    for col, context in FEATURE_CONTEXT.items():
        if col in df_labeled.columns:
            print(f"  {col:<15} → {context}")
    print()

    return df_main, df_labeled

#############################################################################################
#Part 10: Main
#############################################################################################

def main():
    print("#" * 60)
    print("   ABLATION STUDY — BRUTE FORCE ATTACK DETECTION")
    print("   Azure Authentication Records")
    print("#" * 60)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)  #Create outputs folder if it doesn't exist.

    #Load data
    df_main, df_labeled = load_datasets()  #Load both main and labeled datasets from config paths.
    if df_main is None or df_labeled is None:
        print("[FAILURE] Could not load data. Exiting.")
        return  #Stop execution if either file failed to load.

    #Experiment 1: Column Ablation
    #Tests all 9 feature combinations (baseline, leave-one-out, single-feature) across all 10 models to measure which features matter most.
    print("\n" + "#" * 60)
    print("#   EXPERIMENT 1: Feature Column Ablation              #")
    print("#   Which features matter most for brute force detection? #")
    print("#" * 60)
    ablation_df = run_column_ablation(df_labeled, ALL_FEATURE_COLUMNS)
    ablation_df.to_csv(f'{OUTPUTS_DIR}ablation_column_results.csv', index=False)  #Save full results for reference.
    print(f"\n[SAVED] Saved: ablation_column_results.csv ({len(ablation_df)} rows)")

    #Experiment 2: Term Importance
    #Uses Random Forest and Logistic Regression to identify which specific words/terms are the strongest brute force attack indicators.
    print("\n" + "#" * 60)
    print("#   EXPERIMENT 2: TF-IDF Term Importance               #")
    print("#   Which specific terms signal a brute force attack?  #")
    print("#" * 60)
    importance_df, lr_importance_df = run_term_importance(
        df_labeled, ALL_FEATURE_COLUMNS, top_n=20
    )
    importance_df.to_csv(f'{OUTPUTS_DIR}ablation_term_importance_rf.csv', index=False)    #RF importance scores.
    lr_importance_df.to_csv(f'{OUTPUTS_DIR}ablation_term_importance_lr.csv', index=False) #LR coefficients.

    #Experiment 3: Contamination Sensitivity
    #Tests how much the assumed attack rate (contamination) affects detection performance for Isolation Forest, LOF and One-Class SVM.
    print("\n" + "#" * 60)
    print("#   EXPERIMENT 3: Contamination Rate Sensitivity       #")
    print("#   How does the assumed attack rate affect detection?  #")
    print("#" * 60)
    sens_df = run_contamination_sensitivity(df_labeled)
    sens_df.to_csv(f'{OUTPUTS_DIR}ablation_contamination_sensitivity.csv', index=False)  #Save sensitivity results.

    #Plotting
    #All charts saved to OUTPUTS_DIR, must run after all 3 experiments complete.
    print("\n" + "#" * 60)
    print("#              GENERATING CHARTS                       #")
    print("#" * 60)
    drop_df = plot_column_ablation(ablation_df)          #F1 heatmap, avg F1 bar chart, F1 drop heatmap.
    plot_term_importance(importance_df, lr_importance_df, top_n=20)  #RF importance + LR coefficients charts.
    plot_contamination_sensitivity(sens_df)              #F1 vs contamination rate line chart.

    #Summary
    #Prints ranked feature importance + key cybersecurity finding.
    #Also exports table_ablation_summary.tex for the IEEE paper.
    print_ablation_summary(ablation_df, drop_df)

    print("\n[COMPLETE] Ablation study complete! Check '../outputs/' for all results.")



#Only runs main() when this file is executed directly.
#Prevents main() from running if this file is imported by another script.
if __name__ == "__main__":
    main()