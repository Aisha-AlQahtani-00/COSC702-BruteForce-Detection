# ============================================================
# Scenario_Test.py
# Scenario-Based Testing — Brute Force Attack Detection
# Dataset: Microsoft Azure Authentication Records
#############################################################################################
#Spring 2026
#COSC 702: Advanced AI-Driven Software Engineering
#Project: Benchmarking the Detection of Cloud-based User Authentication Log Anomalies Using Machine Learning Techniques
#Code Component: Scenarios Test v2 (Updated 29 Apr)
#Datasets:
#   1. Small dataset        (1k records)
#   2. Medium dataset       (15k records)
#   3. Large dataset        (20k records)
#   4. Very large dataset   (100k records)
#   5. Healthy logs only    (No attacks, false positive test)
#   6. Empty dataset        (Edge case robustness test)
#Submitted to: Dr. Jamal Bentahar
#Done by: Aisha AlQahtani, & Salwa Mohammed Razaulla
#############################################################################################

#############################################################################################
#Part 0: Configuration
#############################################################################################

#Importing the configuration file containing library imports, file paths, feature columns, model lists and colors, & shared helper functions.

from config import *

#############################################################################################
#Part 1: Scenario-Test-Specific Set-up
#############################################################################################


#Scenario-specific paths and settings, these are kept here (not in config.py) because they are only relevant to the scenario testing file.

#Paths to the pre-trained models saved by Main_AD_Code.py, these .pkl files must exist before this script can run.

MODELS = {
    #Unsupervised
    'Isolation Forest':     '../models/isolation_forest.pkl',
    'K-Means Clustering':   '../models/kmeans.pkl',
    'Elliptic Envelope':    '../models/elliptic_envelope.pkl',
    'Local Outlier Factor': '../models/lof.pkl',
    'One-Class SVM':        '../models/one_class_svm.pkl',
    # Supervised
    'Random Forest':        '../models/random_forest.pkl',
    'SVM':                  '../models/svm.pkl',
    'Logistic Regression':  '../models/logistic_regression.pkl',
    'Gradient Boosting':    '../models/gradient_boosting.pkl',
    'KNN':                  '../models/knn.pkl',
}

#Path to the TF-IDF vectorizer saved by Main_AD_Code.py; critical: must use the same vectorizer to maintain the same feature space.
VECTORIZER_PATH = '../models/tfidf_vectorizer.pkl'

#The 6 test scenarios, each represents a different real-world data condition.
#This also includes some new labeled datasets after the demonstration on Tuesday 28th of April.

SCENARIOS = {
    'Small (1k)':        "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Dataset1k.csv",
    'Medium (15k)':      "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Dataset15k.csv",
    'Large (20k)':       "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Dataset20k.csv",
    'Very Large (100k)': "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Dataset100k.csv",
    'Healthy Logs':      "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Healthy_logs_sample.csv",
    'Empty Dataset':     "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/empty_dataset.csv",
}

LABELED_SCENARIOS = {
    'Small (1k)':        "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Labeled/Labeled1k.csv",
    'Medium (15k)':      "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Labeled/Labeled15k.csv",
    'Large (20k)':       "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Labeled/Labeled20k.csv",
    'Very Large (100k)': "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Labeled/Labeled100k.csv",
    'Healthy Logs':      "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Labeled/Healthy_logs_sample_labeled.csv",
    'Empty Dataset':     None,
}

#############################################################################################
#Part 2: Loading the Pre-trained Models & Vectorizer
#############################################################################################

def load_models_and_vectorizer():
    """
    Load all 10 pre-trained models and the TF-IDF vectorizer
    saved by Main_AD_Code.py. These were trained on brute force
    attack data and will now be tested on new scenarios.
    """
    print("[1] Loading pre-trained brute force detection models...")

    loaded_models = {}
    for model_name, path in MODELS.items():
        try:
            loaded_models[model_name] = joblib.load(path)  #Load each model from its .pkl file.
            model_type = 'Unsupervised' if model_name in UNSUPERVISED_MODELS else 'Supervised'
            print(f"  [SUCCESS] {model_name:<25} ({model_type})")
        except Exception as e:
            print(f"  [FAILURE] {model_name} failed to load: {e}")  #Log failure but continue loading remaining models.

    try:
        vectorizer = joblib.load(VECTORIZER_PATH)  #Load the same vectorizer used during training.
        print(f"  [SUCCESS] TF-IDF vectorizer loaded\n")
    except Exception as e:
        print(f"  [FAILURE] Vectorizer failed to load: {e}")
        vectorizer = None

    return loaded_models, vectorizer  #Both returned for use in run_scenario().


#############################################################################################
#Part 3: Data Loading & Pre-Processing
#############################################################################################

def load_scenario_data(scenario_name, file_path):
    """
    Load a scenario dataset with full edge case handling.
    Returns (df, status) where status is one of:
    'ok', 'empty', 'missing_column', 'load_error'
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('ï»¿', '', regex=True)  #Clean up column name encoding artifacts.
    except Exception as e:
        print(f"  [FAILURE] Could not load file: {e}")
        return None, 'load_error'

    # Edge case 1 — empty file
    if df.empty:
        print(f"  [WARNING] Empty dataset — edge case scenario")
        return df, 'empty'

    # Edge case 2 — none of the required feature columns exist
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if len(missing_cols) == len(FEATURE_COLUMNS):
        print(f"  [WARNING] None of the feature columns found")
        return df, 'missing_column'

    print(f"  [SUCCESS] Loaded {len(df):,} records")

    #Print status distribution for cybersecurity context.
    #It shows how many Failure/Success/Interrupted records are in this scenario.
    if 'Status' in df.columns:
        status_counts = df['Status'].value_counts()
        print(f"  [INFO] Status distribution: {status_counts.to_dict()}")

    return df, 'ok'


def vectorize_with_pretrained(df, vectorizer):
    """
    Transform text using the PRE-TRAINED vectorizer from Main_AD_Code.py.
    Uses transform() NOT fit_transform() — critical to maintain
    the same feature space the models were trained on.
    If fit_transform() were used here, the vocabulary would change
    and the models would receive features they were never trained on.
    """
    combined = combine_columns(df, FEATURE_COLUMNS)  #combine_columns() imported from config.py.
    texts    = combined.apply(preprocess_text)        #preprocess_text() imported from config.py.
    matrix   = vectorizer.transform(texts)            #transform() only; vocabulary already fixed from training.
    return matrix

#############################################################################################
#Part 4: Predictions
#############################################################################################

def predict_with_model(model_name, model, matrix):
    """
    Run predictions using a pre-trained model.
    All predictions normalized to -1 (anomaly/attack) and 1 (normal)
    regardless of whether the model is supervised or unsupervised.
    """
    try:
        #Fit SVD reduction to match training dimensions
        #If fewer features than expected, pad with zeros to match training size
        reduced = get_reduced(matrix)
        if reduced.shape[1] < 50:
            padding = np.zeros((reduced.shape[0], 50 - reduced.shape[1]))
            reduced = np.hstack([reduced, padding])

        if model_name in ['Isolation Forest', 'Local Outlier Factor']:
            preds = model.predict(matrix)   #Sparse matrix input: returns -1 or 1.

        elif model_name in ['Elliptic Envelope', 'One-Class SVM']:
            preds = model.predict(reduced)  #Dense input required: returns -1 or 1.

        elif model_name == 'K-Means Clustering':
            #K-Means doesn't have a predict() for anomaly detection so we recompute distances and apply the same 70th percentile threshold.
            _, distances = pairwise_distances_argmin_min(matrix, model.cluster_centers_)
            threshold    = np.percentile(distances, 70)
            preds        = np.where(distances >= threshold, -1, 1)

        elif model_name in SUPERVISED_MODELS:
            preds_raw = model.predict(matrix)           #Supervised models return 0 (normal) or 1 (attack).
            preds     = np.where(preds_raw == 1, -1, 1) #Convert to -1/1 convention for consistency.

        else:
            return None, None

        anomaly_rate = round(np.sum(preds == -1) / len(preds) * 100, 2)  #% of records flagged as attack.
        return preds, anomaly_rate

    except Exception as e:
        print(f"      [FAILURE] Prediction failed: {e}")
        return None, None

#############################################################################################
#Part 5: Brute-Force Finding Interpreter
#############################################################################################

def interpret_brute_force_finding(scenario_name, anomaly_rate, model_name, df=None, f1=None):
    """
    Translate anomaly detection rate into a plain-English
    cybersecurity finding with actionable context.
    """

    #Healthy logs.
    if scenario_name == 'Healthy Logs':
        if anomaly_rate == 0:
            return 'PASS — Zero false positives on clean authentication logs'
        elif anomaly_rate < 2:
            return f'ACCEPTABLE — {anomaly_rate}% false positive rate on healthy logs'
        elif anomaly_rate < 10:
            return f'CAUTION — {anomaly_rate}% false positives, may alert on legitimate logins'
        else:
            return f'FAIL — {anomaly_rate}% false positives, unreliable for brute force detection'

    #Empty dataset.
    elif scenario_name == 'Empty Dataset':
        return 'EDGE CASE HANDLED — No data to process'

    #Regular datasets.
    else:
        failure_hint = ''
        if df is not None and 'Status' in df.columns:
            failure_rate = (df['Status'] == 'Failure').sum() / len(df) * 100
            if failure_rate > 0:
                failure_hint = f' (dataset contains {failure_rate:.1f}% login failures)'

        #If F1 is available and high, override the detection rate finding.
        if f1 is not None and f1 >= 0.90:
            return f'ACCURATE DETECTION — F1={f1:.4f}, well-calibrated to actual attack rate{failure_hint}'

        if anomaly_rate == 0:
            return f'NO DETECTION — Model failed to flag any brute force attempts{failure_hint}'
        elif anomaly_rate < 5:
            return f'LOW DETECTION — Only {anomaly_rate}% flagged, may miss attacks{failure_hint}'
        elif anomaly_rate <= 35:
            return f'NORMAL DETECTION — {anomaly_rate}% flagged, consistent with expected attack rate{failure_hint}'
        elif anomaly_rate <= 60:
            return f'HIGH DETECTION — {anomaly_rate}% flagged, possible over-detection{failure_hint}'
        else:
            return f'OVER-DETECTION — {anomaly_rate}% flagged, model may be too aggressive{failure_hint}'

#############################################################################################
#Part 6: Run Scenarios
#############################################################################################

def run_scenario(scenario_name, file_path, loaded_models, vectorizer):
    """
    Run all 10 pre-trained models against a single scenario dataset.
    Computes full metrics if labeled data is available for this scenario.
    Returns a list of result dictionaries — one per model.
    """
    print(f"\n  {'═'*58}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"  {'═'*58}")

    results = []

    #Load scenario data
    df, status = load_scenario_data(scenario_name, file_path)

    #Handle edge case: empty dataset
    if status == 'empty':
        for model_name in loaded_models:
            results.append({
                'scenario':     scenario_name,
                'model':        model_name,
                'type':         'Unsupervised' if model_name in UNSUPERVISED_MODELS else 'Supervised',
                'status':       'EDGE CASE',
                'n_records':    0,
                'n_anomalies':  None,
                'anomaly_rate': None,
                'finding':      'EDGE CASE HANDLED — Empty dataset processed gracefully',
                'f1':           None,
                'precision':    None,
                'recall':       None,
                'auc_roc':      None,
            })
        return results

    #Handle load errors or missing columns
    if status in ('load_error', 'missing_column'):
        for model_name in loaded_models:
            results.append({
                'scenario':     scenario_name,
                'model':        model_name,
                'type':         'Unsupervised' if model_name in UNSUPERVISED_MODELS else 'Supervised',
                'status':       'ERROR',
                'n_records':    0,
                'n_anomalies':  None,
                'anomaly_rate': None,
                'finding':      f'Could not process: {status}',
                'f1':           None,
                'precision':    None,
                'recall':       None,
                'auc_roc':      None,
            })
        return results

    #Load labeled data if available (AFTER status checks)
    #df is guaranteed to exist here so len(df) is safe
    labeled_df   = None
    labeled_path = LABELED_SCENARIOS.get(scenario_name)
    if labeled_path:
        try:
            labeled_df = load_labeled_data(labeled_path)  #load_labeled_data() from config.py
            if labeled_df is not None:
                print(f"  [SUCCESS] Labeled data loaded — {len(labeled_df):,} records")
                if len(labeled_df) != len(df):
                    print(f"  [WARNING] Length mismatch — scenario: {len(df)}, "
                          f"labeled: {len(labeled_df)} — trimming to match")
                    labeled_df = labeled_df.iloc[:len(df)].reset_index(drop=True)
        except Exception as e:
            print(f"  [WARNING] Could not load labeled data: {e}")

    #Vectorize using the pre-trained vectorizer
    try:
        matrix = vectorize_with_pretrained(df, vectorizer)
        print(f"  [SUCCESS] Vectorized: {matrix.shape}")
    except Exception as e:
        print(f"  [FAILURE] Vectorization failed: {e}")
        return results

    #Run all 10 models
    print(f"\n  {'Model':<25} {'Type':<14} {'Flagged':>8} {'Rate':>7}   {'F1':>6}   Finding")
    print(f"  {'─'*100}")

    for model_name, model in loaded_models.items():
        model_type          = 'Unsupervised' if model_name in UNSUPERVISED_MODELS else 'Supervised'
        preds, anomaly_rate = predict_with_model(model_name, model, matrix)

        if preds is None:
            results.append({
                'scenario':     scenario_name,
                'model':        model_name,
                'type':         model_type,
                'status':       'PREDICTION ERROR',
                'n_records':    len(df),
                'n_anomalies':  None,
                'anomaly_rate': None,
                'finding':      'Model prediction failed',
                'f1':           None,
                'precision':    None,
                'recall':       None,
                'auc_roc':      None,
            })
            continue

        n_anomalies = int(np.sum(preds == -1))

        #Compute full metrics if labeled data available
        # kip F1 for Healthy Logs (All records are Normal so F1 is meaningless here).
        metrics = None
        if labeled_df is not None and scenario_name != 'Healthy Logs':
            try:
                if model_name in UNSUPERVISED_MODELS:
                    true_unsup = labeled_df['label_unsup'].values
                    reduced    = get_reduced(matrix)
                    if model_name == 'Isolation Forest':
                        scores = model.decision_function(matrix)
                    elif model_name == 'K-Means Clustering':
                        _, distances = pairwise_distances_argmin_min(matrix, model.cluster_centers_)
                        scores = -distances
                    elif model_name in ['Elliptic Envelope', 'One-Class SVM']:
                        scores = model.decision_function(reduced)
                    elif model_name == 'Local Outlier Factor':
                        scores = model.decision_function(matrix)
                    else:
                        scores = preds
                    metrics = compute_metrics_unsup(true_unsup, preds, scores)  # from config.py

                else:
                    true_sup  = labeled_df['label_sup'].values
                    preds_sup = np.where(preds == -1, 1, 0)  # Convert back to 0/1
                    scores    = model.predict_proba(matrix)[:, 1] if hasattr(model, 'predict_proba') else preds_sup
                    metrics   = compute_metrics_sup(true_sup, preds_sup, scores)  # from config.py

            except Exception as e:
                print(f"  [WARNING] Metrics failed for {model_name}: {e}")

        #Pass F1 to finding interpreter if metrics available
        #This allows the interpreter to give accurate findings when detection rate appears low but F1 is high (e.g. 100k dataset).
        f1_val  = metrics['f1'] if metrics else None
        finding = interpret_brute_force_finding(
            scenario_name, anomaly_rate, model_name, df, f1=f1_val
        )

        #Format F1 for display.
        f1_display = f"{metrics['f1']:.4f}" if metrics else "N/A"

        print(
            f"  {model_name:<25} {model_type:<14} "
            f"{n_anomalies:>8,} {anomaly_rate:>6.1f}%  {f1_display:>6}   {finding}"
        )

        results.append({
            'scenario':     scenario_name,
            'model':        model_name,
            'type':         model_type,
            'status':       'OK',
            'n_records':    len(df),
            'n_anomalies':  n_anomalies,
            'anomaly_rate': anomaly_rate,
            'finding':      finding,
            'f1':           metrics['f1']        if metrics else None,
            'precision':    metrics['precision'] if metrics else None,
            'recall':       metrics['recall']    if metrics else None,
            'auc_roc':      metrics['auc_roc']   if metrics else None,
        })

    return results

#############################################################################################
#Part 7: Plotting
#############################################################################################

def plot_anomaly_rates(results_df):
    """
    Grouped bar chart: Brute force detection rate per model per scenario.
    Each group of bars = one model, each bar colour = one scenario.
    """
    plot_df = results_df[
        (results_df['status'] == 'OK') &
        (results_df['anomaly_rate'].notna())
    ].copy()

    if plot_df.empty:
        print("[WARNING] No data to plot.")
        return

    models    = UNSUPERVISED_MODELS + SUPERVISED_MODELS  #Full model order from config.py.
    scenarios = [s for s in SCENARIOS.keys() if s in plot_df['scenario'].unique()]
    x         = np.arange(len(models))
    width     = 0.8 / max(len(scenarios), 1)  #Bar width scales with number of scenarios.

    fig, ax = plt.subplots(figsize=(16, 7))

    for i, scenario in enumerate(scenarios):
        s_df = plot_df[plot_df['scenario'] == scenario]
        vals = [
            s_df[s_df['model'] == m]['anomaly_rate'].values[0]
            if not s_df[s_df['model'] == m].empty else 0
            for m in models
        ]
        offset = (i - len(scenarios) / 2) * width + width / 2  #Centre bars around each model position.
        bars   = ax.bar(
            x + offset, vals, width,
            label=scenario,
            color=SCENARIO_COLORS.get(scenario, 'grey'),  #SCENARIO_COLORS imported from config.py.
            alpha=0.85, edgecolor='white', linewidth=0.5
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=5.5
                )

    #Reference line: expected attack rate based on labeled dataset.
    ax.axhline(y=20.6, color='red', linestyle='--', alpha=0.4,
               linewidth=1.2, label='Expected brute force rate (~20.6%)')

    #Vertical divider between unsupervised (left) and supervised (right).
    ax.axvline(x=4.5, color='grey', linestyle='-', alpha=0.3, linewidth=1)
    ax.text(1.8, ax.get_ylim()[1] * 0.95, 'Unsupervised', ha='center',
            fontsize=9, color='#1f77b4', fontweight='bold')
    ax.text(7.0, ax.get_ylim()[1] * 0.95, 'Supervised', ha='center',
            fontsize=9, color='#d62728', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=8)
    ax.set_ylabel('Brute Force Detection Rate (%)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Model', fontweight='bold', fontsize=11)
    ax.set_title('Brute Force Attack Detection Rate per Model across All Scenarios',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}scenarios_anomaly_rates.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: scenarios_anomaly_rates.png")


def plot_false_positive_analysis(results_df):
    """
    Bar chart focusing on the Healthy Logs scenario only.
    The most critical cybersecurity test, a good brute force
    detector should NOT flag legitimate authentication activity.
    Green = no false positives, orange = moderate, red = high.
    """
    healthy_df = results_df[
        (results_df['scenario'] == 'Healthy Logs') &
        (results_df['anomaly_rate'].notna())
    ].copy()

    if healthy_df.empty:
        print("[!] No healthy logs results to plot.")
        return

    #Reorder models to match consistent unsupervised then supervised order.
    model_order = [m for m in UNSUPERVISED_MODELS + SUPERVISED_MODELS if m in healthy_df['model'].values]
    healthy_df  = healthy_df.set_index('model').reindex(model_order).reset_index()

    #Color each bar by false positive severity.
    colors = [
        '#1a9641' if r == 0 else       #Green, no false positives.
        '#fdae61' if r < 10 else       #Orange, moderate.
        '#d7191c'                       #Red, high false positive rate.
        for r in healthy_df['anomaly_rate']
    ]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(
        healthy_df['model'], healthy_df['anomaly_rate'],
        color=colors, edgecolor='white', linewidth=0.5, alpha=0.9
    )

    #5% threshold, industry standard acceptable false positive rate.
    ax.axhline(y=5, color='grey', linestyle='--', alpha=0.6,
               label='5% acceptable false positive threshold')

    #Divider between unsupervised and supervised.
    ax.axvline(x=4.5, color='grey', linestyle='-', alpha=0.3, linewidth=1)
    ax.text(2.0, ax.get_ylim()[1] * 0.9, 'Unsupervised',
            ha='center', fontsize=9, color='#1f77b4', fontweight='bold')
    ax.text(7.0, ax.get_ylim()[1] * 0.9, 'Supervised',
            ha='center', fontsize=9, color='#d62728', fontweight='bold')

    for bar, val in zip(bars, healthy_df['anomaly_rate']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold'
        )

    ax.set_ylabel('False Positive Rate (%)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Model', fontweight='bold', fontsize=11)
    ax.set_title('False Positive Rate on Healthy Authentication Logs\n'
                 '(Lower is better — a good brute force detector should NOT flag legitimate logins)',
                 fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    patch1 = mpatches.Patch(color='#1a9641', label='✓ No false positives')
    patch2 = mpatches.Patch(color='#fdae61', label='⚠ Moderate false positives')
    patch3 = mpatches.Patch(color='#d7191c', label='✗ High false positives')
    ax.legend(handles=[patch1, patch2, patch3], fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}scenarios_false_positive_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: scenarios_false_positive_analysis.png")


def plot_heatmap(results_df):
    """
    Heatmap: Models × Scenarios showing brute force detection rate.
    Gives an at-a-glance view of how each model behaves across all
    scenarios; dark red = high detection, light green = low detection.
    """
    plot_df = results_df[results_df['anomaly_rate'].notna()].copy()
    if plot_df.empty:
        return

    #Pivot so rows = models, columns = scenarios, values = detection rate.
    pivot = plot_df.pivot_table(
        index='model', columns='scenario',
        values='anomaly_rate', aggfunc='mean'
    )
    #nforce consistent row and column ordering.
    ordered_cols = [s for s in SCENARIOS.keys() if s in pivot.columns]
    ordered_rows = [m for m in UNSUPERVISED_MODELS + SUPERVISED_MODELS if m in pivot.index]
    pivot        = pivot.loc[ordered_rows, ordered_cols]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',  #Red = high detection, green = low.
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Brute Force Detection Rate (%)'},
        ax=ax, vmin=0, vmax=100  #Fix scale to 0-100% for consistency.
    )
    ax.set_title('Brute Force Detection Rate Heatmap\n(Models × Scenarios | % of records flagged as attack)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.tick_params(axis='y', rotation=0)

    #White line dividing unsupervised (top) from supervised (bottom) rows.
    ax.axhline(y=5, color='white', linewidth=3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}scenarios_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: scenarios_heatmap.png")


def plot_detection_consistency(results_df):
    """
    Line chart: Detection rate vs dataset size (1k → 100k).
    Shows whether models stay consistent as data volume grows —
    flat lines = reliable model, erratic lines = unstable model.
    Critical for assessing real-world deployment confidence.
    """
    size_scenarios = ['Small (1k)', 'Medium (15k)', 'Large (20k)', 'Very Large (100k)']
    plot_df = results_df[
        (results_df['scenario'].isin(size_scenarios)) &
        (results_df['status'] == 'OK') &
        (results_df['anomaly_rate'].notna())
    ].copy()

    if plot_df.empty:
        print("[WARNING] No size scenario data to plot.")
        return

    #Map scenario name to actual record count for the x-axis.
    size_map = {
        'Small (1k)': 1000, 'Medium (15k)': 15000,
        'Large (20k)': 20000, 'Very Large (100k)': 100000
    }
    plot_df['n_records'] = plot_df['scenario'].map(size_map)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Brute Force Detection Consistency Across Dataset Sizes\n'
                 '(Consistent lines = reliable model | Erratic lines = unstable)',
                 fontsize=13, fontweight='bold')

    #Plot unsupervised and supervised separately for clarity.
    for ax, model_group, title in zip(
        axes,
        [UNSUPERVISED_MODELS, SUPERVISED_MODELS],
        ['Unsupervised Models', 'Supervised Models']
    ):
        group_df = plot_df[plot_df['model'].isin(model_group)]

        for model_name in model_group:
            m_df = group_df[group_df['model'] == model_name].sort_values('n_records')
            if m_df.empty:
                continue
            ax.plot(
                m_df['n_records'], m_df['anomaly_rate'],
                marker='o',
                color=MODEL_COLORS.get(model_name, 'grey'),  #MODEL_COLORS imported from config.py.
                linewidth=2, markersize=7, label=model_name
            )

        #Reference line, expected attack rate from labeled dataset.
        ax.axhline(y=20.6, color='red', linestyle='--', alpha=0.4,
                   linewidth=1.2, label='Expected attack rate (~20.6%)')

        ax.set_xlabel('Number of Records', fontweight='bold')
        ax.set_ylabel('Detection Rate (%)', fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))  #Format x-axis as 1,000 not 1000.

    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}scenarios_detection_consistency.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: scenarios_detection_consistency.png")

#Newly added plot after the Apr 28 discussion and feedback.
def plot_flagged_vs_ground_truth(results_df):
    """
    Grouped bar chart comparing flagged rate (what the model detected)
    vs actual attack rate (ground truth labels) per model per scenario.
    Shows how close each model's detection is to reality.
    """
    #Only plot scenarios that have both anomaly_rate (Just % of flags) and ground truth f1.
    plot_df = results_df[
        (results_df['status'] == 'OK') &
        (results_df['anomaly_rate'].notna()) &
        (results_df['f1'].notna())
    ].copy()

    if plot_df.empty:
        print("[WARNING] No labeled scenario data available for comparison plot.")
        return

    scenarios    = plot_df['scenario'].unique()
    model_order  = UNSUPERVISED_MODELS + SUPERVISED_MODELS
    models_avail = [m for m in model_order if m in plot_df['model'].unique()]

    fig, axes = plt.subplots(len(scenarios), 1,
                             figsize=(14, 5 * len(scenarios)))
    if len(scenarios) == 1:
        axes = [axes]

    fig.suptitle('Flagged Rate vs Ground Truth F1 — All Models per Scenario\n'
                 '(Blue = detection rate | Red = F1 against ground truth labels)',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, scenario in zip(axes, scenarios):
        s_df = plot_df[plot_df['scenario'] == scenario]
        x    = np.arange(len(models_avail))
        width = 0.35

        flagged_rates = [
            s_df[s_df['model'] == m]['anomaly_rate'].values[0]
            if not s_df[s_df['model'] == m].empty else 0
            for m in models_avail
        ]
        f1_scores = [
            s_df[s_df['model'] == m]['f1'].values[0] * 100  #Scaling.
            if not s_df[s_df['model'] == m].empty else 0
            for m in models_avail
        ]

        bars1 = ax.bar(x - width/2, flagged_rates, width,
                       label='Detection rate (%)',
                       color='#2C7BB6', alpha=0.85, edgecolor='white')
        bars2 = ax.bar(x + width/2, f1_scores, width,
                       label='F1-Score (scaled to %)',
                       color='#D7191C', alpha=0.85, edgecolor='white')

        #Value labels here
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=7,
                        color='#2C7BB6')
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=7,
                        color='#D7191C')

        #Divider between unsupervised and supervised (Aesthetics)
        n_unsup = len([m for m in models_avail if m in UNSUPERVISED_MODELS])
        ax.axvline(x=n_unsup - 0.5, color='grey', linestyle='--', alpha=0.4, linewidth=1)

        ax.set_title(f'Scenario: {scenario}', fontweight='bold', fontsize=11, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\n') for m in models_avail], fontsize=8)
        ax.set_ylabel('Rate (%)', fontweight='bold')
        ax.set_ylim(0, 115)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f'{OUTPUTS_DIR}scenarios_flagged_vs_groundtruth.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: scenarios_flagged_vs_groundtruth.png")

#############################################################################################
#Part 7: Summary
#############################################################################################

def print_scenario_summary(results_df):
    """
    Print a brute force focused cybersecurity summary of all
    scenario results, grouped by scenario then by model type.
    """
    print("\n")
    print("#" * 75)
    print("    SCENARIO TESTING SUMMARY — BRUTE FORCE ATTACK DETECTION")
    print("#" * 75)

    #Loop through each scenario and print all model results.
    for scenario in SCENARIOS.keys():
        s_df = results_df[results_df['scenario'] == scenario]
        if s_df.empty:
            continue

        print(f"\n  📋 {scenario}")
        print("  " + "─" * 70)

        #Print unsupervised models first, then supervised for consistency.
        for model_type, group in [('Unsupervised', UNSUPERVISED_MODELS), ('Supervised', SUPERVISED_MODELS)]:
            type_df = s_df[s_df['model'].isin(group)]
            if type_df.empty:
                continue
            print(f"  [{model_type}]")
            for _, row in type_df.iterrows():
                if row['anomaly_rate'] is not None:
                    print(
                        f"    {row['model']:<25} "
                        f"{row['n_anomalies']:>6,} flagged "
                        f"({row['anomaly_rate']:.1f}%)  →  {row['finding']}"
                    )
                else:
                    print(f"    {row['model']:<25} {row['finding']}")

    #Key cybersecurity insights.
    print("\n")
    print("=" * 75)
    print("  KEY CYBERSECURITY INSIGHTS")
    print("=" * 75)

    #False positive analysis: which models are safest on clean data.
    healthy = results_df[
        (results_df['scenario'] == 'Healthy Logs') &
        (results_df['anomaly_rate'].notna())
    ]
    if not healthy.empty:
        best_fp  = healthy.loc[healthy['anomaly_rate'].idxmin()]   #Model with lowest false positive rate.
        worst_fp = healthy.loc[healthy['anomaly_rate'].idxmax()]   #Model with highest false positive rate.
        zero_fp  = healthy[healthy['anomaly_rate'] == 0]           #Models with perfect false positive rate.
        print(f"\n  False Positive Performance on Healthy Logs:")
        print(f"    Best model  : {best_fp['model']} — {best_fp['anomaly_rate']:.1f}% false positive rate")
        print(f"    Worst model : {worst_fp['model']} — {worst_fp['anomaly_rate']:.1f}% false positive rate")
        if not zero_fp.empty:
            print(f"    Zero FP     : {', '.join(zero_fp['model'].tolist())} — perfect on clean logs ✓")

    #Detection consistency, which models are most stable across dataset sizes.
    size_scenarios = ['Small (1k)', 'Medium (15k)', 'Large (20k)', 'Very Large (100k)']
    size_df = results_df[
        (results_df['scenario'].isin(size_scenarios)) &
        (results_df['anomaly_rate'].notna())
    ]
    if not size_df.empty:
        #Standard deviation of detection rate across sizes: lower = more consistent.
        consistency = size_df.groupby('model')['anomaly_rate'].std().sort_values()
        print(f"\n  Detection Consistency (lower std = more consistent across sizes):")
        for model, std in consistency.items():
            bar = '█' * int(std / 2)  #Visual bar scaled by std.
            print(f"    {model:<25} std={std:.2f}%  {bar}")

    #Edge case handling check.
    empty = results_df[results_df['scenario'] == 'Empty Dataset']
    if not empty.empty:
        print(f"\n  Edge Case (Empty Dataset): All models handled gracefully ✓")

    print("=" * 75)

#Export to IEEE-style LaTeX tables (For the assignment submission).

    #Table 1: Full scenario results (detection rate per model per scenario).
    lines = []
    lines.append(r'\begin{table*}[htbp]')  #table* spans both columns in IEEE two-column layout.
    lines.append(r'\caption{Brute Force Detection Rate (\%) Across All Scenarios}')
    lines.append(r'\label{tab:scenarios}')
    lines.append(r'\centering')

    #Dynamic columns: one per scenario plus model and type.
    scenario_list = list(SCENARIOS.keys())
    col_format    = 'll' + 'c' * len(scenario_list)
    lines.append(f'\\begin{{tabular}}{{{col_format}}}')
    lines.append(r'\hline')

    #Header row with scenario names.
    header = r'\textbf{Model} & \textbf{Type} & ' + \
             ' & '.join([f'\\textbf{{{s}}}' for s in scenario_list]) + r' \\'
    lines.append(header)
    lines.append(r'\hline')

    #One row per model.
    for model_name in UNSUPERVISED_MODELS + SUPERVISED_MODELS:
        model_type = 'Unsupervised' if model_name in UNSUPERVISED_MODELS else 'Supervised'
        row_vals   = []
        for scenario in scenario_list:
            match = results_df[
                (results_df['model'] == model_name) &
                (results_df['scenario'] == scenario) &
                (results_df['anomaly_rate'].notna())
            ]
            val = f"{match['anomaly_rate'].values[0]:.1f}\\%" if not match.empty else 'N/A'
            row_vals.append(val)
        lines.append(f"{model_name} & {model_type} & " + ' & '.join(row_vals) + r' \\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    tex_path = f'{OUTPUTS_DIR}table_scenario_results.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n[SAVED] Saved: table_scenario_results.tex")

    #Table 2: False positive rate on healthy logs.
    lines2 = []
    lines2.append(r'\begin{table}[htbp]')
    lines2.append(r'\caption{False Positive Rate on Healthy Authentication Logs}')
    lines2.append(r'\label{tab:false_positives}')
    lines2.append(r'\centering')
    lines2.append(r'\begin{tabular}{llcc}')
    lines2.append(r'\hline')
    lines2.append(r'\textbf{Model} & \textbf{Type} & \textbf{False Positive Rate (\%)} & \textbf{Verdict} \\')
    lines2.append(r'\hline')

    healthy = results_df[
        (results_df['scenario'] == 'Healthy Logs') &
        (results_df['anomaly_rate'].notna())
    ]
    for model_name in UNSUPERVISED_MODELS + SUPERVISED_MODELS:
        match = healthy[healthy['model'] == model_name]
        if match.empty:
            continue
        rate      = match['anomaly_rate'].values[0]
        model_type = 'Unsupervised' if model_name in UNSUPERVISED_MODELS else 'Supervised'
        verdict   = 'PASS' if rate == 0 else 'ACCEPTABLE' if rate < 2 else 'CAUTION' if rate < 10 else 'FAIL'
        lines2.append(f"{model_name} & {model_type} & {rate:.1f}\\% & {verdict} \\\\")

    lines2.append(r'\hline')
    lines2.append(r'\end{tabular}')
    lines2.append(r'\end{table}')

    tex_path2 = f'{OUTPUTS_DIR}table_false_positive_analysis.tex'
    with open(tex_path2, 'w') as f:
        f.write('\n'.join(lines2))
    print(f"[SAVED] Saved: table_false_positive_analysis.tex")

def export_dataset_table():
    """
    Exports an IEEE-style LaTeX table describing all datasets
    used in the scenario testing — goes in the Dataset section
    of the IEEE paper.
    """
    #Dataset details: name, path, size, type and purpose
    datasets = [
    #Training datasets
    ('Dataset13k', 'Training', '13,010', 'Labeled', 'Model training and evaluation'),
    #Testing scenarios
    ('Dataset1k', 'Testing', '1,000', 'Unlabeled', 'Small scale scenario test'),
    ('Dataset15k', 'Testing', '15,000', 'Unlabeled', 'Medium scale scenario test'),
    ('Dataset20k', 'Testing', '20,000', 'Unlabeled', 'Large scale scenario test'),
    ('Dataset100k', 'Testing', '100,000', 'Unlabeled', 'Very large scale scenario test'),
    ('Healthy Logs Sample', 'Testing', '~50', 'Unlabeled', 'False positive evaluation'),
    ('Empty Dataset', 'Testing', '0', 'N/A', 'Edge case robustness test'),
    ]

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\caption{Dataset Description — Azure Authentication Records}')
    lines.append(r'\label{tab:datasets}')
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{lllll}')
    lines.append(r'\hline')
    lines.append(r'\textbf{Dataset} & \textbf{Split} & \textbf{Records} & \textbf{Labels} & \textbf{Purpose} \\')
    lines.append(r'\hline')

    for name, split, size, labels, purpose in datasets:
        lines.append(f"{name} & {split} & {size} & {labels} & {purpose} \\\\")

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex_path = f'{OUTPUTS_DIR}table_datasets.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"[SAVED] Saved: table_datasets.tex")

#############################################################################################
#Part 8: Main
#############################################################################################

def main():
    print("#" * 60)
    print("  SCENARIO-BASED BRUTE FORCE DETECTION TESTING")
    print("  Azure Authentication Records")
    print("#" * 60)
    print("\n  Testing pre-trained models across 6 test scenarios")
    print("  to evaluate robustness of brute force attack detection.\n")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)  #Create outputs folder if it doesn't exist.

    #Load pre-trained models and vectorizer.
    #Models must have been saved by Main_AD_Code.py before this runs.
    loaded_models, vectorizer = load_models_and_vectorizer()

    if not loaded_models or vectorizer is None:
        print("\n[FAILURE] Could not load models or vectorizer.")
        print("    Make sure Main_AD_Code.py has been run first.")
        return

    #Run all 6 scenarios
    print("\n[2] Running scenario tests...\n")
    all_results = []

    for scenario_name, file_path in SCENARIOS.items():
        results = run_scenario(scenario_name, file_path, loaded_models, vectorizer)
        all_results.extend(results)  #Add this scenario's results to the master list.

    results_df = pd.DataFrame(all_results)  #Convert full results list to DataFrame.

    #Save results to CSV.
    results_df.to_csv(f'{OUTPUTS_DIR}scenario_results.csv', index=False)
    print(f"\n[SUCCESS] Results saved: scenario_results.csv")

    #Print summary to console.
    print_scenario_summary(results_df)

    #Generate all 4 charts
    print("\n[3] Generating charts...")
    plot_anomaly_rates(results_df)          #Detection rate per model per scenario.
    plot_false_positive_analysis(results_df) #False positive rate on healthy logs.
    plot_heatmap(results_df)                #Full heatmap: models × scenarios.
    plot_detection_consistency(results_df)  #Detection rate vs dataset size.

    print("\n[SUCCESS] All done! Check '../outputs/' for all charts and results.")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    export_dataset_table()  #Export dataset description table for our IEEE-style paper.
    plot_flagged_vs_ground_truth(results_df)  #Flagged rate vs ground truth F1.

#Only runs main() when this file is executed directly.
#Prevents main() from running if this file is imported by another script.

if __name__ == "__main__":
    main()

