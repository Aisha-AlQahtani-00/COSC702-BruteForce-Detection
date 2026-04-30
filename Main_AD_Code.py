#############################################################################################
#Spring 2026
#COSC 702: Advanced AI-Driven Software Engineering
#Project: Benchmarking the Detection of Cloud-based User Authentication Log Anomalies Using Machine Learning Techniques
#Code Component: Main Anomaly Detection Code
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
#Part 1: Unsupervised Models
#############################################################################################

def run_unsupervised_models(tfidf_matrix, true_labels_unsup):
    """Train and evaluate 5 unsupervised models. Save all as .pkl files."""
    results  = {} #Stores evaluation metrics for each model.
    reduced  = get_reduced(tfidf_matrix) #Compress the TF-IDF matrix into 50 dense dimensions once, which is shared by the Elliptic Envelope and One-Class SVM so it doesn't run twice.

    #Stores raw predictions (-1/1) and anomaly scores for the comparison plots.
    all_preds  = {}
    all_scores = {}

    #------------------------------------------------------------------------------#
    # 1. Isolation Forest

    print("#" * 60)
    print("UNSUPERVISED MODEL 1: Isolation Forest")
    print("#" * 60)
    iso = IsolationForest(contamination=0.3, random_state=42) #Contamination=0.3 tells the model to expect ~30% anomalies. random_state=42 ensure reproducible results.
    iso.fit(tfidf_matrix) #Trains on the full TF-IDF matrix.
    preds_iso  = iso.predict(tfidf_matrix) #Returns -1 (anomaly) or 1 (normal) for each record.
    scores_iso = iso.decision_function(tfidf_matrix) #Returns anomaly scores (More negative = more anomalous)
    results['Isolation Forest'] = evaluate_unsupervised(
        true_labels_unsup, preds_iso, scores_iso, "Isolation Forest"
    )
    all_preds['Isolation Forest']  = preds_iso
    all_scores['Isolation Forest'] = scores_iso
    joblib.dump(iso, '../models/isolation_forest.pkl') #Saves trained model to disk for Scenarios_Test.py
    print("[SAVED] Saved: isolation_forest.pkl")

    #------------------------------------------------------------------------------#
    #2. K-Means Clustering

    print("\n" + "#" * 60)
    print("UNSUPERVISED MODEL 2: K-Means Clustering")
    print("#" * 60)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) #2 clusters = normal vs attack. n_init=10 runs 10 initializations and picks the best.
    kmeans.fit(tfidf_matrix)
    _, distances = pairwise_distances_argmin_min(tfidf_matrix, kmeans.cluster_centers_) #Memory-efficient distance to nearest cluster center, which avoids the SIGKILL crash on large datasets encountered before.
    threshold_km = np.percentile(distances, 70) #Top 30% most distant points are flagged as anomalies.
    preds_km     = np.where(distances >= threshold_km, -1, 1) #-1 for anomaly, 1 for normal.
    scores_km    = -distances #Negate distance so higher score = more anomalous (Consistent with other models).
    results['K-Means Clustering'] = evaluate_unsupervised(
        true_labels_unsup, preds_km, scores_km, "K-Means Clustering"
    )
    all_preds['K-Means Clustering']  = preds_km
    all_scores['K-Means Clustering'] = scores_km
    joblib.dump(kmeans, '../models/kmeans.pkl')
    print("[SAVED] Saved: kmeans.pkl")

    #------------------------------------------------------------------------------#

    #3. Elliptic Envelope
    print("\n" + "#" * 60)
    print("UNSUPERVISED MODEL 3: Elliptic Envelope")
    print("#" * 60)

    ee = EllipticEnvelope(contamination=0.3, support_fraction=0.9, random_state=42) #support_fraction = 0.9 uses 90% of data to fit covariance matrix, preventing matrix errors.
    ee.fit(reduced) #Uses SVD-reduces matrix; Elliptic Envelope requires dense input.
    preds_ee = ee.predict(reduced)
    scores_ee = ee.decision_function(reduced)

    #Replace NaN scores with the minimum non-NaN score (Since we had a Nan score for AUC-ROC value before).
    #NaN means the point was too far outside the covariance envelope.
    nan_mask = np.isnan(scores_ee)
    if nan_mask.any():
        print(f"  [!] {nan_mask.sum()} NaN scores detected in Elliptic Envelope — replacing with min score")
        scores_ee[nan_mask] = np.nanmin(scores_ee) #NaN points are treated as most anomalous since they're furthest outside the envelope.

    results['Elliptic Envelope'] = evaluate_unsupervised(
        true_labels_unsup, preds_ee, scores_ee, "Elliptic Envelope"
    )
    all_preds['Elliptic Envelope']  = preds_ee
    all_scores['Elliptic Envelope'] = scores_ee
    joblib.dump(ee, '../models/elliptic_envelope.pkl')
    print("[SAVED] Saved: elliptic_envelope.pkl")

    #------------------------------------------------------------------------------#

    #4. Local Outlier Factor
    print("\n" + "#" * 60)
    print("UNSUPERVISED MODEL 4: Local Outlier Factor (LOF)")
    print("#" * 60)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.3, novelty=True) #novelty=True is required to allow predict() on new unseen data in Scenarios_Test.py
    lof.fit(tfidf_matrix)
    preds_lof  = lof.predict(tfidf_matrix)
    scores_lof = lof.decision_function(tfidf_matrix) #Higher score = more normal, lower = more anomalous
    results['Local Outlier Factor'] = evaluate_unsupervised(
        true_labels_unsup, preds_lof, scores_lof, "Local Outlier Factor"
    )
    all_preds['Local Outlier Factor']  = preds_lof
    all_scores['Local Outlier Factor'] = scores_lof
    joblib.dump(lof, '../models/lof.pkl')
    print("[SAVED] Saved: lof.pkl")

    #------------------------------------------------------------------------------#

    #5. One-Class SVM
    print("\n" + "#" * 60)
    print("UNSUPERVISED MODEL 5: One-Class SVM")
    print("#" * 60)

    oc_svm = OneClassSVM(nu=0.3, kernel='rbf', gamma='scale') #nu=0.3 is equivalent to contamination; upper bound on anomaly fraction. gamme='scale' auto-adjusts to the feature space.
    oc_svm.fit(reduced) #Uses SVD-reduced matrix; One-Class SVM requires dense input.
    preds_ocsvm  = oc_svm.predict(reduced)
    scores_ocsvm = oc_svm.decision_function(reduced)
    results['One-Class SVM'] = evaluate_unsupervised(
        true_labels_unsup, preds_ocsvm, scores_ocsvm, "One-Class SVM"
    )
    all_preds['One-Class SVM']  = preds_ocsvm
    all_scores['One-Class SVM'] = scores_ocsvm
    joblib.dump(oc_svm, '../models/one_class_svm.pkl')
    print("[SAVED] Saved: one_class_svm.pkl")

    return results, all_preds, all_scores #This is where all metrics, predictions, and scores for use in comparison plots are returned.

#------------------------------------------------------------------------------#


def evaluate_unsupervised(true_labels, predicted_labels, scores, model_name):
    """Compute and print metrics for an unsupervised model."""
    #Convert -1/1 to 0/1 binary; where unsupervised models output -1 (anomaly) and 1 (normal).
    #However, metric functions expect 1 (anomaly) and 0 (normal).
    true_binary = np.where(np.array(true_labels)      == -1, 1, 0)
    pred_binary = np.where(np.array(predicted_labels) == -1, 1, 0)

    acc  = accuracy_score(true_binary, pred_binary)
    prec = precision_score(true_binary, pred_binary, zero_division=0) #zero_division=0 prevents error when a model predicts no anomalies.
    rec  = recall_score(true_binary, pred_binary, zero_division=0)
    f1   = f1_score(true_binary, pred_binary, zero_division=0)

    try:
        auc = roc_auc_score(true_binary, scores) #Uses raw scores not binary predictions for a smoother ROC curve.
    except Exception as e:
        print(f"  [!] AUC-ROC failed for {model_name}: {e}")
        auc = float('nan')

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n{classification_report(true_binary, pred_binary, target_names=['Normal', 'Brute Force'], zero_division=0)}")

#Return metrics dict; also includes raw arrays needed for confusion matrices,
#ROC curves, PR curves, and the model agreement heatmap.
    return {
        'model': model_name, 'type': 'Unsupervised',
        'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1, 'auc_roc': auc,
        'true_binary': true_binary,
        'pred_binary': pred_binary,
        'scores': scores
    }

#############################################################################################
#Part 2: Supervised Models
#############################################################################################

def run_supervised_models(tfidf_matrix, true_labels_sup):
    """Train and evaluate 5 supervised models. Save all as .pkl files."""
    results  = {} #Stores evaluation metrics for each model.
    all_preds  = {} #Stores raw predictions (0/1) for comparison.
    all_scores = {} #Stores probability scores for ROC and PR curves.

    X = tfidf_matrix
    y = np.array(true_labels_sup)

    #80/20 train/test split; stratify=y preserves the 79/21 Normal/ Anomaly
    #Ratio in both sets so the test set always contain enough attack examples.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[SUCCESS] Supervised split — Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}\n")

    #------------------------------------------------------------------------------#
    #1. Random Forest

    print("#" * 60)
    print("SUPERVISED MODEL 1: Random Forest")
    print("#" * 60)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    #n_estimators=100 builds 100 decision trees and combines their votes.
    #class_weight='balanced' increases the penalty for missing the minority attack class (20.6%).
    rf.fit(X_train, y_train)
    res, preds, scores = evaluate_supervised(rf, X_test, y_test, "Random Forest")
    results['Random Forest'] = res
    all_preds['Random Forest']  = preds
    all_scores['Random Forest'] = scores
    joblib.dump(rf, '../models/random_forest.pkl')
    print("[SAVED] Saved: random_forest.pkl") #Saved for Scenarios_Tes.py.

    #------------------------------------------------------------------------------
    #2. SVM
    print("\n" + "#" * 60)
    print("SUPERVISED MODEL 2: SVM (RBF Kernel)")
    print("#" * 60)

    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    #kernel='rbf' handles non-linearly separable data.
    #probability=True enables predict_proba() needed for ROC and PR curves.
    svm.fit(X_train, y_train)
    res, preds, scores = evaluate_supervised(svm, X_test, y_test, "SVM")
    results['SVM'] = res
    all_preds['SVM']  = preds
    all_scores['SVM'] = scores
    joblib.dump(svm, '../models/svm.pkl')
    print("[SUCCESS] Saved: svm.pkl")

    #------------------------------------------------------------------------------
    #3. Logistic Regression
    print("\n" + "#" * 60)
    print("SUPERVISED MODEL 3: Logistic Regression")
    print("#" * 60)

    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    #max_iter=1000 gives enough iterations to converge on the high-dimensional TF-IDF space.
    lr.fit(X_train, y_train)
    res, preds, scores = evaluate_supervised(lr, X_test, y_test, "Logistic Regression")
    results['Logistic Regression'] = res
    all_preds['Logistic Regression']  = preds
    all_scores['Logistic Regression'] = scores
    joblib.dump(lr, '../models/logistic_regression.pkl')
    print("[SAVED] Saved: logistic_regression.pkl")

    #------------------------------------------------------------------------------
    #4. Gradient Boosting
    print("\n" + "#" * 60)
    print("SUPERVISED MODEL 4: Gradient Boosting")
    print("#" * 60)

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    #Builds 100 trees sequentially, each correcting the errors of the previous, learning_rate=0.1 controls how much each tree corrects (Lower is more conservative).
    gb.fit(X_train, y_train)
    res, preds, scores = evaluate_supervised(gb, X_test, y_test, "Gradient Boosting")
    results['Gradient Boosting'] = res
    all_preds['Gradient Boosting']  = preds
    all_scores['Gradient Boosting'] = scores
    joblib.dump(gb, '../models/gradient_boosting.pkl')
    print("[SAVED] Saved: gradient_boosting.pkl")

    #------------------------------------------------------------------------------
    #5. KNN
    print("\n" + "#" * 60)
    print("SUPERVISED MODEL 5: K-Nearest Neighbors (KNN)")
    print("#" * 60)
    knn = KNeighborsClassifier(n_neighbors=5)
    #Classifies each record by majority vote of its 5 nearest neighbors in the TD-IDF space.
    knn.fit(X_train, y_train)
    res, preds, scores = evaluate_supervised(knn, X_test, y_test, "KNN")
    results['KNN'] = res
    all_preds['KNN']  = preds
    all_scores['KNN'] = scores
    joblib.dump(knn, '../models/knn.pkl')
    print("[SAVED] Saved: knn.pkl")

    return results, all_preds, all_scores, y_test #y_test returned separately for the model agreement plot.


def evaluate_supervised(model, X_test, y_test, model_name):
    """Compute and print metrics for a supervised model."""
    preds  = model.predict(X_test) #Binary predictions- 0 (normal) or 1 (attack)
    scores = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else preds #Probability of being an attack; used for smoother ROC/ PR curves.

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)

    try:
        auc = roc_auc_score(y_test, scores)
    except Exception:
        auc = float('nan')

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n{classification_report(y_test, preds, target_names=['Normal', 'Brute Force'], zero_division=0)}")

    #Return metrics dict including raw arrays needed for confusion matrices:
    #ROC Curves, PR curves and the model agreement heatmap.
    result = {
        'model': model_name, 'type': 'Supervised',
        'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1, 'auc_roc': auc,
        'true_binary': y_test,
        'pred_binary': preds,
        'scores': scores
    }
    return result, preds, scores

#############################################################################################
#Part 3: Comparison Table
#############################################################################################


def build_comparison_table(all_results):
    """Build a pandas DataFrame from all model results."""
    #String out the raw arrays (true_binary, pred_binary, scores) from each result
    #Those are only needed for plots and would clutter the comparison table.
    rows = [{k: v for k, v in r.items()
             if k not in ('true_binary', 'pred_binary', 'scores')}
            for r in all_results.values()]
    df   = pd.DataFrame(rows)
    #Sort by type first (Supervised/Unsupervised) then by F1 descending so best models appear at the top.
    df   = df.sort_values(['type', 'f1'], ascending=[True, False]).reset_index(drop=True)
    return df


def print_comparison_table(df):
    """Print the full comparison table."""
    print("\n")
    print("#" * 80)
    print("         FULL COMPARATIVE RESULTS — ALL 10 MODELS")
    print("#" * 80)
    # :<25 and :>7 are format specifiers; left alignment name in 25 chars, right align metrics in 7 chars
    print(f"{'Model':<25} {'Type':<14} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC-ROC':>9}")
    print("-" * 80)
    for _, row in df.iterrows(): #Iterate over each model's row.
        print(
            f"{row['model']:<25} {row['type']:<14} "
            f"{row['accuracy']:>7.4f} {row['precision']:>7.4f} "
            f"{row['recall']:>7.4f} {row['f1']:>7.4f} {row['auc_roc']:>9.4f}"  # .4f = 4 decimal places.
        )
    print("=" * 80)

    #IEEE-style LaTeX table to be used for later.
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\caption{Comparative Results of Brute Force Attack Detection Models}')
    lines.append(r'\label{tab:comparison}')
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{llccccc}')
    lines.append(r'\hline')  #Top rule
    lines.append(
        r'\textbf{Model} & \textbf{Type} & \textbf{Acc} & \textbf{Prec} & \textbf{Rec} & \textbf{F1} & \textbf{AUC-ROC} \\')
    lines.append(r'\hline')  #Header rule

    for _, row in df.iterrows():
        #Format AUC-ROC as N/A if NaN (e.g. Elliptic Envelope)
        auc = f"{row['auc_roc']:.4f}" if not pd.isna(row['auc_roc']) else 'N/A'
        lines.append(
            f"{row['model']} & {row['type']} & "
            f"{row['accuracy']:.4f} & {row['precision']:.4f} & "
            f"{row['recall']:.4f} & {row['f1']:.4f} & {auc} \\\\"
        )

    lines.append(r'\hline')  #Bottom rule
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    #Save to file
    tex_path = '../outputs/table_model_comparison.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"[SAVED] Saved: table_model_comparison.tex")


#############################################################################################
#Part 4: Algorithm Comparison Plots
#############################################################################################

def plot_grouped_bar_comparison(df):
    metrics       = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    model_order = UNSUPERVISED_MODELS + SUPERVISED_MODELS #Reorder rows to match model_order
    df_ordered  = df.set_index('model').reindex(model_order).reset_index()
    colors      = ['#2C7BB6' if t == 'Unsupervised' else '#D7191C'
                   for t in df_ordered['type']] #Blue for unsupervised, red for supervised.

    fig, axes = plt.subplots(5, 1, figsize=(16, 20)) #5 rows; ine subplot per metric.
    fig.suptitle('Brute Force Detection — Algorithm Comparison (All 10 Models)',
                 fontsize=14, fontweight='bold')

    for ax, metric, label in zip(axes, metrics, metric_labels): #Loop through each metric and its subplot.
        bars = ax.bar(
            df_ordered['model'], df_ordered[metric],
            color=colors, edgecolor='white', linewidth=0.5, alpha=0.9
        )
        #Divider between unsupervised and supervised
        ax.axvline(x=4.5, color='grey', linestyle='--', alpha=0.4, linewidth=1) #Vertical line separating unsupervised (lef) from supervised (right).
        ax.set_ylabel(label, fontweight='bold', fontsize=10)
        ax.set_ylim(0, 1.15) #Extra 0.15 headroom above 1.0 for value labels.
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False) #Remove chart borders for a cleaner look.

        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h): #Skip NaN values (e.g. Elliptic Envelope AUC-ROC).
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=8) #Value label above each bar.

    patch1 = mpatches.Patch(color='#2C7BB6', label='Unsupervised')
    patch2 = mpatches.Patch(color='#D7191C', label='Supervised')
    fig.legend(handles=[patch1, patch2], loc='upper right', fontsize=10) #Single shared legend for all subplots.
    plt.tight_layout()
    plt.savefig('../outputs/chart_algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: chart_algorithm_comparison.png")


def plot_f1_ranking(df):
    """Horizontal bar chart: All 10 models ranked by F1-Score."""
    df_sorted = df.sort_values('f1', ascending=True).reset_index(drop=True) #Ascending to highest F1 appears at the top.
    colors    = ['#2C7BB6' if t == 'Unsupervised' else '#D7191C' for t in df_sorted['type']]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df_sorted['model'], df_sorted['f1'], #barh = horizontal bar chart.
                   color=colors, edgecolor='white', linewidth=0.5, alpha=0.9)
    ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Brute Force Detection — F1-Score Ranking (All 10 Models)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    for bar, val in zip(bars, df_sorted['f1']):
        if not np.isnan(val):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9) #Value label to the right of each bar.

    patch1 = mpatches.Patch(color='#2C7BB6', label='Unsupervised')
    patch2 = mpatches.Patch(color='#D7191C', label='Supervised')
    ax.legend(handles=[patch1, patch2], fontsize=10)
    plt.tight_layout()
    plt.savefig('../outputs/chart_f1_ranking.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SUCCESS] Saved: chart_f1_ranking.png")


def plot_confusion_matrices(all_results):
    """
    Confusion matrices for all 10 models in one figure.
    Critical for cybersecurity — shows false negatives (missed attacks)
    vs false positives (false alarms) for each algorithm.
    """
    model_order = UNSUPERVISED_MODELS + SUPERVISED_MODELS
    fig, axes   = plt.subplots(2, 5, figsize=(22, 9)) #2 rows x 5 cols; one matrix per model.
    fig.suptitle('Confusion Matrices — Brute Force Attack Detection (All 10 Models)\n'
                 'TN=Correct Normal | FP=False Alarm | FN=Missed Attack | TP=Correct Detection',
                 fontsize=13, fontweight='bold')

    axes_flat = axes.flatten() #Convert 2D grid of axes into a flat list for easy indexing.

    for idx, model_name in enumerate(model_order):
        if model_name not in all_results:
            continue

        res    = all_results[model_name]
        y_true = res['true_binary']
        y_pred = res['pred_binary']
        cm     = confusion_matrix(y_true, y_pred) #Returns [[TN, FP], [FN, TP]].
        ax     = axes_flat[idx]

        #Green background for correct cells (TN, TP), red for errors (FP, FN).
        cm_colors = np.array([
            ['#2ecc71', '#e74c3c'],
            ['#e74c3c', '#2ecc71']
        ])

        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    color=cm_colors[i][j], alpha=0.3 #Transparent coloured background per cell.
                ))

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', # fmt='d' shows integers not floats.
            xticklabels=['Normal', 'Brute Force'],
            yticklabels=['Normal', 'Brute Force'],
            ax=ax, cbar=False, linewidths=0.5,
            annot_kws={'size': 11, 'weight': 'bold'}
        )

        model_type = 'Unsupervised' if model_name in UNSUPERVISED_MODELS else 'Supervised'
        color      = '#2C7BB6' if model_type == 'Unsupervised' else '#D7191C'
        ax.set_title(f'{model_name}\n({model_type})',
                     fontsize=9, fontweight='bold', color=color)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('Actual', fontsize=8)
        ax.tick_params(labelsize=8)

        #Highlight FN (bottom-left cell0; missed attacks are the most critical failures in cybersecurity.
        if cm.shape == (2, 2):
            fn = cm[1][0] #Row 1 = actual attack, Col 0 = predicted normal = missed attack.
            ax.text(0, 1, f'Missed\nAttacks\n{fn}',
                    ha='center', va='center', fontsize=7,
                    color='#c0392b', fontweight='bold')

    plt.tight_layout()
    plt.savefig('../outputs/chart_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SUCCESS] Saved: chart_confusion_matrices.png")


def plot_roc_curves(all_results):
    """
    ROC curves for all 10 models on one chart.
    The gold standard for comparing classifiers —
    shows the tradeoff between detection rate and false alarm rate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7)) #Split into unsupervised (lef) and unsupervised (right).
    fig.suptitle('ROC Curves — Brute Force Attack Detection\n'
                 '(Higher AUC = better at separating attacks from normal traffic)',
                 fontsize=13, fontweight='bold')

    for ax, model_group, title in zip(
        axes,
        [UNSUPERVISED_MODELS, SUPERVISED_MODELS],
        ['Unsupervised Models', 'Supervised Models']
    ):
        for model_name in model_group:
            if model_name not in all_results:
                continue

            res    = all_results[model_name]
            y_true = res['true_binary']
            scores = res['scores']

            try:
                fpr, tpr, _ = roc_curve(y_true, scores) #fpr = false alarm rate, tpr = detection rate each threshold.
                auc         = roc_auc_score(y_true, scores)
                ax.plot(
                    fpr, tpr,
                    color=MODEL_COLORS.get(model_name, 'grey'),
                    linewidth=2,
                    label=f'{model_name} (AUC={auc:.3f})'
                )
            except Exception:
                pass #Skip models whose scores can't produce a valid ROC curve.

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='Random (AUC=0.5)') #Diagonal = a model that guesses randomly.
        ax.set_xlabel('False Positive Rate (False Alarm Rate)', fontweight='bold', fontsize=10)
        ax.set_ylabel('True Positive Rate (Attack Detection Rate)', fontweight='bold', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig('../outputs/chart_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: chart_roc_curves.png")


def plot_precision_recall_curves(all_results):
    """
    Precision-Recall curves for all 10 models.
    More informative than ROC for imbalanced datasets (79/21 split).
    Shows the tradeoff between catching all attacks vs avoiding false alarms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Precision-Recall Curves — Brute Force Attack Detection\n'
                 '(More informative than ROC for imbalanced data | Higher AP = better)',
                 fontsize=13, fontweight='bold')

    for ax, model_group, title in zip(
        axes,
        [UNSUPERVISED_MODELS, SUPERVISED_MODELS],
        ['Unsupervised Models', 'Supervised Models']
    ):
        for model_name in model_group:
            if model_name not in all_results:
                continue

            res    = all_results[model_name]
            y_true = res['true_binary']
            scores = res['scores']

            try:
                prec, rec, _ = precision_recall_curve(y_true, scores) #Precision and recall at every threshold.
                ap           = average_precision_score(y_true, scores) #Single number summary of the curve.
                ax.plot(
                    rec, prec,
                    color=MODEL_COLORS.get(model_name, 'grey'),
                    linewidth=2,
                    label=f'{model_name} (AP={ap:.3f})'
                )
            except Exception:
                pass

        #A random classifier would achieve precision equal to the attack rate (20%).
        baseline = 0.206
        ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.4,
                   linewidth=1, label=f'Baseline (AP={baseline:.3f})')

        ax.set_xlabel('Recall (Attack Detection Rate)', fontweight='bold', fontsize=10)
        ax.set_ylabel('Precision (Accuracy of Attack Flags)', fontweight='bold', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('../outputs/chart_precision_recall_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: chart_precision_recall_curves.png")


def plot_model_agreement(all_results):
    """
    Model agreement heatmap — shows what % of records each pair
    of models agree on flagging as a brute force attack.
    High agreement = models are learning similar patterns.
    Low agreement = models capture different aspects of attacks.
    """
    model_order = UNSUPERVISED_MODELS + SUPERVISED_MODELS
    models_avail = [m for m in model_order if m in all_results]


    #Unsupervised models predict on all 13,010 records, supervised only on the 2,602 test set.
    #Trim all arrays to the shortest length so they can be stacked into one matrix.
    min_len = min(len(all_results[m]['pred_binary']) for m in models_avail)
    pred_matrix = np.column_stack([
        all_results[m]['pred_binary'][:min_len] for m in models_avail
    ])

    #For each pair of models, calculate what % of records they both predicted the same way.
    n = len(models_avail)
    agreement = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            agreement[i, j] = np.mean(pred_matrix[:, i] == pred_matrix[:, j]) * 100

    agreement_df = pd.DataFrame(agreement, index=models_avail, columns=models_avail)

    fig, ax = plt.subplots(figsize=(13, 10))
    mask    = np.eye(n, dtype=bool)   #Mask the diagnal; a model always agrees 100% with itself, not informative.

    sns.heatmap(
        agreement_df,
        annot=True, fmt='.1f',
        cmap='YlOrRd', #Yellow = low agreement, red = high agreement
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Agreement (% of records with same prediction)'},
        ax=ax,
        mask=mask,
        vmin=50, vmax=100, #Scale starts at 50% since random agreement would already be ~50%.
        annot_kws={'size': 9}
    )

    #Manually fill the masked diagnomal with a readable label.
    for i in range(n):
        ax.text(i + 0.5, i + 0.5, '100%\n(self)',
                ha='center', va='center', fontsize=8,
                color='grey', fontweight='bold')

    #Whie lines separating the unsupervised block from the supervised block.
    n_unsup = len([m for m in models_avail if m in UNSUPERVISED_MODELS])
    ax.axhline(y=n_unsup, color='white', linewidth=3)
    ax.axvline(x=n_unsup, color='white', linewidth=3)

    ax.set_title('Model Agreement Heatmap — Brute Force Detection\n'
                 '(% of authentication records where both models agree on the prediction)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

    #Qudrant labels above the heatmap.
    ax.text(n_unsup / 2, -0.8, 'Unsupervised',
            ha='center', fontsize=9, color='#2C7BB6', fontweight='bold')
    ax.text(n_unsup + (n - n_unsup) / 2, -0.8, 'Supervised',
            ha='center', fontsize=9, color='#D7191C', fontweight='bold')

    plt.tight_layout()
    plt.savefig('../outputs/chart_model_agreement.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: chart_model_agreement.png")


def plot_radar_best_models(df):
    """Radar chart comparing best unsupervised vs best supervised model."""
    metrics      = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    labels_radar = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']

    unsup_df   = df[df['type'] == 'Unsupervised'].reset_index(drop=True)
    sup_df     = df[df['type'] == 'Supervised'].reset_index(drop=True)
    best_unsup = unsup_df.loc[unsup_df['f1'].idxmax()] #Pick the unsupervised with the highest F1.
    best_sup   = sup_df.loc[sup_df['f1'].idxmax()] #Pick the supervised model with the highest F1.

    #Append first value to the end of list to close the radar polygon.
    vals_unsup = [best_unsup[m] for m in metrics] + [best_unsup[metrics[0]]]
    vals_sup   = [best_sup[m]   for m in metrics] + [best_sup[metrics[0]]]
    angles     = np.linspace(0, 2 * np.pi, len(labels_radar), endpoint=False).tolist()
    angles    += angles[:1] #Close the polygon by repeating the first angle.

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True)) #polar=True creates the circular radar layout.
    ax.plot(angles, vals_unsup, 'o-', color='#2C7BB6', linewidth=2, label=best_unsup['model'])
    ax.fill(angles, vals_unsup, color='#2C7BB6', alpha=0.2) #Shaded area for unsupervised model.
    ax.plot(angles, vals_sup,   's-', color='#D7191C', linewidth=2, label=best_sup['model'])
    ax.fill(angles, vals_sup,   color='#D7191C', alpha=0.2) #Shaded area for supervised model.

    ax.set_thetagrids(np.degrees(angles[:-1]), labels_radar, fontsize=11) #Place metric labels around the radar.
    ax.set_ylim(0, 1)
    ax.set_title('Best Model Comparison\n(Best Unsupervised vs Best Supervised)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../outputs/chart_radar_best_models.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] Saved: chart_radar_best_models.png")


#############################################################################################
#Part 5: Main Code
#############################################################################################

def main():
    file_path         = "/Users/aisha/Desktop/COSC_702/Project/Datasets/Training/Dataset13k.csv"
    labeled_data_path = "/Users/aisha/Desktop/COSC_702/Project/Datasets/Training/Labeled13k.csv"

    os.makedirs('../outputs', exist_ok=True) #Creates the outputs folder if it doesn't already exist.
    os.makedirs('../models',  exist_ok=True) #Creates the model folder if it does not already exist.

    #------------------------------------------------------------------------------
    #Load data
    df_main    = load_data(file_path) #Unlabeled full dataset.
    labeled_df = load_labeled_data(labeled_data_path) #Manually labeled dataset with ground truth.

    if df_main is None or labeled_df is None:
        print("[FAILURE] Could not load data. Exiting.")
        return #Stop execution if either file failed to load.

    #------------------------------------------------------------------------------
    #Combine columns & vectorize
    print("[SUCCESS] Combining feature columns for vectorization...")
    combined_text        = combine_columns(labeled_df, FEATURE_COLUMNS) #Merges Status, Location, IP, Application into one string per row.
    tfidf_matrix, vectorizer = vectorize_text(combined_text) #Convers combines text into a numerical TF-IDF matrix.
    joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl') #Save vectorizer, Scenarios_Test.py must use the same vocabulary when transforming new data.
    print("[SAVED] Saved: tfidf_vectorizer.pkl\n")

    true_labels_unsup = labeled_df['label_unsup'].values #Ground truth in -1/1 format for unsupervised models.
    true_labels_sup   = labeled_df['label_sup'].values #Ground truth in 0/1 format for supervised models.

    #------------------------------------------------------------------------------
    #Run unsupervised models
    print("\n" + "#" * 60)
    print("#       UNSUPERVISED ANOMALY DETECTION (5 Models)      #")
    print("#" * 60)
    unsup_results, unsup_preds, unsup_scores = run_unsupervised_models(
        tfidf_matrix, true_labels_unsup
    )

    #------------------------------------------------------------------------------
    #Run supervised models
    print("\n" + "#" * 60)
    print("#        SUPERVISED ANOMALY DETECTION (5 Models)       #")
    print("#" * 60)
    sup_results, sup_preds, sup_scores, y_test = run_supervised_models(
        tfidf_matrix, true_labels_sup
    )

    #------------------------------------------------------------------------------
    #Merge all results
    all_results   = {**unsup_results, **sup_results} #Merge both dict into one ** unpacks each dict.
    comparison_df = build_comparison_table(all_results) #Build the formatted results DataFrame.
    print_comparison_table(comparison_df) #Print to console.
    comparison_df.to_csv('../outputs/model_comparison_results.csv', index=False) #Save to CSV.
    print("[SAVED] Saved: model_comparison_results.csv")

    #------------------------------------------------------------------------------
    #Generate algorithm comparison plots
    #All 7 charts saved to ../outputs/ ; must be run after both model groups are complete.
    print("\n[SAVING] Generating algorithm comparison charts...")
    plot_grouped_bar_comparison(comparison_df) #5 metric bar charts for all 10 models.
    plot_f1_ranking(comparison_df) #All models ranked by F1.
    plot_confusion_matrices(all_results) #2x5 grid of confusion matrices.
    plot_roc_curves(all_results) #ROC curves splits by model type.
    plot_precision_recall_curves(all_results) #PR curves with baseline.
    plot_model_agreement(all_results) #Pairwise agreement.
    plot_radar_best_models(comparison_df) #Radar chart of best unsupervised vs supervised.

    print("\n[COMPLETE] All done! Check '../outputs/' and '../models/' for results.")
    print(" You can now run Ablation_Study.py and Scenario_Test.py!")

#------------------------------------------------------------------------------
#Only runs main() when this files is executed directly.
#Prevents main() from running if this file is imported by another script.
if __name__ == "__main__":
    main()