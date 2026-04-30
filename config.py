#############################################################################################
#Spring 2026
#COSC 702: Advanced AI-Driven Software Engineering
#Project: Benchmarking the Detection of Cloud-based User Authentication Log Anomalies Using Machine Learning Techniques
#Code Component: Configuration File
#Dataset: N/A
#Submitted to: Dr. Jamal Bentahar
#Done by: Aisha AlQahtani, & Salwa Mohammed Razaulla
#############################################################################################

#############################################################################################
#IMPORTING LIBRARIES
#############################################################################################

import pandas as pd #Loads and manipulates CSV datasets into DataFrames.
import numpy as np #Handles numerical operations and array math.
import nltk #Natural Language Toolkit; used for stopword removal during text cleaning.
import string #Provides the punctuation list used during text cleaning.
import os #Creates output and model directories if they don't exist.
import warnings #Used in line 28; to supress non-critical warnings during model training.
import joblib #Saves and loads trained models as .pkl files for reuse.

#This part suppresses non-critical warning messages that the scikit-learn
#prints during training (Like convergence warnings from Logistic Regression or deprecation notices).
#It also keeps the console output clean and readable.
warnings.filterwarnings('ignore')

#The line below imports a built-in list of common English words like "the", "and", "is", and "a".
#These words get removed during text pre-processing because they appear everywhere and add no useful signal for detecting brute-force attacks.
from nltk.corpus import stopwords

#Libraries for Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer #Converts text into numerical features using word frequency scores.
from sklearn.decomposition import TruncatedSVD #Reduces matrix dimensions that can't handle sparse data (Elliptic Envelope, One-Class SVM).

#Importing Unsupervised Models
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

#Importing Supervised Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#Importing Evaluation Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
    pairwise_distances_argmin_min, average_precision_score
)

#Importing Plotting Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

#Downloads the stopwords list (Common words like "the", "and") needed for text cleaning.
nltk.download('stopwords', quiet=True)

#############################################################################################
#FILE PATHS
#############################################################################################

MAIN_DATASET_PATH = "/Users/aisha/Desktop/COSC_702/Project/Datasets/Training/Dataset13k.csv"
LABELED_DATASET_PATH = "/Users/aisha/Desktop/COSC_702/Project/Datasets/Training/Labeled13k.csv"

#Testing scenarios for Scenarios_Test.py.
SCENARIOS = {
    'Small (1k)': "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/dataset1k.csv",
    'Medium (15k)': "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Dataset15k.csv",
    'Large (20k)': "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Dataset20k.csv",
    'Very Large (100k)': "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Dataset100k.csv",
    'Healthy Logs': "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/Healthy logs sample.csv",
    'Empty Dataset': "/Users/aisha/Desktop/COSC_702/Project/Datasets/Testing/empty_dataset.csv",
}

MODELS_DIR = "../models/"  #Where trained .pkl files are saved.
OUTPUTS_DIR = "../outputs/"  #Where charts and CSVs are saved.

#############################################################################################
#FEATURE COLUMNS
#############################################################################################

#The 4 columns combined into one text string per record before vectorization.
#Using Status alone only has 3 unique values (Success/Failure/Interrupted), which causes all models to produce identical results.
#Combining all 4 columns gives richer features for meaningful anomaly detection.
FEATURE_COLUMNS = ['Status', 'Location', 'IP address', 'Application']

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS

#Cybersecurity role of each feature column, used in ablation study output.
FEATURE_CONTEXT = {
    'Status': 'Primary brute force indicator, repeated failures signal attack.',
    'Location': 'Geographic anomaly, logins from unexpected regions.',
    'IP address': 'Network anomaly, known malicious IPs or unusual ranges.',
    'Application': 'Target surface, which apps are being attacked.',
}


#############################################################################################
#MODEL LISTS & COLORS
#############################################################################################

#Model order used consistently across all charts for visual alignment.
UNSUPERVISED_MODELS = [
    'Isolation Forest', 'K-Means Clustering', 'Elliptic Envelope',
    'Local Outlier Factor', 'One-Class SVM'
]
SUPERVISED_MODELS = [
    'Random Forest', 'SVM', 'Logistic Regression',
    'Gradient Boosting', 'KNN'
]

#Consistent color per model used across all charts.
#Blue shades for unsupervised, warm/red shades for supervised.
MODEL_COLORS = {
    'Isolation Forest': '#1f77b4',
    'K-Means Clustering': '#aec7e8',
    'Elliptic Envelope': '#6baed6',
    'Local Outlier Factor': '#08519c',
    'One-Class SVM': '#9ecae1',
    'Random Forest': '#d62728',
    'SVM': '#ff7f0e',
    'Logistic Regression': '#e6550d',
    'Gradient Boosting': '#fd8d3c',
    'KNN': '#fdae6b',
}

SCENARIO_COLORS = {
    'Small (1k)': '#4393c3',
    'Medium (15k)': '#2166ac',
    'Large (20k)': '#053061',
    'Very Large (100k)': '#313695',
    'Healthy Logs': '#1a9641',
    'Empty Dataset': '#bdbdbd',
}

#############################################################################################
#SHARED HELPER FUNCTIONS
#User for Main_AD_Code.py, Ablation_Study.py and Scenarios_Test.py.
#############################################################################################

def preprocess_text(text):
    """Lowercase, remove punctuation, strip stopwords."""
    stop_words = set(stopwords.words('english'))  #Common words like "the", "and", "is" that add no detection signal.
    if pd.isna(text):
        return ''  #Return empty string for null/missing values to avoid errors.
    if isinstance(text, str):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))  #Lowercase and strip punctuation.
        tokens = [w for w in text.split() if w not in stop_words]  #Split into words and remove stopwords.
        return ' '.join(tokens)  #Rejoin cleaned words into a single string.
    return ''  #Return empty string for non-string types (e.g. numbers).


def combine_columns(df, columns):
    """
    Combine multiple feature columns into one rich text string per row.
    e.g. "Failure Moscow 45.33.32.156 AzureAD" per record.
    """
    valid_cols = [c for c in columns if c in df.columns]  #Only use columns that exist in the dataframe.
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"    [WARNING] Columns not found and skipped: {missing}")
    #Fills nulls with empty string, converts all values to string, then joins each row's values with a space.
    return df[valid_cols].fillna('').astype(str).apply(
        lambda row: ' '.join(row.values), axis=1
    )


def vectorize_text(texts, max_features=500):
    """Fit a TF-IDF vectorizer and return matrix + fitted vectorizer."""
    vectorizer = TfidfVectorizer(max_features=max_features)  #Limits vocabulary to top 500 most informative words.
    cleaned = [preprocess_text(t) for t in texts]  #Apply text cleaning to every record.
    matrix = vectorizer.fit_transform(cleaned)  #Fit vocabulary and transform text into a numerical sparse matrix.
    print(f"[SUCCESS] TF-IDF matrix shape: {matrix.shape}")
    print(f"    Unique features: {len(vectorizer.get_feature_names_out())}\n")
    return matrix, vectorizer  #Both returned; vectorizer saved as .pkl so AD_Scenarios.py uses the same vocabulary.


def get_reduced(matrix, n_components=50):
    """SVD-reduce sparse TF-IDF matrix for dense-input models."""
    n_comp = min(n_components, matrix.shape[1] - 1)  #Cap at (features - 1) to prevent ValueError if matrix is small.
    if n_comp < 1:
        return matrix.toarray()  #If only 1 feature exists, just convert to dense directly.
    svd = TruncatedSVD(n_components=n_comp, random_state=42)  # random_state=42 ensures reproducible results
    return svd.fit_transform(matrix)  #Compress sparse matrix into 50 dense dimensions.


def load_data(file_path):
    """Load and clean a dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')  #utf-8-sig handles BOM characters some CSV exports include.
        df.columns = df.columns.str.strip().str.replace('ï»¿', '',
                                                        regex=True)  #Removes whitespace and encoding artifacts from column names.
        if df.empty:
            print("WARNING: DataFrame is empty after loading.")
        else:
            print(f"[✓] Dataset loaded — {len(df):,} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"[✗] Error loading data: {e}")
        return None


def load_labeled_data(labeled_data_path):
    """Load the manually labeled dataset and map labels to binary integers."""
    try:
        labeled_df = pd.read_csv(labeled_data_path, encoding='utf-8-sig')
        labeled_df.columns = labeled_df.columns.str.strip().str.replace('ï»¿', '', regex=True)

        if 'Anomaly_Label' not in labeled_df.columns:
            print("[✗] 'Anomaly_Label' column not found in labeled data.")
            return None

        #Unsupervised models use -1 (anomaly) and 1 (normal) — scikit-learn convention.
        labeled_df['label_unsup'] = labeled_df['Anomaly_Label'].apply(
            lambda x: -1 if str(x).strip().lower() == 'anomaly' else 1
        )
        #Supervised models use 1 (anomaly) and 0 (normal) — standard binary classification convention.
        labeled_df['label_sup'] = labeled_df['Anomaly_Label'].apply(
            lambda x: 1 if str(x).strip().lower() == 'anomaly' else 0
        )

        print(f"[SUCCESS] Labeled dataset loaded — {len(labeled_df):,} rows")
        print(f"    Label distribution:\n{labeled_df['Anomaly_Label'].value_counts().to_string()}\n")
        return labeled_df
    except Exception as e:
        print(f"[FAILURE] Error loading labeled data: {e}")
        return None


def compute_metrics_unsup(true_unsup, preds, scores):
    """Compute metrics for unsupervised model (-1/1 labels)."""
    #Convert -1/1 to 0/1 — unsupervised models output -1 (anomaly) and 1 (normal) but metric functions expect 1 (anomaly) and 0 (normal).
    true_bin = np.where(np.array(true_unsup) == -1, 1, 0)
    pred_bin = np.where(np.array(preds) == -1, 1, 0)
    try:
        auc = roc_auc_score(true_bin, scores)
    except Exception:
        auc = float('nan')
    return {
        'accuracy': accuracy_score(true_bin, pred_bin),
        'precision': precision_score(true_bin, pred_bin, zero_division=0),
        'recall': recall_score(true_bin, pred_bin, zero_division=0),
        'f1': f1_score(true_bin, pred_bin, zero_division=0),
        'auc_roc': auc
    }


def compute_metrics_sup(y_true, y_pred, y_proba=None):
    """Compute metrics for supervised model (0/1 labels)."""
    if y_proba is None:
        y_proba = y_pred
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float('nan')
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': auc
    }