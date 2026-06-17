"""
ABLATION STUDY: L1 REGULARIZATION (XGBoost reg_alpha)
Baseline: 3-channel (Lead I, Lead III, aVL) -> 82.77% accuracy
Pipeline: precomputed latent features (96) + morphological features (45) -> XGBoost reg_alpha=0.1
"""

import os
import random
import numpy as np
import wfdb
import scipy.signal as sp_signal
from scipy import stats as sp_stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, f1_score, accuracy_score
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
import joblib
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ──
SEG_BEFORE   = 90   # samples before R-peak in each beat segment
SEG_AFTER    = 144  # samples after R-peak in each beat segment
RANDOM_STATE = 42

XGBOOST_PARAMS = {
    'colsample_bytree': 0.9,
    'learning_rate':    0.05,
    'max_depth':        8,
    'min_child_weight': 5,
    'n_estimators':     700,
    'scale_pos_weight': 0.16331820447475331,  # compensates class imbalance (Normal >> Abnormal)
    'subsample':        1.0,
    'gamma':            0.5,
    'reg_alpha':        0.1,   # L1 penalty on leaf weights — the ablation variable
    'objective':        'binary:logistic',
    'eval_metric':      'logloss',
    'random_state':     RANDOM_STATE,
    'tree_method':      'hist',
    'n_jobs':           -1,
}

RECORDS_PATH   = r"F:\youssef guc\bachelor\shaoxing\WFDB_ShaoxingUniv"
LABELS_PATH    = r"F:\youssef guc\bachelor\shaoxing\labels_all\Master_Labels_All.csv"
FEATURE_DIR    = r"F:\youssef guc\bachelor\feature extraction testing"
SEGMENTS_CACHE = os.path.join(FEATURE_DIR, "cached_3ch_segments.npy")   # raw beat segments cache
LABELS_CACHE   = os.path.join(FEATURE_DIR, "cached_3ch_labels.npy")     # corresponding labels cache
INFERENCE_DIR  = os.path.join(FEATURE_DIR, "inference_artifacts")


def load_precomputed_three_lead_features(feature_dir):
    """Load saved per-lead autoencoder features and concatenate into (N, 96)."""
    lead_indices = [0, 2, 4]   # Lead I, Lead III, aVL
    lead_names   = ["Lead I", "Lead III", "aVL"]
    all_features, all_labels = [], []

    # Load the .npy feature file for each of the 3 leads
    for i, idx in enumerate(lead_indices):
        feat_path = os.path.join(feature_dir, f"saved_features_lead{idx}.npy")
        lab_path  = os.path.join(feature_dir, f"saved_labels_lead{idx}.npy")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Missing: {feat_path}")
        features = np.load(feat_path)
        labels   = np.load(lab_path)
        print(f"  Lead {idx:>2} ({lead_names[i]:>8}): {features.shape[0]:>7,} beats")
        all_features.append(features)
        all_labels.append(labels)

    # Trim all leads to the same beat count so they can be concatenated
    min_beats    = min(f.shape[0] for f in all_features)
    all_features = [f[:min_beats] for f in all_features]
    all_labels   = [l[:min_beats] for l in all_labels]
    print(f"\n  Aligning to minimum beat count: {min_beats:,}")

    # Stack the 3 lead feature arrays side by side: (N,32) x3 -> (N, 96)
    X = np.concatenate(all_features, axis=1)
    y = all_labels[0].astype(int)
    print(f"  Final feature matrix: {X.shape}")
    return X, y


def extract_morphological_features(segments):
    """Extract 15 hand-crafted features per lead x 3 leads = 45 features per beat."""
    N, n_leads, seg_len = segments.shape
    all_lead_feats = []

    for lead_idx in range(n_leads):
        sig = segments[:, lead_idx, :].astype(np.float32)  # (N, 234) for this lead

        # RR interval: distance in samples to the next peak after the R-peak
        rr = np.full(N, float(SEG_AFTER), dtype=np.float32)
        for i in range(N):
            pks, _ = sp_signal.find_peaks(sig[i], distance=50)
            after  = pks[pks > SEG_BEFORE]
            if len(after) > 0:
                rr[i] = float(after[0] - SEG_BEFORE)

        # QRS duration: count samples near R-peak where energy exceeds 50% of peak energy
        r_sq    = sig[:, SEG_BEFORE] ** 2
        qrs_win = sig[:, SEG_BEFORE - 50 : SEG_BEFORE + 50] ** 2
        qrs_dur = np.sum(qrs_win > (0.5 * r_sq)[:, None], axis=1).astype(np.float32)

        # Stack all 15 features into a (N, 15) array for this lead
        lead_feats = np.stack([
            rr,                                                                          # 1.  RR interval
            qrs_dur,                                                                     # 2.  QRS duration
            sig[:, SEG_BEFORE],                                                          # 3.  R-peak amplitude
            np.mean(sig, axis=1),                                                        # 4.  Mean amplitude
            np.std(sig, axis=1),                                                         # 5.  Std amplitude
            sp_stats.skew(sig, axis=1).astype(np.float32),                              # 6.  Skewness
            sp_stats.kurtosis(sig, axis=1).astype(np.float32),                          # 7.  Kurtosis
            np.sum(sig ** 2, axis=1) / seg_len,                                          # 8.  Signal energy
            np.sum(np.diff(np.sign(sig), axis=1) != 0, axis=1).astype(np.float32) / seg_len,  # 9.  Zero crossing rate
            np.mean(sig[:, 0:60], axis=1),                                               # 10. P-wave mean
            np.mean(sig[:, 120:], axis=1),                                               # 11. T-wave mean
            np.std(sig[:, 0:60], axis=1),                                                # 12. P-wave std
            np.std(sig[:, 120:], axis=1),                                                # 13. T-wave std
            np.mean(sig[:, 100:130], axis=1),                                            # 14. ST segment mean
            np.max(sig[:, 70:110], axis=1) - np.min(sig[:, 70:110], axis=1),            # 15. QRS peak-to-peak
        ], axis=1).astype(np.float32)

        all_lead_feats.append(lead_feats)

    # Concatenate features from all 3 leads: (N, 15) x3 -> (N, 45)
    features = np.concatenate(all_lead_feats, axis=1)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  # replace any bad values
    return features


def extract_frequency_features(segments, fs=500):
    """Extract 7 FFT-based features per lead x 3 leads = 21 frequency features per beat."""
    N, n_leads, seg_len = segments.shape
    freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)  # frequency axis in Hz, shape (seg_len//2+1,)
    all_lead_feats = []

    for lead_idx in range(n_leads):
        sig = segments[:, lead_idx, :].astype(np.float32)  # (N, 234)

        # Compute power spectrum for all beats at once: |FFT|^2, shape (N, freq_bins)
        power = np.abs(np.fft.rfft(sig, axis=1)) ** 2

        # 1. Dominant frequency: frequency bin with highest power
        dom_freq = freqs[np.argmax(power, axis=1)].astype(np.float32)

        # 2. Spectral entropy: how spread out the power is across frequencies
        power_norm  = power / (power.sum(axis=1, keepdims=True) + 1e-8)
        spec_entropy = sp_stats.entropy(power_norm.T).astype(np.float32)  # entropy per beat

        # 3-5. Band powers: energy concentrated in low, mid, and high frequency bands
        bp_low  = np.sum(power[:, (freqs >= 0.5) & (freqs <= 5)],   axis=1).astype(np.float32)
        bp_mid  = np.sum(power[:, (freqs >  5)   & (freqs <= 15)],  axis=1).astype(np.float32)
        bp_high = np.sum(power[:, (freqs >  15)  & (freqs <= 40)],  axis=1).astype(np.float32)

        # 6. Spectral centroid: frequency-weighted average — indicates dominant spectral location
        spec_centroid = (np.sum(freqs * power, axis=1) / (np.sum(power, axis=1) + 1e-8)).astype(np.float32)

        # 7. Total spectral power: overall signal energy in the frequency domain
        total_power = np.sum(power, axis=1).astype(np.float32)

        lead_feats = np.stack(
            [dom_freq, spec_entropy, bp_low, bp_mid, bp_high, spec_centroid, total_power],
            axis=1
        ).astype(np.float32)

        all_lead_feats.append(lead_feats)

    # Concatenate features from all 3 leads: (N, 7) x3 -> (N, 21)
    features = np.concatenate(all_lead_feats, axis=1)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def preprocess_signal(segment, fs):
    """Bandpass filter 0.5–40 Hz to remove baseline wander and high-frequency noise."""
    nyq = 0.5 * fs
    try:
        b, a = sp_signal.butter(1, [0.5 / nyq, 40.0 / nyq], btype='band')
        return sp_signal.filtfilt(b, a, segment)
    except Exception:
        return segment  # return unfiltered if filtering fails


def extract_three_lead_segments(record_path, label):
    """Read one WFDB record, detect R-peaks, and return fixed-length beat segments."""
    segments, labels = [], []
    try:
        record   = wfdb.rdrecord(record_path)
        full_sig = record.p_signal
        fs       = record.fs

        # Extract the 3 required leads by column index
        lead_I, lead_III, lead_aVL = (
            full_sig[:, 0].copy(), full_sig[:, 2].copy(), full_sig[:, 4].copy()
        )
        # Replace NaN/inf with 0 before filtering
        for sig in [lead_I, lead_III, lead_aVL]:
            np.nan_to_num(sig, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        lead_I   = preprocess_signal(lead_I, fs)
        lead_III = preprocess_signal(lead_III, fs)
        lead_aVL = preprocess_signal(lead_aVL, fs)

        # Detect R-peaks on Lead I using squared signal to amplify peaks
        peaks, _ = sp_signal.find_peaks(
            lead_I ** 2, distance=int(fs * 0.5), height=np.mean(lead_I ** 2) * 1.5
        )
        # Cut a fixed window around each R-peak and stack the 3 leads
        for peak in peaks:
            start, end = peak - SEG_BEFORE, peak + SEG_AFTER
            if start >= 0 and end < len(lead_I):
                seg = np.stack(
                    [lead_I[start:end], lead_III[start:end], lead_aVL[start:end]], axis=0
                )
                if not np.isnan(seg).any() and not np.isinf(seg).any():
                    segments.append(seg)
                    labels.append(label)
    except Exception:
        pass  # skip corrupted or missing records silently
    return segments, labels


def find_optimal_threshold(model, X_val, y_val):
    """Find the probability threshold that maximises Normal-class F1 on the validation set."""
    prob_normal     = model.predict_proba(X_val)[:, 0]
    y_normal_binary = (y_val == 0).astype(int)

    # Build the full precision-recall curve across all candidate thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_normal_binary, prob_normal)
    p, r  = precisions[:-1], recalls[:-1]
    denom = p + r
    f1_scores = np.where(denom > 0, 2.0 * p * r / denom, 0.0)

    # Pick the threshold with the highest F1
    best_idx = np.argmax(f1_scores)
    best_t   = thresholds[best_idx]

    print(f"  Optimal threshold:  {best_t:.6f}")
    print(f"  Normal Precision:   {p[best_idx]:.4f}")
    print(f"  Normal Recall:      {r[best_idx]:.4f}")
    print(f"  Normal F1-score:    {f1_scores[best_idx]:.4f}")
    return best_t, f1_scores[best_idx]


def apply_threshold(model, X, threshold):
    """Predict Normal (0) if P(Normal) >= threshold, else Abnormal (1)."""
    return np.where(model.predict_proba(X)[:, 0] >= threshold, 0, 1)


def generate_report(y_true, y_pred, label="Test"):
    """Print classification report, confusion matrix, and Normal-class F1."""
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE REPORT - {label}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred,
          target_names=["Normal", "Abnormal"], digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix:")
    print(f"                    Pred Normal    Pred Abnormal")
    print(f"  Actual Normal     {cm[0,0]:>10,}       {cm[0,1]:>10,}")
    print(f"  Actual Abnormal   {cm[1,0]:>10,}       {cm[1,1]:>10,}")

    # Manually compute Normal-class precision, recall, F1 from the confusion matrix
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    prec = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec  = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    print(f"\n  Normal Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"{'='*60}\n")
    return cm


def optuna_objective(trial, X_train, y_train, X_val, y_val):
    # Use 30% subsample of training data per trial so search finishes in reasonable time
    rng = np.random.RandomState(trial.number)
    idx = rng.choice(len(y_train), size=int(len(y_train) * 0.3), replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]

    params = {
        'max_depth':        trial.suggest_int('max_depth', 4, 12),
        'n_estimators':     trial.suggest_int('n_estimators', 200, 600),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma':            trial.suggest_float('gamma', 0.0, 2.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda':       trial.suggest_float('reg_lambda', 0.0, 2.0),
        'scale_pos_weight': float(np.sum(y_sub==0)) / float(np.sum(y_sub==1)),
        'objective':        'binary:logistic',
        'eval_metric':      'logloss',
        'random_state':     RANDOM_STATE,
        'tree_method':      'hist',
        'n_jobs':           -1,
    }
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_sub, y_sub)
    y_val_pred = clf.predict(X_val)
    return accuracy_score(y_val, y_val_pred)


def main():
    # Fix seeds so every run produces identical splits and XGBoost results
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("=" * 60)
    print("  ABLATION STUDY: L1 REGULARIZATION (XGBoost reg_alpha)")
    print("  3-Channel: Lead I + Lead III + aVL")
    print(f"  XGB reg_alpha: {XGBOOST_PARAMS['reg_alpha']}")
    print("  Baseline:  3-Channel (82.77% accuracy)")
    print("=" * 60)

    # Step 1: Load 32-dim latent features saved from per-lead autoencoders -> (N, 96)
    print("\n[1/4] Loading precomputed per-lead features...")
    X, y = load_precomputed_three_lead_features(FEATURE_DIR)

    # Step 1b: Load raw beat segments needed to compute morphological features
    print("\n[1b/4] Loading raw 3-channel segments for morphological features...")
    if os.path.exists(SEGMENTS_CACHE) and os.path.exists(LABELS_CACHE):
        # Fast path: segments were already extracted and saved on a previous run
        segments = np.load(SEGMENTS_CACHE)
        print(f"  Loaded {segments.shape[0]:,} cached segments.")
    else:
        # Slow path (one-time): read every WFDB record and extract beat segments
        print("  Cache not found — extracting from WFDB records (one-time cost)...")
        _labels_df = pd.read_csv(LABELS_PATH)
        _records   = []
        for _, row in _labels_df.iterrows():
            _fname = str(row['Patient_ID']).replace('.hea', '')
            _rpath = os.path.join(RECORDS_PATH, _fname)
            _lbl   = 0 if str(row['Diagnostic_Label']).strip().lower() == 'normal' else 1
            _records.append((_rpath, _lbl))
        _results    = Parallel(n_jobs=-1)(
            delayed(extract_three_lead_segments)(p, l)
            for p, l in tqdm(_records, desc="Extracting")
        )
        _seg_list   = [s for r in _results for s in r[0]]
        _seg_labels = np.array([l for r in _results for l in r[1]], dtype=int)
        segments    = np.array(_seg_list, dtype=np.float32)
        np.save(SEGMENTS_CACHE, segments)   # cache for future runs
        np.save(LABELS_CACHE,   _seg_labels)
        print(f"  Extracted and cached {segments.shape[0]:,} segments.")

    # Align beat counts: latent features and raw segments may differ slightly
    X_latent = X
    _min_n   = min(X_latent.shape[0], segments.shape[0])
    X_latent = X_latent[:_min_n]
    y        = y[:_min_n]
    segments = segments[:_min_n]
    print(f"  Aligned to {_min_n:,} beats.")

    # Step 1c: Compute 45 morphological features
    print("\n[1c/4] Extracting morphological features (15 per lead × 3 leads = 45)...")
    X_morph = extract_morphological_features(segments)

    # Step 1d: Compute 21 frequency-domain features and concatenate all -> (N, 162)
    print("\n[1d/4] Extracting frequency features (7 per lead × 3 leads = 21)...")
    X_freq  = extract_frequency_features(segments)
    X       = np.concatenate([X_latent, X_morph, X_freq], axis=1)
    print(f"  Latent: {X_latent.shape[1]}  +  Morphological: {X_morph.shape[1]}  "
          f"+  Frequency: {X_freq.shape[1]}  =  Combined: {X.shape[1]} features")

    # Step 2: Stratified 60/20/20 split into train, validation, and test sets
    print("\n[2/4] Splitting data (60/20/20 stratified)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y_train_full
    )
    print(f"  Train: {X_train.shape[0]:>8,}   Val: {X_val.shape[0]:>8,}   Test: {X_test.shape[0]:>8,}")
    print(f"  Train Normal: {np.sum(y_train==0):,}  Abnormal: {np.sum(y_train==1):,}")
    print(f"  Test  Normal: {np.sum(y_test==0):,}  Abnormal: {np.sum(y_test==1):,}")

    # Step 2b: SMOTE — oversample Normal class in the training set only
    print("\n[SMOTE] Oversampling Normal class in training set only...")
    print(f"  Before — Normal: {np.sum(y_train==0):,}  Abnormal: {np.sum(y_train==1):,}  Total: {len(y_train):,}")
    smote = SMOTE(
        sampling_strategy=0.8,
        random_state=RANDOM_STATE,
        k_neighbors=5
    )
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"  After  — Normal: {np.sum(y_train==0):,}  Abnormal: {np.sum(y_train==1):,}  Total: {len(y_train):,}")

    # Step 3: Optuna hyperparameter search + final XGBoost training
    print("\n[3/4] Running Optuna hyperparameter search (50 trials)...")
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=50,
        show_progress_bar=True
    )
    print(f"\n  Best validation accuracy: {study.best_value*100:.2f}%")
    print(f"  Best params found: {study.best_params}")

    best_params = study.best_params.copy()
    best_params['scale_pos_weight'] = float(np.sum(y_train==0)) / float(np.sum(y_train==1))
    best_params['objective']        = 'binary:logistic'
    best_params['eval_metric']      = 'logloss'
    best_params['random_state']     = RANDOM_STATE
    best_params['tree_method']      = 'hist'
    best_params['n_jobs']           = -1

    print("\n  Retraining with best params on full training set...")
    clf = xgb.XGBClassifier(**best_params)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    print(f"  Training Accuracy: {accuracy_score(y_train, y_train_pred)*100:.2f}%")

    # Step 4: Evaluate with default threshold (t=0.5)
    print("\n[4/4] Evaluating with default threshold (t=0.5)...")
    y_test_default = clf.predict(X_test)
    generate_report(y_test, y_test_default, "Test - Default (t=0.5)")

    # Summary comparison against the baseline
    acc_default = accuracy_score(y_test, y_test_default) * 100
    f1_default  = f1_score(y_test, y_test_default, pos_label=0, zero_division=0)

    print("=" * 60)
    print("  ABLATION COMPARISON vs BASELINE")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Baseline (3ch)':<15} {'+ L1 reg_alpha':<16}")
    print(f"  {'-'*25} {'-'*15} {'-'*16}")
    print(f"  {'Accuracy (default)':<25} {'75.68%':<15} {acc_default:<15.2f}%")
    print(f"  {'Normal F1 (default)':<25} {'0.6207':<15} {f1_default:<15.4f}")
    print(f"  {'XGB reg_alpha (L1)':<25} {'0 (none)':<15} {XGBOOST_PARAMS['reg_alpha']:<15}")
    print(f"  {'Frequency features':<25} {'No':<15} {'Yes (+21)':<15}")
    print(f"  {'Feature dimensions':<25} {'96 (latent)':<15} {'162 (96+45+21)':<15}")
    print(f"  {'SMOTE strategy':<25} {'No':<15} {'0.8':<15}")
    print(f"  {'Optuna trials':<25} {'No':<15} {'50 trials':<15}")
    print(f"  {'Best val accuracy':<25} {'N/A':<15} {study.best_value*100:<15.2f}%")
    print(f"  {'Train size after SMOTE':<25} {'307,097':<15} {len(y_train):,}")
    delta = acc_default - 75.68
    sign  = '+' if delta >= 0 else ''
    print(f"\n  Accuracy delta vs baseline:  {sign}{delta:.2f}%")
    print("=" * 60)

    # ── STEP 5: Save all inference artifacts ──────────────────────────
    print("\n[5/5] Saving inference artifacts...")
    os.makedirs(INFERENCE_DIR, exist_ok=True)

    # 5a. Save trained XGBoost classifier
    joblib.dump(clf, os.path.join(INFERENCE_DIR, "clf_xgboost.joblib"))
    print("  Saved: clf_xgboost.joblib")

    # 5b. Find and save optimal classification threshold from validation set
    best_threshold, best_f1 = find_optimal_threshold(clf, X_val, y_val)
    inference_config = {
        "optimal_threshold":  float(best_threshold),
        "seg_before":         SEG_BEFORE,
        "seg_after":          SEG_AFTER,
        "random_state":       RANDOM_STATE,
        "feature_dim":        int(X_train.shape[1]),
        "n_latent_per_lead":  32,
        "lead_indices":       [0, 2, 4],
        "lead_names":         ["Lead I", "Lead III", "aVL"],
        "fs":                 500,
    }
    with open(os.path.join(INFERENCE_DIR, "inference_config.json"), "w") as f:
        json.dump(inference_config, f, indent=2)
    print("  Saved: inference_config.json")

    # 5c. Save feature statistics from training set for input validation
    feature_stats = {
        "mean": X_train.mean(axis=0).tolist(),
        "std":  X_train.std(axis=0).tolist(),
    }
    with open(os.path.join(INFERENCE_DIR, "feature_stats.json"), "w") as f:
        json.dump(feature_stats, f)
    print("  Saved: feature_stats.json")

    # 5d. Build and save the plain-English feature name mapping for all 162 features
    lead_names_short = ["Lead_I", "Lead_III", "aVL"]
    morph_names = [
        "RR_interval", "QRS_duration", "R_peak_amplitude",
        "mean_amplitude", "amplitude_std", "waveform_skewness",
        "waveform_kurtosis", "signal_energy", "zero_crossing_rate",
        "P_wave_mean", "T_wave_mean", "P_wave_std",
        "T_wave_std", "ST_segment_mean", "QRS_peak_to_peak"
    ]
    freq_names = [
        "dominant_frequency", "spectral_entropy",
        "low_band_power", "mid_band_power", "high_band_power",
        "spectral_centroid", "total_spectral_power"
    ]
    feature_names = []
    for i in range(32 * 3):
        lead  = lead_names_short[i // 32]
        idx   = (i % 32) + 1
        feature_names.append(f"latent_pattern_{idx}_{lead}")
    for lead in lead_names_short:
        for name in morph_names:
            feature_names.append(f"{name}_{lead}")
    for lead in lead_names_short:
        for name in freq_names:
            feature_names.append(f"{name}_{lead}")

    with open(os.path.join(INFERENCE_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
    print("  Saved: feature_names.json")

    # ── STEP 6: SHAP explainability ───────────────────────────────────
    print("\n[6/6] Computing SHAP values on test set (stratified 1,000-sample subset)...")
    rng_shap  = np.random.RandomState(RANDOM_STATE)
    shap_idx  = np.concatenate([
        rng_shap.choice(np.where(y_test == 0)[0], size=500, replace=False),
        rng_shap.choice(np.where(y_test == 1)[0], size=500, replace=False),
    ])
    X_shap    = X_test[shap_idx]
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_shap)

    # Save the explainer for reuse in the real-time model
    joblib.dump(explainer, os.path.join(INFERENCE_DIR, "shap_explainer.joblib"))
    print("  Saved: shap_explainer.joblib")

    # 6a. Global SHAP summary bar plot — top 20 features by mean absolute value
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False
    )
    plt.title("Global Feature Importance — Mean |SHAP value|", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(INFERENCE_DIR, "shap_global_bar.png"), dpi=150)
    plt.close()
    print("  Saved: shap_global_bar.png")

    # 6b. Global SHAP beeswarm plot — direction and distribution of feature effects
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feature_names,
        max_display=20,
        show=False
    )
    plt.title("Global Feature Impact — Direction and Magnitude", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(INFERENCE_DIR, "shap_global_beeswarm.png"), dpi=150)
    plt.close()
    print("  Saved: shap_global_beeswarm.png")

    # 6c. Save SHAP values and base value for use by real-time model
    np.save(os.path.join(INFERENCE_DIR, "shap_values_test.npy"), shap_values)
    with open(os.path.join(INFERENCE_DIR, "shap_base_value.json"), "w") as f:
        json.dump({"base_value": float(explainer.expected_value)}, f)
    print("  Saved: shap_values_test.npy")
    print("  Saved: shap_base_value.json")

    print("\n" + "="*60)
    print("  ALL ARTIFACTS SAVED TO:")
    print(f"  {INFERENCE_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
