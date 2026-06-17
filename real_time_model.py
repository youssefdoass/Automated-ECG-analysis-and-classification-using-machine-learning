import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
import scipy.signal as sp_signal
from scipy import stats as sp_stats
import wfdb

# ── Hard-coded configuration ───────────────────────────────────────
INFERENCE_DIR  = r"F:\youssef guc\bachelor\feature extraction testing\inference_artifacts"
RECORDS_PATH   = r"F:\youssef guc\bachelor\shaoxing\WFDB_ShaoxingUniv"
LABELS_PATH    = r"F:\youssef guc\bachelor\shaoxing\labels_all\Master_Labels_All.csv"
FS             = 500
SEG_BEFORE     = 90
SEG_AFTER      = 144

FEATURE_DIR       = r"F:\youssef guc\bachelor\feature extraction testing"
LEAD_INDICES      = [0, 2, 4]   # Lead I, Lead III, aVL — hardcoded, never changed
BEAT_INDEX_CACHE  = r"F:\youssef guc\bachelor\feature extraction testing\shaoxing_beat_index.npy"

# These are the ONLY leads ever used — hardcoded, never changed
LEAD_I_IDX     = 0
LEAD_III_IDX   = 2
LEAD_AVL_IDX   = 4
LEAD_NAMES     = ["Lead I", "Lead III", "aVL"]


def build_beat_index(labels_path, records_path,
                     seg_before, seg_after, fs,
                     save_path):
    """
    Walk every record in the Shaoxing dataset in labels CSV order,
    run the exact same R-peak detection used during training,
    and store the beat count for each record.

    Saves a (N_records, 2) array where:
        col 0 = cumulative beat offset at the START of this record
        col 1 = number of beats in this record

    This only runs once. On subsequent runs it loads from disk.
    """
    if os.path.exists(save_path):
        print(f"  Loading beat index from cache: {save_path}")
        return np.load(save_path)

    print("  Beat index not found. Building from scratch...")
    print("  This runs once and takes a few minutes.")

    labels_df  = pd.read_csv(labels_path)
    index_rows = []
    cumulative = 0

    for row_num, (_, row) in enumerate(labels_df.iterrows()):
        fname       = str(row['Patient_ID']).replace('.hea', '').strip()
        record_path = os.path.join(records_path, fname)
        beat_count  = 0

        try:
            record   = wfdb.rdrecord(record_path)
            full_sig = record.p_signal
            rec_fs   = record.fs

            lead_I   = full_sig[:, 0].astype(np.float32).copy()
            np.nan_to_num(lead_I, copy=False,
                          nan=0.0, posinf=0.0, neginf=0.0)

            # Identical bandpass filter to training script
            nyq = 0.5 * rec_fs
            try:
                b, a   = sp_signal.butter(
                    1, [0.5 / nyq, 40.0 / nyq], btype='band'
                )
                lead_I = sp_signal.filtfilt(b, a, lead_I)
            except Exception:
                pass

            # Identical R-peak detection to training script
            peaks, _ = sp_signal.find_peaks(
                lead_I ** 2,
                distance=int(rec_fs * 0.5),
                height=np.mean(lead_I ** 2) * 1.5
            )

            # Count only valid peaks (same boundary check as training)
            for peak in peaks:
                start = peak - seg_before
                end   = peak + seg_after
                if start >= 0 and end < len(lead_I):
                    beat_count += 1

        except Exception:
            pass

        index_rows.append([cumulative, beat_count])
        cumulative += beat_count

        if row_num % 500 == 0:
            print(f"    Processed {row_num:,} / "
                  f"{len(labels_df):,} records "
                  f"({cumulative:,} beats so far)...")

    beat_index = np.array(index_rows, dtype=np.int64)
    np.save(save_path, beat_index)
    print(f"  Beat index saved: {save_path}")
    print(f"  Total records: {len(beat_index):,}")
    print(f"  Total beats:   {beat_index[-1,0] + beat_index[-1,1]:,}")
    return beat_index


def get_record_beat_slice(beat_index, labels_path, target_fname):
    """
    Given the beat index array and a target record filename,
    return (beat_offset, beat_count) for that record.
    """
    labels_df = pd.read_csv(labels_path)
    for row_num, (_, row) in enumerate(labels_df.iterrows()):
        fname = str(row['Patient_ID']).replace('.hea', '').strip()
        if fname == target_fname:
            beat_offset = int(beat_index[row_num, 0])
            beat_count  = int(beat_index[row_num, 1])
            return beat_offset, beat_count
    raise ValueError(f"Record '{target_fname}' not found in labels CSV.")


def preprocess_signal(signal, fs):
    """Bandpass filter 0.5-40 Hz. Identical to training script."""
    np.nan_to_num(signal, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    nyq = 0.5 * fs
    try:
        b, a = sp_signal.butter(1, [0.5 / nyq, 40.0 / nyq], btype='band')
        return sp_signal.filtfilt(b, a, signal)
    except Exception:
        return signal


def segment_beats(leads, fs, seg_before, seg_after):
    """
    Detect R-peaks on the first lead and cut fixed-length windows.
    Returns list of (3, seg_len) arrays and list of peak indices.
    """
    lead_I   = leads[0]
    peaks, _ = sp_signal.find_peaks(
        lead_I ** 2,
        distance=int(fs * 0.5),
        height=np.mean(lead_I ** 2) * 1.5
    )
    beat_segments = []
    valid_peaks   = []
    for peak in peaks:
        start, end = peak - seg_before, peak + seg_after
        if start >= 0 and end < len(lead_I):
            seg = np.stack([lead[start:end] for lead in leads], axis=0)
            if not np.isnan(seg).any() and not np.isinf(seg).any():
                beat_segments.append(seg)
                valid_peaks.append(peak)
    return beat_segments, valid_peaks


def extract_morphological_features(segments, seg_before, seg_after):
    """15 morphological features per lead x 3 leads = 45. Identical to training."""
    N, n_leads, seg_len = segments.shape
    all_lead_feats      = []
    for lead_idx in range(n_leads):
        sig     = segments[:, lead_idx, :].astype(np.float32)
        rr      = np.full(N, float(seg_after), dtype=np.float32)
        for i in range(N):
            pks, _ = sp_signal.find_peaks(sig[i], distance=50)
            after  = pks[pks > seg_before]
            if len(after) > 0:
                rr[i] = float(after[0] - seg_before)
        r_sq    = sig[:, seg_before] ** 2
        qrs_win = sig[:, seg_before - 50: seg_before + 50] ** 2
        qrs_dur = np.sum(
            qrs_win > (0.5 * r_sq)[:, None], axis=1
        ).astype(np.float32)
        lead_feats = np.stack([
            rr,
            qrs_dur,
            sig[:, seg_before],
            np.mean(sig, axis=1),
            np.std(sig, axis=1),
            sp_stats.skew(sig, axis=1).astype(np.float32),
            sp_stats.kurtosis(sig, axis=1).astype(np.float32),
            np.sum(sig ** 2, axis=1) / seg_len,
            np.sum(np.diff(np.sign(sig), axis=1) != 0,
                   axis=1).astype(np.float32) / seg_len,
            np.mean(sig[:, 0:60], axis=1),
            np.mean(sig[:, 120:], axis=1),
            np.std(sig[:, 0:60], axis=1),
            np.std(sig[:, 120:], axis=1),
            np.mean(sig[:, 100:130], axis=1),
            np.max(sig[:, 70:110], axis=1) - np.min(sig[:, 70:110], axis=1),
        ], axis=1).astype(np.float32)
        all_lead_feats.append(lead_feats)
    features = np.concatenate(all_lead_feats, axis=1)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_frequency_features(segments, fs=500):
    """7 FFT features per lead x 3 leads = 21. Identical to training."""
    N, n_leads, seg_len = segments.shape
    freqs               = np.fft.rfftfreq(seg_len, d=1.0 / fs)
    all_lead_feats      = []
    for lead_idx in range(n_leads):
        sig           = segments[:, lead_idx, :].astype(np.float32)
        power         = np.abs(np.fft.rfft(sig, axis=1)) ** 2
        dom_freq      = freqs[np.argmax(power, axis=1)].astype(np.float32)
        power_norm    = power / (power.sum(axis=1, keepdims=True) + 1e-8)
        spec_entropy  = sp_stats.entropy(power_norm.T).astype(np.float32)
        bp_low        = np.sum(
            power[:, (freqs >= 0.5) & (freqs <= 5)], axis=1
        ).astype(np.float32)
        bp_mid        = np.sum(
            power[:, (freqs > 5) & (freqs <= 15)], axis=1
        ).astype(np.float32)
        bp_high       = np.sum(
            power[:, (freqs > 15) & (freqs <= 40)], axis=1
        ).astype(np.float32)
        spec_centroid = (
            np.sum(freqs * power, axis=1) /
            (np.sum(power, axis=1) + 1e-8)
        ).astype(np.float32)
        total_power   = np.sum(power, axis=1).astype(np.float32)
        lead_feats    = np.stack(
            [dom_freq, spec_entropy, bp_low, bp_mid,
             bp_high, spec_centroid, total_power], axis=1
        ).astype(np.float32)
        all_lead_feats.append(lead_feats)
    features = np.concatenate(all_lead_feats, axis=1)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def load_latent_features(feature_dir, original_indices,
                         beat_offset, beat_count):
    """
    Load pre-saved latent features for a specific record using
    the exact beat offset and count from the beat index.
    Returns (beat_count, 96) array.
    """
    all_latents = []
    for idx in original_indices:
        feat_path = os.path.join(
            feature_dir, f"saved_features_lead{idx}.npy"
        )
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Latent feature file not found: {feat_path}"
            )
        feats     = np.load(feat_path, mmap_mode='r')
        end_idx   = beat_offset + beat_count
        if end_idx > feats.shape[0]:
            raise IndexError(
                f"Beat slice [{beat_offset}:{end_idx}] exceeds "
                f"array length {feats.shape[0]} for lead {idx}."
            )
        chunk = feats[beat_offset:end_idx].copy()
        all_latents.append(chunk)
    return np.concatenate(all_latents, axis=1)


def readable_feature(feature_name, shap_val):
    """Convert a raw feature name and SHAP value into a plain-English phrase."""
    direction = "elevated" if shap_val > 0 else "reduced"
    concern   = "increasing concern" if shap_val > 0 else "reducing concern"

    templates = {
        "RR_interval":          f"an {direction} RR interval ({concern} for arrhythmia)",
        "QRS_duration":         f"an {direction} QRS duration ({concern} for conduction abnormality)",
        "R_peak_amplitude":     f"an {direction} R-peak amplitude ({concern})",
        "mean_amplitude":       f"an {direction} mean signal amplitude ({concern})",
        "amplitude_std":        f"{direction} amplitude variability ({concern})",
        "waveform_skewness":    f"{direction} waveform asymmetry ({concern})",
        "waveform_kurtosis":    f"{direction} waveform peakedness ({concern})",
        "signal_energy":        f"{direction} overall signal energy ({concern})",
        "zero_crossing_rate":   f"a {direction} zero-crossing rate ({concern})",
        "P_wave_mean":          f"an {direction} P-wave amplitude ({concern} for atrial activity)",
        "T_wave_mean":          f"an {direction} T-wave amplitude ({concern} for repolarisation)",
        "P_wave_std":           f"{direction} P-wave variability ({concern})",
        "T_wave_std":           f"{direction} T-wave variability ({concern})",
        "ST_segment_mean":      f"an {direction} ST-segment level ({concern} for ischaemia)",
        "QRS_peak_to_peak":     f"an {direction} QRS peak-to-peak range ({concern})",
        "dominant_frequency":   f"a {direction} dominant frequency ({concern})",
        "spectral_entropy":     f"{direction} spectral entropy ({concern} for rhythm regularity)",
        "low_band_power":       f"{direction} low-frequency band power ({concern})",
        "mid_band_power":       f"{direction} mid-frequency band power ({concern})",
        "high_band_power":      f"{direction} high-frequency band power ({concern})",
        "spectral_centroid":    f"a {direction} spectral centroid ({concern})",
        "total_spectral_power": f"{direction} total spectral power ({concern})",
    }

    for key, template in templates.items():
        if key in feature_name:
            parts = feature_name.split("_")
            lead  = parts[-1].replace("Lead", "Lead ")
            return f"{template} in {lead}"

    if "latent_pattern" in feature_name:
        parts = feature_name.split("_")
        lead  = parts[-1].replace("Lead", "Lead ")
        return f"an {direction} learned waveform pattern ({concern}) in {lead}"

    return None


def generate_explanation(label, confidence, shap_vals, feature_names):
    """
    Generate a plain-English paragraph explaining the classification.
    Driven entirely by SHAP values — no invented clinical claims.
    """
    abs_shap    = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap)[::-1][:5]

    descriptions = []
    for idx in top_indices:
        phrase = readable_feature(feature_names[idx], shap_vals[idx])
        if phrase:
            descriptions.append(phrase)

    opening = (
        f"This beat was classified as {label} with "
        f"{confidence:.1f}% confidence."
    )

    if len(descriptions) >= 3:
        body = (
            f"The three features that most strongly influenced this decision were "
            f"{descriptions[0]}, {descriptions[1]}, and {descriptions[2]}."
        )
    elif len(descriptions) == 2:
        body = (
            f"The two features that most strongly influenced this decision were "
            f"{descriptions[0]} and {descriptions[1]}."
        )
    elif len(descriptions) == 1:
        body = (
            f"The feature that most strongly influenced this decision was "
            f"{descriptions[0]}."
        )
    else:
        body = "The model based its decision on a combination of signal features."

    if label == "Normal":
        closing = (
            "No features of significant clinical concern were detected. "
            "This result reflects the model's assessment only and does not "
            "constitute a clinical diagnosis."
        )
    else:
        closing = (
            "These signal characteristics deviated from the patterns observed "
            "in normal beats during training. This result reflects the model's "
            "assessment only and does not constitute a clinical diagnosis."
        )

    return f"{opening} {body} {closing}"


def main():
    print("="*60)
    print("  REAL-TIME ECG CLASSIFIER — SHAOXING MODE")
    print("  Leads: Lead I, Lead III, aVL (hardcoded)")
    print("  Features: latent (96) + morphological (45) + "
          "frequency (21) = 162")
    print("="*60)

    # ── Step 1: Load model and config ─────────────────────────────
    print("\n[1/5] Loading trained model and configuration...")
    clf = joblib.load(os.path.join(INFERENCE_DIR, "clf_xgboost.joblib"))
    with open(os.path.join(INFERENCE_DIR, "inference_config.json")) as f:
        config = json.load(f)
    threshold     = config["optimal_threshold"]
    lead_indices  = config["lead_indices"]      # [0, 2, 4]
    with open(os.path.join(INFERENCE_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    print(f"  Model loaded. Threshold: {threshold:.4f}")

    # Build SHAP explainer
    explainer = shap.TreeExplainer(clf)
    print("  SHAP explainer ready.")

    # ── Step 2: Build or load beat index ──────────────────────────
    print("\n[2/5] Loading beat index for Shaoxing dataset...")
    beat_index = build_beat_index(
        LABELS_PATH, RECORDS_PATH,
        SEG_BEFORE, SEG_AFTER, FS,
        BEAT_INDEX_CACHE
    )
    print(f"  Beat index loaded: {len(beat_index):,} records indexed.")

    # ── Step 3: Load target record ────────────────────────────────
    print("\n[3/5] Loading first record from Shaoxing dataset...")
    labels_df   = pd.read_csv(LABELS_PATH)
    first_row   = labels_df.iloc[0]
    fname       = str(first_row['Patient_ID']).replace('.hea', '').strip()
    raw_label   = str(first_row['Diagnostic_Label']).strip().lower()
    true_label  = "Normal" if raw_label == "normal" else "Abnormal"
    record_path = os.path.join(RECORDS_PATH, fname)

    print(f"  Record:     {fname}")
    print(f"  True label: {true_label}")

    record     = wfdb.rdrecord(record_path)
    full_sig   = record.p_signal
    n_leads_in = full_sig.shape[1]

    if n_leads_in < 5:
        raise ValueError(
            f"Record {fname} has only {n_leads_in} leads. "
            f"Need at least 5."
        )

    # Extract ONLY the 3 required leads — all others discarded
    lead_I   = full_sig[:, LEAD_I_IDX].astype(np.float32).copy()
    lead_III = full_sig[:, LEAD_III_IDX].astype(np.float32).copy()
    lead_aVL = full_sig[:, LEAD_AVL_IDX].astype(np.float32).copy()
    assert len({len(lead_I), len(lead_III), len(lead_aVL)}) == 1
    print(f"  Signal length: {len(lead_I):,} samples "
          f"({len(lead_I)/FS:.1f} seconds)")
    print(f"  Confirmed: exactly 3 leads extracted.")

    # ── Step 4: Preprocess and segment ────────────────────────────
    print("\n[4/5] Preprocessing and segmenting beats...")
    leads = [
        preprocess_signal(lead_I,   FS),
        preprocess_signal(lead_III, FS),
        preprocess_signal(lead_aVL, FS),
    ]
    assert len(leads) == 3

    beat_segments, valid_peaks = segment_beats(
        leads, FS, SEG_BEFORE, SEG_AFTER
    )

    if len(beat_segments) == 0:
        print("  No valid beats detected.")
        return

    segments_arr = np.array(beat_segments, dtype=np.float32)
    assert segments_arr.shape[1] == 3
    n_beats_runtime = len(beat_segments)
    print(f"  Detected {n_beats_runtime} beats at runtime.")

    # Get the stored beat offset and count for this record
    beat_offset, beat_count_stored = get_record_beat_slice(
        beat_index, LABELS_PATH, fname
    )
    print(f"  Stored beat count for this record: {beat_count_stored}")
    print(f"  Beat offset in .npy arrays:        {beat_offset}")

    # Use stored beat count for latent indexing — this is the ground truth
    # If runtime detection differs slightly, trust the stored count
    if n_beats_runtime != beat_count_stored:
        print(f"  Warning: runtime detected {n_beats_runtime} beats "
              f"but stored count is {beat_count_stored}.")
        print(f"  Using minimum of the two to stay aligned.")
        n_beats_use = min(n_beats_runtime, beat_count_stored)
        segments_arr = segments_arr[:n_beats_use]
    else:
        n_beats_use = n_beats_runtime
    print(f"  Using {n_beats_use} beats for classification.")

    # ── Step 5: Extract features and classify ─────────────────────
    print("\n[5/5] Extracting features and classifying beats...")

    # Latent features from pre-saved .npy arrays (exact slice)
    X_latent = load_latent_features(
        FEATURE_DIR, lead_indices,
        beat_offset, n_beats_use
    )

    # Morphological and frequency features computed fresh
    X_morph  = extract_morphological_features(
        segments_arr, SEG_BEFORE, SEG_AFTER
    )
    X_freq   = extract_frequency_features(segments_arr, fs=FS)

    # Combine all features
    X = np.concatenate([X_latent, X_morph, X_freq], axis=1)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Feature matrix: {X.shape}  "
          f"(latent 96 + morphological 45 + frequency 21 = 162)")

    # Classify and explain each beat
    print("\n" + "="*60)
    print("  CLASSIFICATION RESULTS")
    print("="*60)

    normal_count   = 0
    abnormal_count = 0

    for i in range(n_beats_use):
        beat_features = X[i].reshape(1, -1)
        proba         = clf.predict_proba(beat_features)[0]
        prob_normal   = proba[0]
        prob_abnormal = proba[1]
        label         = ("Normal" if prob_normal >= threshold
                         else "Abnormal")
        confidence    = (prob_normal * 100 if label == "Normal"
                         else prob_abnormal * 100)

        # SHAP on this single beat
        shap_vals   = explainer.shap_values(beat_features)[0]
        explanation = generate_explanation(
            label, confidence, shap_vals, feature_names
        )

        if label == "Normal":
            normal_count += 1
        else:
            abnormal_count += 1

        print(f"\nBeat {i+1:>3}  |  {label:<9}  |  "
              f"Confidence: {confidence:>5.1f}%")
        print(f"  {explanation}")

    # Summary
    majority = ("Normal" if normal_count >= abnormal_count
                else "Abnormal")
    match    = "CORRECT" if majority == true_label else "INCORRECT"

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  Record:              {fname}")
    print(f"  True label:          {true_label}")
    print(f"  Beats classified:    {n_beats_use}")
    print(f"  Normal beats:        {normal_count}")
    print(f"  Abnormal beats:      {abnormal_count}")
    print(f"  Majority prediction: {majority}  → {match}")
    print(f"  Threshold used:      {threshold:.4f}")
    print("="*60)
    print("\nNote: This result reflects the model's assessment only "
          "and does not constitute a clinical diagnosis.")


if __name__ == "__main__":
    main()
