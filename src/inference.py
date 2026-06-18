import numpy as np
import torch

from src.severity import compute_severity


def run_inference(preprocessor, model, threshold_config, classifier_model,
                  classifier_config, raw_window: np.ndarray) -> dict:
    """
    Pipeline complet : signal brut -> anomalie + classification + sévérité.

    Paramètres :
        raw_window : np.ndarray de longueur window_size

    Retourne un dict avec :
        mse, threshold, is_anomaly, severity_score,
        predicted_fault (label lisible),
        spectrogram, recon_spectrogram, signal_reconstructed,
        features (rms, kurtosis, variance, fft_freqs, fft_amplitudes, fft_peak_hz)
    """
    LABEL_TO_IDX = {
        "Normal": 0,
        "Ball_007": 1, "Ball_014": 2, "Ball_021": 3,
        "IR_007": 4,   "IR_014": 5,   "IR_021": 6,
        "OR_007": 7,   "OR_014": 8,   "OR_021": 9,
    }

    # 1. STFT -> spectrogramme 128x32
    magnitude_128, original_stft = preprocessor.compute_stft(raw_window)
    features = preprocessor.extract_features(raw_window)

    # 2. Normalisation
    scaled_magnitude = preprocessor.transform_magnitude(magnitude_128)

    # 3. Inférence autoencoder (détection d'anomalie)
    tensor_x = torch.tensor(scaled_magnitude, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        tensor_y = model(tensor_x)

    mse = torch.mean((tensor_x - tensor_y) ** 2).item()
    threshold = threshold_config["threshold"]
    is_anomaly = mse > threshold

    # 4. Score de sévérité 0-100
    severity_score = compute_severity(mse, threshold)

    # 5. Reconstruction signal 1D
    recon_scaled = tensor_y.squeeze(0).squeeze(0).cpu().numpy()
    recon_magnitude_128 = preprocessor.inverse_transform_magnitude(recon_scaled)
    reconstructed_1d = preprocessor.reconstruct_signal_from_stft(recon_magnitude_128, original_stft)

    # 6. Classification (uniquement si anomalie détectée)
    predicted_fault = "Normal"
    if is_anomaly and classifier_model is not None:
        # Utilise les 32 premières lignes (basses fréquences) pour le classifieur
        # DenseNet redimensionne en interne via F.interpolate
        tensor_x_32 = tensor_x[:, :, :32, :]  # shape (1, 1, 32, 32)
        with torch.no_grad():
            class_outputs = classifier_model(tensor_x_32)
            predicted_class_idx = torch.argmax(class_outputs, dim=1).item()

        labels_map = (
            classifier_config.get("labels_map", LABEL_TO_IDX)
            if classifier_config else LABEL_TO_IDX
        )
        idx_to_label = {v: k for k, v in labels_map.items()}
        raw_label = idx_to_label.get(predicted_class_idx, "Unknown")
        predicted_fault = _format_label(raw_label)

    return {
        "mse": mse,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "severity_score": severity_score,
        "predicted_fault": predicted_fault,
        "spectrogram": magnitude_128,
        "recon_spectrogram": recon_magnitude_128,
        "signal_reconstructed": reconstructed_1d,
        "features": features,
    }


def _format_label(label: str) -> str:
    if label == "Normal":
        return "Normal"
    parts = label.split("_")
    if len(parts) != 2:
        return label
    fault_type, severity = parts
    severity_map = {"007": '0.007"', "014": '0.014"', "021": '0.021"'}
    type_map = {
        "Ball": "Defaut de Bille",
        "IR":   "Defaut Bague Interieure",
        "OR":   "Defaut Bague Exterieure",
    }
    return f"{type_map.get(fault_type, fault_type)} ({severity_map.get(severity, severity)})"
