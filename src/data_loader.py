import os
import pickle
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler

class VibrationPreprocessor:
    def __init__(self, window_size=2048, step_size=512, fs=48000):  # FIXED P1 P2
        self.window_size = window_size
        self.step_size = step_size
        self.fs = fs
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def load_de_signal(self, filepath):
        """
        Loads the Drive End (DE) vibration signal from a MATLAB (.mat) file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        mat = sio.loadmat(filepath)
        de_keys = [k for k in mat.keys() if "DE_time" in k]
        
        if not de_keys:
            raise ValueError(f"No Drive End (DE) data found in {filepath}. Available keys: {list(mat.keys())}")
        
        # Robust key matching
        de_key = de_keys[0]
        filename = os.path.basename(filepath)
        for key in de_keys:
            for token in filename.split("_"):
                if token.isdigit() and token in key:
                    de_key = key
                    break
        
        signal_data = mat[de_key].flatten()
        return signal_data

    def segment_signal(self, signal_data):
        """
        Segments the 1D signal into sliding windows of length window_size.
        """
        windows = []
        for i in range(0, len(signal_data) - self.window_size + 1, self.step_size):
            windows.append(signal_data[i : i + self.window_size])
        return np.array(windows)

    def compute_stft(self, window):
        """
        Computes the STFT of a window.
        Returns:
            magnitude: np.ndarray of shape (128, 32) (sliced Nyquist bin)
            Zxx: np.ndarray of shape (129, 32) (complex STFT)
        """
        # Hann window, size 256, overlap 199 (hop_size 57) keeps 2048 samples at 32 STFT frames.  # FIXED P2
        f, t, Zxx = signal.stft(
            window, 
            fs=self.fs, 
            window='hann', 
            nperseg=256, 
            noverlap=199,  # FIXED P2
            boundary=None, 
            padded=False
        )
        magnitude = np.abs(Zxx)
        if magnitude.shape != (129, 32):  # FIXED P2
            raise ValueError(f"Expected STFT shape (129, 32), got {magnitude.shape}")  # FIXED P2
        # Squeeze Nyquist bin out to make it exactly 128x32
        magnitude_truncated = magnitude[:128, :]
        return magnitude_truncated, Zxx

    def reconstruct_signal_from_stft(self, magnitude_128, original_stft):
        """
        Reconstructs the 1D physical scale signal from the 128-bin magnitude and the original complex STFT.
        """
        magnitude_129 = np.zeros((129, 32))
        magnitude_129[:128, :] = magnitude_128
        # Keep the original 129th bin magnitude
        magnitude_129[128, :] = np.abs(original_stft[128, :])
        
        # Phase extraction
        phase = np.angle(original_stft)
        
        # Combine magnitude and phase
        recon_Zxx = magnitude_129 * np.exp(1j * phase)
        
        # Inverse STFT
        _, recon_signal = signal.istft(
            recon_Zxx, 
            fs=self.fs, 
            window='hann', 
            nperseg=256, 
            noverlap=199,  # FIXED P2
            boundary=None
        )
        if len(recon_signal) < self.window_size:  # FIXED P2
            recon_signal = np.pad(recon_signal, (0, self.window_size - len(recon_signal)))  # FIXED P2
        else:  # FIXED P2
            recon_signal = recon_signal[:self.window_size]  # FIXED P2
        return recon_signal

    def fit_scaler(self, normal_signal):
        """
        Fits the MinMaxScaler on the normal signal (segmented, converted to STFT magnitude, and flattened).
        """
        windows = self.segment_signal(normal_signal)
        magnitudes = []
        for w in windows:
            mag, _ = self.compute_stft(w)
            magnitudes.append(mag.flatten()) # shape: (4096,)
            
        self.scaler.fit(np.array(magnitudes))
        self.is_fitted = True

    def transform_magnitude(self, magnitude_128):
        """
        Normalizes a 128x32 magnitude spectrogram using the pre-fitted scaler.
        """
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted. Run fit_scaler first.")
            
        shape = magnitude_128.shape
        # Flatten to 1D, scale, and reconstruct to 2D
        if len(shape) == 2: # Single sample (128, 32)
            flattened = magnitude_128.flatten().reshape(1, -1)
            scaled = self.scaler.transform(flattened)
            return scaled[0].reshape(shape)
        else: # Batch of samples (N, 128, 32)
            N = shape[0]
            flattened = magnitude_128.reshape(N, -1)
            scaled = self.scaler.transform(flattened)
            return scaled.reshape(shape)

    def inverse_transform_magnitude(self, scaled_magnitude_128):
        """
        Converts the normalized magnitude spectrogram back to its physical scale.
        """
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted.")
            
        shape = scaled_magnitude_128.shape
        if len(shape) == 2:
            flattened = scaled_magnitude_128.flatten().reshape(1, -1)
            unscaled = self.scaler.inverse_transform(flattened)
            return unscaled[0].reshape(shape)
        else:
            N = shape[0]
            flattened = scaled_magnitude_128.reshape(N, -1)
            unscaled = self.scaler.inverse_transform(flattened)
            return unscaled.reshape(shape)

    def save_scaler(self, filepath):
        """
        Saves the preprocessor and fitted scaler.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "is_fitted": self.is_fitted,
                "window_size": self.window_size,
                "step_size": self.step_size,
                "fs": self.fs
            }, f)

    def load_scaler(self, filepath):
        """
        Loads the preprocessor configuration and scaler.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.scaler = data["scaler"]
            self.is_fitted = data["is_fitted"]
            self.window_size = data.get("window_size", 2048)  # FIXED P2
            self.step_size = data.get("step_size", 512)
            self.fs = data.get("fs", 48000)  # FIXED P1

    def extract_features(self, window):
        """
        Extracts temporal and frequency features from a raw 1D window.
        """
        # Temporal
        rms = np.sqrt(np.mean(window ** 2))
        kurt = kurtosis(window, fisher=False)
        var = np.var(window)

        # Additional features for baseline RF classifier
        mx = float(np.max(window))
        mn = float(np.min(window))
        mean = float(np.mean(window))
        sd = float(np.std(window))
        skewness_val = float(skew(window))
        abs_mean = np.mean(np.abs(window))
        crest = float(np.max(np.abs(window)) / (rms + 1e-8))
        form = float(rms / (abs_mean + 1e-8))

        # Overall window FFT for plotting
        fft_vals = np.fft.rfft(window)
        fft_amps = np.abs(fft_vals)
        fft_freqs = np.fft.rfftfreq(len(window), d=1.0/self.fs)

        peak_idx = np.argmax(fft_amps)
        peak_freq = fft_freqs[peak_idx]
        spectral_energy = np.sum(fft_amps ** 2)

        return {
            "rms": float(rms),
            "kurtosis": float(kurt),
            "variance": float(var),
            "max": mx,
            "min": mn,
            "mean": mean,
            "sd": sd,
            "skewness": skewness_val,
            "crest": crest,
            "form": form,
            "peak_frequency": float(peak_freq),
            "spectral_energy": float(spectral_energy),
            "fft_amps": fft_amps.tolist(),
            "fft_freqs": fft_freqs.tolist()
        }
