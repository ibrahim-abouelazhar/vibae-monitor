import os
import unittest
import numpy as np
from fastapi.testclient import TestClient

from src.data_loader import VibrationPreprocessor
from src.api.main import app

class TestVibAEMonitor2D(unittest.TestCase):
    def setUp(self):
        self.window_size = 2240
        self.step_size = 512
        self.fs = 12000
        self.preprocessor = VibrationPreprocessor(
            window_size=self.window_size, 
            step_size=self.step_size, 
            fs=self.fs
        )
        self.client = TestClient(app)

    def test_preprocessor_stft_and_reconstruction(self):
        # Create a synthetic signal
        t = np.linspace(0, 1, self.fs)
        # 100 Hz signal
        signal_data = np.sin(2 * np.pi * 100 * t)
        
        # Segment
        windows = self.preprocessor.segment_signal(signal_data)
        self.assertTrue(len(windows) > 0)
        self.assertEqual(windows.shape[1], self.window_size)
        
        # Test STFT computation
        first_window = windows[0]
        mag_128, original_stft = self.preprocessor.compute_stft(first_window)
        
        self.assertEqual(mag_128.shape, (128, 32))
        self.assertEqual(original_stft.shape, (129, 32))
        
        # Test Reconstruction (iSTFT)
        recon_signal = self.preprocessor.reconstruct_signal_from_stft(mag_128, original_stft)
        self.assertEqual(recon_signal.shape, (self.window_size,))
        # Assert reconstruction is highly accurate
        max_diff = np.max(np.abs(first_window - recon_signal))
        self.assertTrue(max_diff < 1e-4)

    def test_preprocessor_features(self):
        t = np.linspace(0, 1, self.fs)
        signal_data = np.sin(2 * np.pi * 100 * t)
        windows = self.preprocessor.segment_signal(signal_data)
        first_window = windows[0]
        
        features = self.preprocessor.extract_features(first_window)
        self.assertIn("rms", features)
        self.assertIn("kurtosis", features)
        self.assertIn("variance", features)
        self.assertIn("peak_frequency", features)
        
        self.assertAlmostEqual(features["rms"], 0.707, places=1)
        self.assertAlmostEqual(features["peak_frequency"], 100.0, delta=15.0)

    def test_api_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertEqual(data["window_size"], 2240)

    def test_api_files(self):
        response = self.client.get("/files")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_api_analyze(self):
        # Create a dummy window of size 2240 (flat signal)
        dummy_window = [0.01] * 2240
        response = self.client.post("/analyze", json={"window": dummy_window})
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("mse", data)
        self.assertIn("status", data)
        self.assertIn("threshold", data)
        self.assertIn("reconstructed", data)
        self.assertIn("features", data)
        self.assertIn("spectrogram", data)
        self.assertIn("recon_spectrogram", data)
        self.assertIn("fault_class", data)
        self.assertIn("baseline_class", data)
        
        self.assertEqual(len(data["reconstructed"]), 2240)
        self.assertEqual(len(data["spectrogram"]), 128)
        self.assertEqual(len(data["spectrogram"][0]), 32)
        self.assertIn(data["status"], ["NORMAL", "ANOMALIE"])
        
        # Test threshold override
        high_override = 999.0
        response_override = self.client.post("/analyze", json={
            "window": dummy_window,
            "threshold_override": high_override
        })
        self.assertEqual(response_override.status_code, 200)
        data_override = response_override.json()
        self.assertEqual(data_override["threshold"], high_override)
        self.assertEqual(data_override["status"], "NORMAL")  # Since MSE is < 999.0

    def test_api_predict_live(self):
        # First get the files
        files_resp = self.client.get("/files")
        self.assertEqual(files_resp.status_code, 200)
        files = files_resp.json()
        self.assertTrue(len(files) > 0)
        
        # Select first file
        filename = files[0]["filename"]
        
        # Post to /predict/live
        payload = {
            "file": filename,
            "window_index": 0
        }
        response = self.client.post("/predict/live", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["window_index"], 0)
        self.assertIn("total_windows", data)
        self.assertIn("signal_1d", data)
        self.assertIn("spectrogram", data)
        self.assertIn("signal_reconstructed", data)
        self.assertIn("fft_freqs", data)
        self.assertIn("fft_amplitudes", data)
        self.assertIn("fft_peak_hz", data)
        self.assertIn("mse", data)
        self.assertIn("threshold", data)
        self.assertIn("is_anomaly", data)
        self.assertIn("rms", data)
        self.assertIn("kurtosis", data)
        self.assertIn("variance", data)
        self.assertIn("timestamp", data)
        
        # Verify sizes
        self.assertEqual(len(data["signal_1d"]), 2240)
        self.assertEqual(len(data["signal_reconstructed"]), 2240)
        self.assertEqual(len(data["spectrogram"]), 128)
        self.assertEqual(len(data["spectrogram"][0]), 32)

if __name__ == "__main__":
    unittest.main()

