import librosa
import numpy as np
import math
from scipy import signal as sp_signal


class TimeDomainFeatures:

  def __init__(self):
    pass

  def duration(self, signal: np.ndarray, sr: int):
    return 1 / sr * len(signal)

  def _frame(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length, axis=0)

  def amplitude_envelope(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return np.array([np.max(np.abs(frame)) for frame in self._frame(signal, frame_length, hop_length)])

  def root_mean_square_energy(self, signal: np.ndarray, frame_length: int, hop_length: int):
    frames = self._frame(signal, frame_length, hop_length)
    return np.sqrt(np.mean(frames ** 2, axis=0))

  def zero_crossing_rate(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return librosa.feature.zero_crossing_rate(signal, frame_length=frame_length, hop_length=hop_length)[0]

  def AE(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return self.amplitude_envelope(signal=signal, frame_length=frame_length, hop_length=hop_length)

  def RSME(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return self.root_mean_square_energy(signal=signal, frame_length=frame_length, hop_length=hop_length)

  def ZCR(self, signal: np.ndarray, frame_length: int, hop_length: int):
    return self.zero_crossing_rate(signal=signal, frame_length=frame_length, hop_length=hop_length)


class FrequencyDomainFeatures:

  def __init__(self):
    pass

  def _freq_for_magnitude(self, magnitude: np.array, sr: int):
    return np.linspace(0, sr, len(magnitude))

  def _magnitude(self, signal: np.array, f_ration: float = 0.5):
    ft = np.fft.fft(signal)
    magnitude = np.abs(ft)
    nup_freq_bins = int(len(magnitude) * f_ration)
    return magnitude[:nup_freq_bins]

  def _calculate_split_frequency_bin(self, sr: int, split_frequency: int, num_frequency_bins: int):
    """Infer the frequency bin associated to a given split frequency."""

    frequency_range = sr / 2
    frequency_delta_per_bin = frequency_range / num_frequency_bins
    split_frequency_bin = math.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)

  def fft(self, signal: np.ndarray, sr: int, frame_length: int, hop_length: int):
    # Step 1 framing
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length, axis=0)
    # Step 2 windowing
    window = sp_signal.windows.hamming(frame_length)
    # Step 3 FFT
    magnitudes = [self._magnitude(frame * window, 0.5) for frame in frames]
    # Step 4 Aggregation
    magnitude = np.mean(magnitudes, axis=0)
    frequency = self._freq_for_magnitude(magnitude, sr)
    return magnitude, frequency

  def stft(self, signal: np.ndarray, frame_length: int, hop_length: int, log_scale: bool = False):
    s_scale = librosa.stft(signal, n_fft=frame_length, hop_length=hop_length)
    return np.abs(s_scale) ** 2 if log_scale == True else s_scale

  def melfilters(self, sr: int, frame_length: int, n_mels: int = 128):
    return librosa.filters.mel(n_fft=frame_length, sr=sr, n_mels=n_mels)

  def melspectogram(self, signal: np.ndarray, sr: int, frame_length: int, hop_length: int, n_mels: int = 128):
    return librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels)

  def mfcc(self, signal: np.ndarray, sr: int, n_mfcc: int = 12):
    return librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, sr=sr)

  def mfcc_derivatives(self, signal: np.ndarray, sr: int, n_mfcc: int = 12):
    mfccs = self.mfcc(signal, sr, n_mfcc)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    return np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

  def band_energy_ratio(self, signal: np.ndarray, sr: int, frame_length: int, hop_length: int, split_frequency: int):
    stft = self.stft(signal, frame_length, hop_length, log_scale=False)
    """Calculate band energy ratio with a given split frequency."""

    split_frequency_bin = self._calculate_split_frequency_bin(sr, split_frequency, len(stft[0]))
    band_energy_ratio = []

    # calculate power spectrogram
    power_spectrogram = np.abs(stft) ** 2
    power_spectrogram = power_spectrogram.T

    # calculate BER value for each frame
    for frame in power_spectrogram:
      sum_power_low_frequencies = frame[:split_frequency_bin].sum()
      sum_power_high_frequencies = frame[split_frequency_bin:].sum()
      band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
      band_energy_ratio.append(band_energy_ratio_current_frame)

    return np.array(band_energy_ratio)

  def spectral_centroid(self, signal: np.ndarray, sr: int, frame_length: int, hop_length: int):
    return librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

  def spectral_bandwidth(self, signal: np.ndarray, sr: int, frame_length: int, hop_length: int):
    return librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

  def BER(self, signal: np.ndarray, sr: int, frame_length: int, hop_length: int, split_frequency: int):
    return self.band_energy_ratio(signal=signal, sr=sr, frame_length=frame_length, hop_length=hop_length,
                                  split_frequency=split_frequency)

  def SC(self, signal: np.ndarray, sr: int, frame_length: int, hop_length: int):
    return self.spectral_centroid(signal=signal, sr=sr, frame_length=frame_length, hop_length=hop_length)

  def BW(self, signal: np.ndarray, sr: int, frame_length: int, hop_length: int):
    return self.spectral_bandwidth(signal=signal, sr=sr, frame_length=frame_length, hop_length=hop_length)
