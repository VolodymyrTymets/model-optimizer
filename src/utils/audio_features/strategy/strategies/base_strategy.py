import os
import numpy as np
import matplotlib.pyplot as plt
import uuid
from PIL import Image, ImageDraw
from src.audio_features.audio_features import FrequencyDomainFeatures
from src.definitions import ASSETS_PATH
from src.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.files import Files

class BaseStrategy(IAFStrategy):
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    self.features = FrequencyDomainFeatures()
    self.frame_length = frame_length
    self.hop_length = hop_length
    self.sr = sr
    self.files = Files()

  def _get_image_path(self, label: str):
    file_name = f"{self.af_type.value}_{uuid.uuid4()}.png"
    directory = self.files.join(ASSETS_PATH, '__af__', self.af_type.value, label)
    self.files.create_folder(directory)
    return os.path.join(directory, file_name)

  def _signal_to_image_matrix(self, signal, height=128, width=None):
    n = len(signal)
    if width is None:
      width = n  # one column per sample (can down/up-sample)
    # resample signal to width
    x = np.linspace(0, n - 1, width).astype(int)
    sig = signal[x]
    # normalize to [0, height-1]
    sig_norm = (sig - sig.min()) / (sig.max() - sig.min() + 1e-12)
    rows = (height - 1 - (sig_norm * (height - 1))).astype(int)
    img = Image.new('L', (width, height), 255)  # white background
    draw = ImageDraw.Draw(img)
    for i in range(width - 1):
      draw.line((i, rows[i], i + 1, rows[i + 1]), fill=0)
    return np.array(img)

  def save_audio_feature(self, af: np.ndarray, label: str):
    matrix = self._signal_to_image_matrix(signal=af) if af.ndim == 1 else af
    file_path = self._get_image_path(label=label)
    normalized = matrix.astype(np.uint8)

    # Create image and save
    img = Image.fromarray(normalized)
    img.save(file_path)
    return matrix

  def get_audio_feature(self, signal: np.ndarray):
    pass