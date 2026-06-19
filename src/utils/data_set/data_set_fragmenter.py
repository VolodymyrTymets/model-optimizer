import numpy as np
import uuid
from src.logger.logger_service import Logger
from src.data_set.data_set_file_worker import DataSetFileWorker
from src.audio_features.audio_features import TimeDomainFeatures, FrequencyDomainFeatures

# on how much (in percent) amplitude should be upper silence to determinate start of fragment
MIN_FRAGMENT_LENGTH = 0.005
MAX_FRAGMENT_LENGTH = 0.05
THRESHOLD_OF_SILENCE_BY_FREQ = 15
frame_length = 512
hop_length = 48

class DataSetFragmenter(DataSetFileWorker):
  def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str]):
    super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)
    self.sample_size = None
    self.fragment = []
    self.noise = []
    self.sr = None
    self.file_name = None
    self.fd = FrequencyDomainFeatures()
    self.logger = Logger('DataSetFragmenter')

  @property
  def min_fragment_length(self):
    if self.sr is None:
      raise ValueError('sr is None')
    return int(self.sr / MIN_FRAGMENT_LENGTH)
  @property
  def max_fragment_length(self):
    if self.sr is None:
      raise ValueError('sr is None')
    return int(self.sr * MAX_FRAGMENT_LENGTH)

  def save_fragment(self, amplitude_chunk):
    self.fragment = np.concatenate((self.fragment, amplitude_chunk))

  def save_noise(self, amplitude_chunk):
    self.noise = np.concatenate((self.noise, amplitude_chunk))

  def clear_fragment(self):
    self.fragment = []

  def clear_noise(self):
    self.noise = []

  def to_chunks(self, lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
      yield lst[i:i + n]

  def write_fragments(self, signal, set_name: str, se_type='fragments'):
    if self.file_name is None:
      raise ValueError('file_name is None')
    if self.sr is None:
      raise ValueError('sr is None')
    out_path = self.files.join(self.files.ASSETS_PATH, self.out_path, set_name, se_type)
    self.files.create_folder(out_path)
    file_name = self.files.join(out_path, f'{self.file_name}_' + uuid.uuid4().hex + '.wav')
    self.logger.log(f'--> write to: {file_name}')
    self.wave_file.write(file_name, signal, self.sr)

  def _read_data_set(self, set_name: str, log: bool = True):
      path = self.files.join(self.files.ASSETS_PATH, self.in_path, set_name)
      files = self.files.get_only_files(path)

      for file in files:
        file_path = self.files.join(path, file)
        if log:
          self.logger.log(f'--> read from: {file_path}', 'blue')
        signal, sr = self.wave_file.read(file_path=file_path)

        yield signal, sr, set_name, path, file

  def find_fragment(self, chunk):
    if len(chunk) < frame_length:
      return
    mag, freq = self.fd.fft(chunk, sr=self.sr, frame_length=frame_length, hop_length=hop_length)
    max_freq = freq[np.argmax(mag)]
    self.logger.log(f'max_freq: {max_freq}')

    len_fragment = len(self.fragment)

    # find in which position of fragment current chunk
    is_start = (len_fragment < self.min_fragment_length)
    is_tail = (self.min_fragment_length < len_fragment < self.max_fragment_length)
    is_end = (len_fragment >= self.max_fragment_length)
    if max_freq >  THRESHOLD_OF_SILENCE_BY_FREQ and is_start:
      self.save_fragment(amplitude_chunk=chunk)
    elif is_tail:
      self.save_fragment(amplitude_chunk=chunk)
    # elif is_end:
    #   self.write_fragments(self.noise, 'noise')
    #   self.write_fragments(self.fragment, 'fragments')
    #   self.clear_fragment()
    #   self.clear_noise()
    else:
      self.save_noise(amplitude_chunk=chunk)
      # self.clear_fragment()

  def start(self):
    self.logger.log('Start fragmenting')
    self.files.create_folder(self.files.join(self.files.ASSETS_PATH, self.out_path))
    for set_name in self.set_names:
      for i, data in enumerate(self._read_data_set(set_name=set_name)):
        signal, sr, set_name, path, file = data
        self.sr = sr
        self.file_name = file
        for chunk in self.to_chunks(signal, frame_length * 4):
          self.find_fragment(chunk)

      self.write_fragments(self.noise, set_name=set_name, se_type='noise')
      self.write_fragments(self.fragment, set_name=set_name, se_type='stimulation')
      self.clear_fragment()
      self.clear_noise()
    self.logger.log('End fragmenting')