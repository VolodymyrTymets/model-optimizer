from src.audio_features.audio_features import FrequencyDomainFeatures
from src.audio_features.types import AFTypes
from src.audio_features.strategy.strategies.wave_strategy import WaveStrategy
from src.audio_features.strategy.strategies.ae_strategy import AEStrategy
from src.audio_features.strategy.strategies.rms_strategy import RMStrategy
from src.audio_features.strategy.strategies.zcr_strategy import ZCRtrategy
from src.audio_features.strategy.strategies.fft_strategy import FFTStrategy
from src.audio_features.strategy.strategies.stft_strategy import STFTStrategy
from src.audio_features.strategy.strategies.ber_strategy import BERtrategy
from src.audio_features.strategy.strategies.sc_strategy import SCStrategy
from src.audio_features.strategy.strategies.bw_strategy import BWtrategy
from src.audio_features.strategy.strategies.mel_strategy import MelStrategy
from src.audio_features.strategy.strategies.mfcc_strategy import MFCCStrategy


class AFStrategyFactory:
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    self.features = FrequencyDomainFeatures()
    self.sr = sr
    self.frame_length = frame_length
    self.hop_length = hop_length

  def create_strategy(self, strategy_type: AFTypes):
    if strategy_type.value == AFTypes.wave.value:
      return WaveStrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    elif strategy_type.value == AFTypes.ae.value:
      return AEStrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    elif strategy_type.value == AFTypes.rms.value:
      return RMStrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    elif strategy_type.value == AFTypes.zcr.value:
      return ZCRtrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    elif strategy_type.value == AFTypes.ae.value:
      return AEStrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    elif strategy_type.value == AFTypes.fft.value:
      return FFTStrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    elif strategy_type.value == AFTypes.ber.value:
      return BERtrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    elif strategy_type.value == AFTypes.sc.value:
      return SCStrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    elif strategy_type.value == AFTypes.bw.value:
      return BWtrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    elif strategy_type.value == AFTypes.stft.value:
      return STFTStrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    elif strategy_type.value == AFTypes.mel.value:
      return MelStrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    elif strategy_type.value == AFTypes.mfcc.value:
      return MFCCStrategy(sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)
    else:
      raise ValueError(f"Unknown strategy type: {strategy_type.value}")
