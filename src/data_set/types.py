from enum import Enum


class ArgumentationTypes(Enum):
  #image
  RandomFlip = 'RandomFlip'
  RandomRotation = 'RandomRotation'
  RandomZoom = 'RandomZoom'
  # signal
  normalization = 'normalization'
  time_stretch = 'time_stretch'
  pitch_shift = 'pitch_shift'
  time_shift = 'time_shift'
  nothing = 'nothing'
