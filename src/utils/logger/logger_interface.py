from abc import ABC


class ILogger(ABC):
  def log(self, message: str, color: str):
    pass

  def error(self, err: Exception):
    pass
