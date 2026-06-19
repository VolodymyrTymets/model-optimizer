from termcolor import colored


class Logger:
  def __init__(self, module_name: str):
    self.module_name = module_name

  def log(self, message: str, color: str = None):
    if color:
      print(colored(f'[{self.module_name}]: {message}', color))
    else:
      print(f'[{self.module_name}]: {message}')

  def error(self, err: Exception):
    print(colored(f'[{self.module_name}]: Error -> {err}', 'red'))
