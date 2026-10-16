import os

# Set before any `src` import: src.definitions reads the task at import and a shell value would change the signal shape.
os.environ['MODEL_OPTIMIZER_TASK'] = 'sound'
