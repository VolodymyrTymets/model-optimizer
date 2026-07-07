from src.model_restorer.model_restorer import ModelRestorer

def main():
    model_restorer = ModelRestorer()
    model_restorer.restore_best_step()

if __name__ == "__main__":
    main()
