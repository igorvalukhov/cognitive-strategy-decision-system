"""Compatibility entry point for the complete training pipeline."""
from .train_module import train

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    args = parser.parse_args()
    train(args.config, args.dataset, args.output)
