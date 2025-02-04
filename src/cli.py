import argparse
import torch
from pathlib import Path

from src.train import train_model, test_model
from src.model.cnn import Net
from src.model.quantized_cnn import QuantizedNet
from src.quantization.weight_quant import quantize_layer_weights
from src.quantization.act_quant import register_activation_profiling_hooks
from src.quantization.bias_quant import quantize_model_biases
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and test CNN models with quantization')
    
    parser.add_argument('--mode', choices=['train', 'test', 'quantize'], 
                       required=True, help='Operation mode')
    parser.add_argument('--model-path', type=Path, default=Path('models/model.pth'),
                       help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training/testing')
    
    return parser.parse_args()

def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    try:
        if args.mode == 'train':
            logger.info(f"Training model for {args.epochs} epochs on {args.device}")
            model = train_model(epochs=args.epochs, device=args.device)
            
            # Create directory if it doesn't exist
            args.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.model_path)
            logger.info(f"Saved model to {args.model_path}")
            
        elif args.mode == 'test' or args.mode == 'quantize':
            if not args.model_path.exists():
                logger.error(f"Model file not found: {args.model_path}")
                return
                
            try:
                # Load base model
                model = Net().to(args.device)
                model.load_state_dict(torch.load(args.model_path))
            except Exception as e:
                logger.error(f"Failed to load model from {args.model_path}: {str(e)}")
                return
                
            if args.mode == 'test':
                accuracy = test_model(model, device=args.device)
                logger.info(f"Test accuracy: {accuracy:.2f}%")
            else:
                # Create and test quantized model
                logger.info("Quantizing model...")
                quantized_model = QuantizedNet(model).to(args.device)
                
                # Profile activations
                quantized_model.profile_activations = True
                register_activation_profiling_hooks(quantized_model)
                
                # Run test to collect activation statistics
                test_model(quantized_model, max_samples=1000)
                
                # Quantize weights and biases
                quantize_layer_weights(quantized_model)
                quantize_model_biases(quantized_model)
                
                # Test quantized model
                accuracy = test_model(quantized_model)
                logger.info(f"Quantized model accuracy: {accuracy:.2f}%")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1
    return 0

if __name__ == '__main__':
    main() 