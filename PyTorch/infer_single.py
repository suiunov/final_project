import argparse
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os
from model import LYT

def main():
    parser = argparse.ArgumentParser(description="Infer single image using LYT-Net")
    parser.add_argument('--image', type=str, required=True, help="Path to input low-light image")
    parser.add_argument('--weights', type=str, default='best_model_LOLv1.pth', help="Path to weights")
    parser.add_argument('--output', type=str, default='enhanced_output.png', help="Path to save output")
    args = parser.parse_args()

    # Automatically use Apple Silicon GPU (mps) if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = LYT().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    # Load and transform image
    if not os.path.exists(args.image):
        print(f"Error: Could not find image {args.image}")
        return

    img = Image.open(args.image).convert('RGB')
    
    # LYT-Net uses multi-stage architectures that downsample. 
    # Therefore, input width and height must be divisible by 8.
    w, h = img.size
    new_w = w - (w % 8)
    new_h = h - (h % 8)
    if new_w != w or new_h != h:
        print(f"Note: Resizing image from {w}x{h} to {new_w}x{new_h} to be divisible by 8.")
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    print("Enhancing image...")
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.clamp(output, 0, 1)
        
    # Save Output
    save_image(output, args.output)
    print(f"Success! Saved enhanced image to: {args.output}")

if __name__ == '__main__':
    main()
