"""
Script to generate synthetic object detection dataset from CIFAR-10.
This creates larger canvas images (640x640) with multiple CIFAR-10 objects randomly placed.
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from tqdm import tqdm


class SyntheticDatasetGenerator:
    def __init__(self, cifar_root='./data', output_root='./synthetic_dataset', 
                 canvas_size=640, object_size=32, seed=42):
        """
        Initialize the synthetic dataset generator.
        
        Args:
            cifar_root: Path to download/load CIFAR-10
            output_root: Path to save synthetic dataset
            canvas_size: Size of the output canvas (width, height)
            object_size: Size of CIFAR-10 objects (32x32)
            seed: Random seed for reproducibility
        """
        self.cifar_root = cifar_root
        self.output_root = output_root
        self.canvas_size = canvas_size
        self.object_size = object_size
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Load CIFAR-10 dataset
        print("Loading CIFAR-10 dataset...")
        transform = transforms.ToTensor()
        self.train_dataset = datasets.CIFAR10(root=cifar_root, train=True, 
                                             download=True, transform=transform)
        self.test_dataset = datasets.CIFAR10(root=cifar_root, train=False, 
                                            download=True, transform=transform)
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for the dataset."""
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.output_root, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_root, split, 'labels'), exist_ok=True)
    
    def _create_canvas(self, background_type='random'):
        """Create a canvas with specified background."""
        if background_type == 'black':
            canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        elif background_type == 'white':
            canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 255
        else:  # random
            canvas = np.random.randint(0, 50, (self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        return canvas
    
    def _paste_object(self, canvas, obj_img, x, y):
        """Paste object image onto canvas at specified position."""
        obj_np = (obj_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        canvas[y:y+self.object_size, x:x+self.object_size] = obj_np
        return canvas
    
    def _get_yolo_format(self, x, y, class_id):
        """Convert bounding box to YOLO format (normalized)."""
        x_center = (x + self.object_size / 2) / self.canvas_size
        y_center = (y + self.object_size / 2) / self.canvas_size
        width = self.object_size / self.canvas_size
        height = self.object_size / self.canvas_size
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def generate_image(self, dataset, min_objects=1, max_objects=5):
        """Generate a single synthetic image with multiple objects."""
        canvas = self._create_canvas()
        labels = []
        
        # Random number of objects
        num_objects = random.randint(min_objects, max_objects)
        
        # Track occupied positions to avoid too much overlap
        occupied_positions = []
        
        attempts = 0
        max_attempts = 100
        
        while len(labels) < num_objects and attempts < max_attempts:
            attempts += 1
            
            # Get random CIFAR-10 image
            idx = random.randint(0, len(dataset) - 1)
            img, class_id = dataset[idx]
            
            # Random position (ensure it fits in canvas)
            max_x = self.canvas_size - self.object_size
            max_y = self.canvas_size - self.object_size
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # Check for excessive overlap
            overlap = False
            for ox, oy in occupied_positions:
                if abs(x - ox) < self.object_size and abs(y - oy) < self.object_size:
                    # Allow some overlap but not complete
                    if abs(x - ox) < self.object_size // 2 and abs(y - oy) < self.object_size // 2:
                        overlap = True
                        break
            
            if not overlap:
                # Paste object
                canvas = self._paste_object(canvas, img, x, y)
                
                # Add label
                label = self._get_yolo_format(x, y, class_id)
                labels.append(label)
                occupied_positions.append((x, y))
        
        return canvas, labels
    
    def generate_dataset(self, num_train=10000, num_val=2000, 
                        min_objects=1, max_objects=5):
        """
        Generate the complete synthetic dataset.
        
        Args:
            num_train: Number of training images
            num_val: Number of validation images
            min_objects: Minimum objects per image
            max_objects: Maximum objects per image
        """
        print(f"\nGenerating synthetic dataset:")
        print(f"  Canvas size: {self.canvas_size}x{self.canvas_size}")
        print(f"  Objects per image: {min_objects}-{max_objects}")
        print(f"  Training images: {num_train}")
        print(f"  Validation images: {num_val}")
        
        # Generate training set
        print("\nGenerating training set...")
        for i in tqdm(range(num_train)):
            canvas, labels = self.generate_image(self.train_dataset, min_objects, max_objects)
            
            # Save image
            img_path = os.path.join(self.output_root, 'train', 'images', f'img_{i:06d}.png')
            Image.fromarray(canvas).save(img_path)
            
            # Save labels
            label_path = os.path.join(self.output_root, 'train', 'labels', f'img_{i:06d}.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
        
        # Generate validation set
        print("\nGenerating validation set...")
        for i in tqdm(range(num_val)):
            canvas, labels = self.generate_image(self.test_dataset, min_objects, max_objects)
            
            # Save image
            img_path = os.path.join(self.output_root, 'val', 'images', f'img_{i:06d}.png')
            Image.fromarray(canvas).save(img_path)
            
            # Save labels
            label_path = os.path.join(self.output_root, 'val', 'labels', f'img_{i:06d}.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
        
        print(f"\n✓ Dataset generated successfully at: {self.output_root}")
        
        # Generate data.yaml
        self._generate_data_yaml()
    
    def _generate_data_yaml(self):
        """Generate data.yaml configuration file for YOLO."""
        yaml_content = f"""# Synthetic CIFAR-10 Object Detection Dataset
path: {os.path.abspath(self.output_root)}
train: train/images
val: val/images

# Number of classes
nc: 10

# Class names
names: {self.class_names}
"""
        
        yaml_path = os.path.join(self.output_root, 'data.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"✓ data.yaml created at: {yaml_path}")


def main():
    """Main function to run the generator."""
    generator = SyntheticDatasetGenerator(
        cifar_root='./data',
        output_root='./synthetic_dataset',
        canvas_size=640,
        object_size=32,
        seed=42
    )
    
    generator.generate_dataset(
        num_train=10000,
        num_val=2000,
        min_objects=1,
        max_objects=5
    )
    
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated data.yaml file")
    print("2. Start training with: yolo train data=synthetic_dataset/data.yaml model=yolov8m.pt")


if __name__ == '__main__':
    main()
