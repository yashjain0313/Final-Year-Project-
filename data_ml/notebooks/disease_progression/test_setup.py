"""
Test script to verify disease progression model setup

Run this before training to ensure everything is configured correctly
"""

import sys
import os

def check_dependencies():
    """Check if all required packages are installed"""
    print("="*50)
    print("CHECKING DEPENDENCIES")
    print("="*50)
    
    required_packages = {
        'tensorflow': 'tensorflow',
        'keras': 'tensorflow.keras',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'PIL': 'pillow'
    }
    
    missing = []
    
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {install_name}")
        except ImportError:
            print(f"✗ {install_name} - NOT FOUND")
            missing.append(install_name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def check_gpu():
    """Check if GPU is available"""
    print("\n" + "="*50)
    print("CHECKING GPU")
    print("="*50)
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ GPU Available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            
            # Check CUDA version
            print(f"\nTensorFlow version: {tf.__version__}")
            print(f"CUDA available: {tf.test.is_built_with_cuda()}")
            
            return True
        else:
            print("⚠️  No GPU detected - training will use CPU")
            print("   This will be significantly slower")
            print("   Consider using Google Colab for free GPU access")
            return False
    
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def check_dataset(dataset_path):
    """Check if dataset is available"""
    print("\n" + "="*50)
    print("CHECKING DATASET")
    print("="*50)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("\nPlease download the dataset:")
        print("1. Install Kaggle API: pip install kaggle")
        print("2. Setup credentials: https://www.kaggle.com/account")
        print("3. Download: kaggle datasets download -d emmarex/plantdisease")
        print("4. Extract to: data_ml/datasets/plant_disease/")
        return False
    
    # Count classes and images
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not classes:
        print(f"❌ No class directories found in: {dataset_path}")
        return False
    
    total_images = 0
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
    
    print(f"✅ Dataset found!")
    print(f"   Path: {dataset_path}")
    print(f"   Classes: {len(classes)}")
    print(f"   Total images: {total_images}")
    
    if total_images < 1000:
        print(f"\n⚠️  Warning: Only {total_images} images found")
        print("   Expected ~54,000 for PlantVillage dataset")
    
    return True


def check_directories():
    """Check if required directories exist"""
    print("\n" + "="*50)
    print("CHECKING DIRECTORIES")
    print("="*50)
    
    required_dirs = [
        'models',
        'logs',
        '../../datasets'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - Creating...")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  ✓ Created {dir_path}")
            except Exception as e:
                print(f"  ✗ Failed to create {dir_path}: {e}")
                all_exist = False
    
    if all_exist:
        print("\n✅ All directories ready!")
    
    return all_exist


def test_model_build():
    """Test if model can be built"""
    print("\n" + "="*50)
    print("TESTING MODEL BUILD")
    print("="*50)
    
    try:
        import tensorflow as tf
        from disease_progression_model import DiseaseProgressionModel
        
        print("Building test model...")
        model_builder = DiseaseProgressionModel(
            num_classes=38,
            sequence_length=5,
            image_size=(224, 224),
            cnn_backbone='efficientnet',
            lstm_units=[256, 128],
            dropout_rate=0.3
        )
        
        model = model_builder.build_model()
        model_builder.compile_model()
        
        total_params = model.count_params()
        print(f"\n✅ Model built successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return True
    
    except Exception as e:
        print(f"❌ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_generator():
    """Test synthetic data generation"""
    print("\n" + "="*50)
    print("TESTING DATA GENERATOR")
    print("="*50)
    
    try:
        import numpy as np
        from disease_progression_model import TemporalDataGenerator
        
        gen = TemporalDataGenerator(sequence_length=5)
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Generate sequence
        sequence = gen.create_progression_sequence(test_image, disease_severity=0.7)
        
        print(f"✅ Data generator working!")
        print(f"   Input shape: {test_image.shape}")
        print(f"   Sequence shape: {sequence.shape}")
        print(f"   Expected: (5, 224, 224, 3)")
        
        if sequence.shape == (5, 224, 224, 3):
            return True
        else:
            print(f"❌ Unexpected sequence shape!")
            return False
    
    except Exception as e:
        print(f"❌ Data generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def estimate_training_time():
    """Estimate training time"""
    print("\n" + "="*50)
    print("TRAINING TIME ESTIMATE")
    print("="*50)
    
    import tensorflow as tf
    
    has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    
    if has_gpu:
        print("With GPU:")
        print("  - Training time: 2-4 hours")
        print("  - Recommended batch size: 16-32")
    else:
        print("With CPU:")
        print("  - Training time: 12-24 hours")
        print("  - Recommended batch size: 4-8")
        print("\n💡 Tip: Use Google Colab for free GPU access")
    
    print("\nFor 50 epochs with ~50,000 images")


def main():
    """Run all checks"""
    print("\n" + "="*70)
    print(" "*15 + "DISEASE PROGRESSION MODEL - SETUP CHECK")
    print("="*70 + "\n")
    
    results = {}
    
    # Run checks
    results['dependencies'] = check_dependencies()
    results['gpu'] = check_gpu()
    results['directories'] = check_directories()
    results['dataset'] = check_dataset('../../datasets/plant_disease/PlantVillage')
    results['model_build'] = test_model_build()
    results['data_generator'] = test_data_generator()
    
    # Estimate training time
    estimate_training_time()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check.replace('_', ' ').title():.<50} {status}")
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to train the model.")
        print("\nNext steps:")
        print("1. Review configuration in train_model.py")
        print("2. Run: python train_model.py")
        print("3. Monitor training with TensorBoard: tensorboard --logdir=logs/")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before training.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install <package>")
        print("- Download dataset from Kaggle")
        print("- Check file paths in train_model.py")
    
    print("\n" + "="*70 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
