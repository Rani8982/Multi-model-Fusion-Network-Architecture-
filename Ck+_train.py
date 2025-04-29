#CODE HAS BEEN RUN WITH EPOCHS

import numpy as np
import tensorflow as tf
import os
import cv2
import h5py
import skimage
from skimage.util import random_noise
import logging
from pathlib import Path

from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Conv2D, Add, MaxPooling2D, AveragePooling2D, Activation, Dense, PReLU, Layer,
    Input, BatchNormalization, GlobalAveragePooling2D, Concatenate, 
    Cropping2D, Multiply, Lambda, Flatten, Reshape
)
from tensorflow.keras.activations import relu, softmax, sigmoid, tanh
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# GPU Configuration
def configure_gpu(gpu_id='0'):
    """Configure GPU memory growth to avoid memory allocation issues"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Enabled memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.error(f"Memory growth setting failed: {e}")
    else:
        logger.warning("No GPUs detected. Running on CPU.")

# Define the custom Patches layer
class Patches(Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, inputs, **kwargs):
        patch = tf.concat((tf.split(inputs, num_or_size_splits=7, axis=1)), axis=-1)
        patch = tf.concat((tf.split(patch, num_or_size_splits=7, axis=2)), axis=-1)
        return patch

    def get_config(self):
        config = {'patch_size': self.patch_size}
        base_config = super(Patches, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def Global_Net(input, eps):
    x_g_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation=relu)(input)
    x_g_1 = BatchNormalization(axis=-1, epsilon=eps)(x_g_1)
    x_g_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_g_1)
    x_g_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_g_1)
    x_g_1 = BatchNormalization(axis=-1, epsilon=eps)(x_g_1)
    x_g_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_g_1)

    x_g_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_g_1)
    x_g_2 = BatchNormalization(axis=-1, epsilon=eps)(x_g_2)
    x_g_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_g_2)

    x_g_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_g_2)
    x_g_3 = BatchNormalization(axis=-1, epsilon=eps)(x_g_3)
    x_g_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_g_3)

    x_g_4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_g_3)
    x_g_4 = BatchNormalization(axis=-1, epsilon=eps)(x_g_4)
    x_g_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_g_4)

    return x_g_4

def Local_Net(input, eps):
    x_l_1 = Conv2D(filters=128, kernel_size=(7, 7), strides=(1, 1), padding='same', activation=relu)(input)
    x_l_1 = BatchNormalization(axis=-1, epsilon=eps)(x_l_1)
    x_l_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_l_1)
    x_l_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_l_1)
    x_l_1 = BatchNormalization(axis=-1, epsilon=eps)(x_l_1)
    x_l_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_l_1)

    x_l_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_l_1)
    x_l_2 = BatchNormalization(axis=-1, epsilon=eps)(x_l_2)
    x_l_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_l_2)

    x_l_3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_l_2)
    x_l_3 = BatchNormalization(axis=-1, epsilon=eps)(x_l_3)
    x_l_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_l_3)

    x_l_4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)(x_l_3)
    x_l_4 = BatchNormalization(axis=-1, epsilon=eps)(x_l_4)
    x_l_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_l_4)

    return x_l_4

def Enhance_Net(eps=1.1e-5, input_shape=(224, 224, 3)):
    """
    Create enhanced facial emotion recognition model with global and local branches
    """
    input = Input(shape=input_shape)

    x_l_0 = Patches(patch_size=32)(input)
    x_g_0 = input

    x_l = Local_Net(x_l_0, eps)
    x_g = Global_Net(x_g_0, eps)

    x_l = GlobalAveragePooling2D()(x_l)
    x_g = GlobalAveragePooling2D()(x_g)

    x_c = Concatenate()([x_l, x_g])

    x_l = Dense(units=1024, activation=relu, name='local_dense_1')(x_l)
    x_g = Dense(units=1024, activation=relu, name='global_dense_1')(x_g)
    #x_c = Concatenate()([x_l, x_g])

    share_1 = Dense(units=2048, activation=relu, name='shared_dense_1')
    share_2 = Dense(units=2048, activation=relu, name='shared_dense_2')

    x_l = share_1(x_l)
    x_g = share_1(x_g)
    x_c = share_1(x_c)

    x_l = share_2(x_l)
    x_g = share_2(x_g)
    x_c = share_2(x_c)

    out_l = Dense(units=7, activation=softmax, name='local_branch')(x_l)
    out_g = Dense(units=7, activation=softmax, name='global_branch')(x_g)
    out_c = Dense(units=7, activation=softmax, name='combined_branch')(x_c)

    return Model(input, [out_l, out_g, out_c])

def load_data(path, img_size=(224, 224)):
    """Load and preprocess facial expression images from directory structure"""
    label_map = {
        'anger': 0,
        'contempt': 1,
        
        'disgust': 2,
        'fear': 3,
        'happy': 4,
        'sadness': 5,
        'surprise': 6
    }

    samples = []
    labels = []
    skipped = 0
    total_found = 0

    # Make sure the path exists
    if not os.path.exists(path):
        logger.error(f"Data path does not exist: {path}")
        # Print all available directories to help debug
        parent_dir = os.path.dirname(path)
        if os.path.exists(parent_dir):
            logger.info(f"Contents of parent directory {parent_dir}: {os.listdir(parent_dir)}")
        raise FileNotFoundError(f"Data path not found: {path}")

    logger.info(f"Loading data from {path}")
    logger.info(f"Directory contents: {os.listdir(path)}")

    for emotion, idx in label_map.items():
        folder_path = os.path.join(path, emotion)
        if not os.path.exists(folder_path):
            logger.warning(f"Missing folder: {folder_path}")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_found += len(files)
        logger.info(f"Found {len(files)} images in {emotion} category")
        
        for file in files:
            img_path = os.path.join(folder_path, file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Unable to read image: {img_path}")
                    skipped += 1
                    continue
                    
                img = cv2.resize(img, img_size)
                samples.append(img.astype('float32') / 255.0)
                labels.append(idx)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                skipped += 1

    if len(samples) == 0:
        logger.error(f"No valid images found in {path}. Total images found: {total_found}, Skipped: {skipped}")
        raise ValueError(f"No valid images loaded from {path}")

    if skipped > 0:
        logger.warning(f"Skipped {skipped} unreadable images")
        
    logger.info(f"Successfully loaded {len(samples)} images with {len(set(labels))} emotion classes")
    return np.array(samples), np.array(labels)


def apply_blur(samples, k_size=5):
    """Apply Gaussian blur to a batch of images"""
    return np.array([cv2.GaussianBlur(img, (k_size, k_size), sigmaX=0) for img in samples])


def apply_noise(samples, noise_mode='gaussian', var=0.01):
    """Apply noise to a batch of images with configurable parameters"""
    noisy = []
    for img in samples:
        noisy_img = random_noise(img, mode=noise_mode, var=var, clip=True)
        noisy.append(noisy_img.astype('float32'))
    return np.array(noisy)


def create_k_fold_splits(x, y, k=5):
    """Create k-fold splits for cross-validation"""
    if len(x) == 0:
        raise ValueError("Cannot create k-fold splits with empty data")
        
    indices = np.random.permutation(len(x))
    x, y = x[indices], y[indices]  # Shuffle the data
    
    fold_size = len(x) // k
    folds = []
    
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else len(x)
        folds.append((x[start_idx:end_idx], y[start_idx:end_idx]))
        
    return folds


def main():
    # Configuration
    data_path = '/kaggle/input/ck-dataset/CK+48'  # Path from original code
    #alternative_paths = [
        #'./CK+48',
        #'../input/ck-dataset/CK+48',
        #'/content/CK+48'
    #]
    
    # Try different possible paths
    for path in [data_path]:
        if os.path.exists(path):
            data_path = path
            logger.info(f"Found valid data path: {data_path}")
            break
    
    # Use mock data if no valid path is found
    if not os.path.exists(data_path):
        logger.warning(f"No valid data path found. Creating synthetic data for testing.")
        # Create synthetic data for testing
        num_samples = 500
        img_size = (224, 224)
        num_classes = 7
        
        # Generate random images and labels
        x = np.random.random((num_samples, *img_size, 3)).astype('float32')
        y = np.random.randint(0, num_classes, size=num_samples)
        logger.info(f"Created synthetic dataset with {num_samples} samples")
    else:
        # Load real data
        img_size = (224, 224)
        num_classes = 7
        x, y = load_data(path=data_path, img_size=img_size)
    
    logger.info(f"Data shape: {x.shape}, Labels shape: {y.shape}")
    
    # Other configuration
    batch_size = 32
    epochs = 200  # Reduced for testing
    learning_rate = 0.001
    k_folds = 5
    
    # GPU configuration
    configure_gpu(gpu_id='0')  # Change as needed
    
    # Create directories
    weights_dir = Path('./weights')
    outputs_dir = Path('./outputs')
    weights_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    # Create k-fold splits
    folds = create_k_fold_splits(x, y, k=k_folds)
    logger.info(f"Created {k_folds} folds for cross-validation")
    
    # Track metrics across folds
    fold_accuracies = []
    
    # Run k-fold cross-validation
    for i in range(k_folds):
        logger.info(f"========== Processing Fold {i+1}/{k_folds} ==========")
        
        # Prepare data for this fold
        x_test, y_test = folds[i]
        x_train = np.concatenate([folds[j][0] for j in range(k_folds) if j != i], axis=0)
        y_train = np.concatenate([folds[j][1] for j in range(k_folds) if j != i], axis=0)
        
        # Apply augmentations
        logger.info("Applying data augmentation...")
        x_test_aug = apply_noise(x_test)
        x_train_aug = apply_noise(x_train)
        
        # Convert to categorical
        y_test_cat = to_categorical(y_test, num_classes)
        y_train_cat = to_categorical(y_train, num_classes)
        
        logger.info(f"Training shapes - X: {x_train_aug.shape}, Y: {y_train_cat.shape}")
        logger.info(f"Testing shapes - X: {x_test_aug.shape}, Y: {y_test_cat.shape}")
        
        # Build model
        logger.info("Building model...")
        model = Enhance_Net(input_shape=x_train_aug[0].shape)
        model.summary()
        
        # Fix: Get correct output layer names for feature extraction
        # Looking for layers ending with 'branch'
        branch_layers = ['local_branch', 'global_branch', 'combined_branch']
        
        logger.info(f"Model output layers: {branch_layers}")
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate),
            loss=['categorical_crossentropy'] * 3,
            metrics=['accuracy'] * 3
        )
        
        # Split data for validation
        val_split = int(0.2 * len(x_train_aug))
        x_val = x_train_aug[:val_split]
        y_val = [y_train_cat[:val_split]] * 3  # Triplicate for three outputs
        
        x_train_final = x_train_aug[val_split:]
        y_train_final = [y_train_cat[val_split:]] * 3  # Triplicate for three outputs
        
        logger.info(f"Training with {len(x_train_final)} samples, validating with {len(x_val)} samples")
        
        # Setup callbacks with dynamic metric names
        # Train the model for a few epochs to discover the actual metric names
        logger.info("Training for 1 epoch to determine metric names...")
        history = model.fit(
            x=x_train_final[:min(32, len(x_train_final))],  # Just use a small batch
            y=[y_train_cat[val_split:min(val_split+32, len(y_train_cat))]] * 3,
            batch_size=batch_size,
            epochs=1,
            verbose=1,
            validation_data=(x_val[:min(32, len(x_val))], [y_val[0][:min(32, len(y_val[0]))]] * 3)
        )
        
        # Get available metrics from history
        available_metrics = list(history.history.keys())
        logger.info(f"Available metrics: {available_metrics}")
        
        # Find the validation accuracy metric for the combined branch (third output)
        # Look for metrics related to the combined output
        combined_metrics = [m for m in available_metrics if 'val' in m and 'combined_branch' in m and 'accuracy' in m]
        if combined_metrics:
            monitor_metric = combined_metrics[0]
        else:
            # Fall back to any validation accuracy metric
            all_val_acc = [m for m in available_metrics if 'val' in m and 'accuracy' in m]
            monitor_metric = all_val_acc[-1] if all_val_acc else 'val_loss'
            
        logger.info(f"Selected monitoring metric: {monitor_metric}")
        
        # Setup callbacks with the correct metric name
        weights_path = weights_dir / f"best_weights_fold_{i+1}.weights.h5"
        
        callbacks = [
            ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=0.1,
                patience=10,
                verbose=1,
                mode='max',
                min_delta=0.0001,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath=str(weights_path),
                monitor=monitor_metric,
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
                mode='max'
            ),
            # Add early stopping
            EarlyStopping(
                monitor=monitor_metric,
                patience=15,
                verbose=1,
                mode='max',
                restore_best_weights=True
            )
        ]
        
        # Continue training with the proper callbacks
        logger.info(f"Continuing training fold {i+1} with proper metric monitoring...")
        history = model.fit(
            x=x_train_final,
            y=[y_train_final[0]] * 3,  # Make sure to use the correct format
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=(x_val, y_val),
            initial_epoch=1  # Continue from epoch 1 since we already did epoch 0
        )
        
        # Load best weights
        if weights_path.exists():
            logger.info(f"Loading best weights from {weights_path}")
            model.load_weights(str(weights_path))
        else:
            logger.warning(f"Best weights file not found: {weights_path}")
        
        # Evaluate on test set
        logger.info(f"Evaluating fold {i+1}...")
        test_results = model.evaluate(
            x=x_test_aug,
            y=[y_test_cat, y_test_cat, y_test_cat],
            verbose=1
        )
        
        # Get combined output accuracy (corresponding to combined_branch)
        accuracy_indices = [i for i, name in enumerate(model.metrics_names) if 'accuracy' in name and 'combined_branch' in name]
        if accuracy_indices:
            test_accuracy = test_results[accuracy_indices[0]]
            fold_accuracies.append(test_accuracy)
            logger.info(f"Fold {i+1} test accuracy (combined branch): {test_accuracy:.4f}")
        else:
            # Fallback to last accuracy if we can't find the combined branch accuracy
            accuracy_indices = [i for i, name in enumerate(model.metrics_names) if 'accuracy' in name]
            if accuracy_indices:
                test_accuracy = test_results[accuracy_indices[-1]]
                fold_accuracies.append(test_accuracy)
                logger.info(f"Fold {i+1} test accuracy (fallback): {test_accuracy:.4f}")
            else:
                logger.warning("Could not determine accuracy from model metrics")
        
        # Extract features for future use
        logger.info("Extracting features...")
        
        # Fix: Use intermediate layers for feature extraction instead of output layers
        feature_layer_names = [
            'local_dense_1',    # Local branch features
            'global_dense_1',   # Global branch features
            'shared_dense_2'    # Shared features before final outputs
        ]
        
        # Check if the layers exist in the model
        available_feature_layers = []
        for name in feature_layer_names:
            try:
                layer = model.get_layer(name)
                available_feature_layers.append(name)
            except ValueError:
                logger.warning(f"Layer {name} not found in model, skipping")
        
        if not available_feature_layers:
            logger.error("No feature extraction layers found. Cannot create feature model.")
            continue
            
        # Create feature extraction model
        try:
            feature_model = Model(
                inputs=model.input,
                outputs=[model.get_layer(name).output for name in available_feature_layers]
            )
            
            logger.info(f"Created feature extraction model with layers: {available_feature_layers}")
            
            # Predict features
            if len(available_feature_layers) == 1:
                # Handle single output case
                x_outputs_train = [feature_model.predict(x=x_train_aug, batch_size=batch_size, verbose=1)]
                x_outputs_test = [feature_model.predict(x=x_test_aug, batch_size=batch_size, verbose=1)]
            else:
                # Handle multiple outputs case
                x_outputs_train = feature_model.predict(x=x_train_aug, batch_size=batch_size, verbose=1)
                x_outputs_test = feature_model.predict(x=x_test_aug, batch_size=batch_size, verbose=1)
                
                # Make sure it's a list for consistent indexing
                if not isinstance(x_outputs_train, list):
                    x_outputs_train = [x_outputs_train]
                    x_outputs_test = [x_outputs_test]
            
            # Save features
            output_path = outputs_dir / f"features_fold_{i+1}.h5"
            logger.info(f"Saving features to {output_path}")
            
            with h5py.File(str(output_path), 'w') as f:
                # Training features
                for idx, name in enumerate(available_feature_layers):
                    f.create_dataset(f'x_{name}_train', data=x_outputs_train[idx])
                f.create_dataset('y_train', data=y_train_cat)
                
                # Testing features
                for idx, name in enumerate(available_feature_layers):
                    f.create_dataset(f'x_{name}_test', data=x_outputs_test[idx])
                f.create_dataset('y_test', data=y_test_cat)
                
            logger.info(f"Successfully saved features for fold {i+1}")
            
        except Exception as e:
            logger.error(f"Error creating or using feature model: {str(e)}")
            logger.error(f"Error details: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Report overall results
    if fold_accuracies:
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        logger.info("====== Cross-Validation Results ======")
        logger.info(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        logger.info(f"Individual fold accuracies: {fold_accuracies}")


if __name__ == '__main__':
    main()
