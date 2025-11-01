import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse  # For command-line arguments
import re        # For finding exp number
import yaml      # For saving hyperparameters

# --- 1. Configuration (Moved to main() for better control) ---
# --- Data ---
BASE_PATH = 'samples/quick_draw_dataset' 
DATASET_SIZE = 728226 
# --- Model Hyperparameters ---
DEFAULT_CONFIG = {
    'BATCH_SIZE': 32,
    'IMG_SIZE': 256,
    'MAX_SEQ_LEN': 2000,
    'LATENT_DIM': 256,
    'RNN_UNITS': 512,
    'EPOCHS': 30,
    'LEARNING_RATE': 0.0001
}

# --- 2. Data Pipeline (Identical to your code) ---

def get_file_paths(base_path):
    png_files, npy_files = [], []
    key_to_png, key_to_npy = {}, {}
    for root, _, files in os.walk(base_path):
        for f in files:
            key_id = os.path.splitext(f)[0]
            if f.endswith('.png'):
                key_to_png[key_id] = os.path.join(root, f)
            elif f.endswith('.npy'):
                key_to_npy[key_id] = os.path.join(root, f)
    
    for key_id in sorted(set(key_to_png.keys()) & set(key_to_npy.keys())):
        png_files.append(key_to_png[key_id])
        npy_files.append(key_to_npy[key_id])
        
    print(f"Found {len(png_files)} matching file pairs.")
    return png_files, npy_files

def load_npy_wrapper(npy_path_tensor):
    npy_path_bytes = npy_path_tensor.numpy()
    npy_path_str = npy_path_bytes.decode('utf-8')
    sequence = np.load(npy_path_str)
    return sequence.astype(np.float32)

def load_and_preprocess(png_path, npy_path, img_size, max_seq_len):
    img = tf.io.read_file(png_path)
    img = tf.image.decode_png(img, channels=1) 
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    
    [sequence] = tf.py_function(load_npy_wrapper, [npy_path], [tf.float32])
    sequence = sequence[:max_seq_len, :]
    
    seq_len = tf.shape(sequence)[0]
    padding = [[0, max_seq_len - seq_len], [0, 0]]
    sequence = tf.pad(sequence, padding, 'CONSTANT')
    sequence.set_shape([max_seq_len, 5])
    
    decoder_input = sequence
    decoder_target = tf.roll(sequence, shift=-1, axis=0)
    return (img, decoder_input), decoder_target

def create_dataset(base_path, batch_size, img_size, max_seq_len):
    png_files, npy_files = get_file_paths(base_path)
    if not png_files:
        print("No files found. Exiting.")
        return None
    
    path_ds = tf.data.Dataset.from_tensor_slices((png_files, npy_files))
    # Use functools.partial to pass extra args to the map function
    from functools import partial
    map_func = partial(load_and_preprocess, img_size=img_size, max_seq_len=max_seq_len)
    
    dataset = path_ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = (
        dataset
        .shuffle(buffer_size=1024)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return dataset

# --- 3. Model Architecture (Identical to your code) ---

def build_encoder(img_shape, latent_dim):
    image_input = layers.Input(shape=img_shape, name="image_input")
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim, activation='relu', name="encoder_output")(x)
    return keras.Model(image_input, x, name="encoder")

def build_decoder(seq_shape, latent_dim, rnn_units):
    sequence_input = layers.Input(shape=seq_shape, name="sequence_input")
    encoder_state = layers.Input(shape=(latent_dim,), name="encoder_state")
    decoder_initial_state = layers.Dense(rnn_units, name="state_projector")(encoder_state)
    decoder_lstm = layers.LSTM(rnn_units, return_sequences=True, name="decoder_lstm")
    decoder_outputs = decoder_lstm(sequence_input, initial_state=[decoder_initial_state, decoder_initial_state])
    output = layers.Dense(5, activation='linear', name="decoder_output")(decoder_outputs)
    return keras.Model([sequence_input, encoder_state], output, name="decoder")

def custom_loss(y_true, y_pred):
    true_dx_dy = y_true[:, :, 0:2]  
    true_p = y_true[:, :, 2:5]      
    pred_dx_dy = y_pred[:, :, 0:2]
    pred_p = y_pred[:, :, 2:5]
    
    squared_error_dx_dy = tf.square(true_dx_dy - pred_dx_dy) 
    loss_dx_dy = tf.reduce_mean(squared_error_dx_dy, axis=-1) 
    cce_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    loss_p = cce_loss(true_p, pred_p) 
    mask = tf.cast(tf.reduce_any(y_true != 0.0, axis=-1), tf.float32) 
    loss_dx_dy = loss_dx_dy * mask
    loss_p = loss_p * mask
    total_loss = loss_dx_dy + loss_p 
    return tf.reduce_mean(total_loss)

def build_and_compile_model(config):
    img_shape = (config['IMG_SIZE'], config['IMG_SIZE'], 1)
    seq_shape = (config['MAX_SEQ_LEN'], 5)
    
    encoder = build_encoder(img_shape, config['LATENT_DIM'])
    decoder = build_decoder(seq_shape, config['LATENT_DIM'], config['RNN_UNITS'])

    image_input = layers.Input(shape=img_shape, name="image_input")
    sequence_input = layers.Input(shape=seq_shape, name="sequence_input")
    
    encoder_output = encoder(image_input)
    decoder_output = decoder([sequence_input, encoder_output])
    
    model = keras.Model(inputs=[image_input, sequence_input], 
                        outputs=decoder_output, 
                        name="png_to_svg_model")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['LEARNING_RATE']),
        loss=custom_loss
    )
    return model

# --- 4. NEW: Helper Functions for Logging ---

def get_exp_dir(base_dir="runs/train"):
    """
    Finds the next available experiment directory, e.g., 'runs/train/exp2'.
    """
    os.makedirs(base_dir, exist_ok=True)
    dirs = os.listdir(base_dir)
    
    max_num = 0
    for d in dirs:
        if d.startswith('exp'):
            match = re.search(r'exp(\d*)', d)
            if match:
                num_str = match.group(1)
                max_num = max(max_num, int(num_str) if num_str else 1)

    next_exp_num = max_num + 1
    exp_dir = os.path.join(base_dir, f'exp{next_exp_num}' if next_exp_num > 1 else 'exp')
    return exp_dir

def save_batch_previews(dataset, exp_dir, filename_prefix, batch_size):
    """Saves a grid of images from the first batch of a dataset."""
    try:
        (img_batch, _), _ = next(iter(dataset))
        
        # Un-normalize images from [-1, 1] to [0, 1]
        img_batch = (img_batch + 1.0) / 2.0
        
        # Determine grid size (try to make it squarish)
        rows = int(np.sqrt(batch_size))
        cols = batch_size // rows
        if rows * cols < batch_size:
            cols += 1 # Ensure all images fit
            
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axs = axs.flatten()
        
        for i in range(batch_size):
            if i < len(img_batch):
                axs[i].imshow(tf.squeeze(img_batch[i]), cmap='gray')
                axs[i].axis('off')
            else:
                axs[i].axis('off') # Hide unused subplots
                
        plt.tight_layout()
        save_path = os.path.join(exp_dir, f"{filename_prefix}_batch_preview.png")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved batch preview to: {save_path}")

    except Exception as e:
        print(f"Could not save batch preview: {e}")

# --- 5. Main Training Execution (MODIFIED) ---
def main(args):
    
    # --- 5.1 Setup Experiment Directory ---
    if args.resume:
        exp_dir = os.path.abspath(os.path.join(os.path.dirname(args.resume), '..'))
        print(f"--- Resuming training from: {args.resume} ---")
        # Load the hparams from the resumed directory
        try:
            with open(os.path.join(exp_dir, 'hparams.yaml'), 'r') as f:
                config = yaml.safe_load(f)
            print("Loaded hparams from resumed run.")
        except Exception as e:
            print(f"Warning: Could not load hparams.yaml, using defaults. Error: {e}")
            config = DEFAULT_CONFIG
    else:
        exp_dir = get_exp_dir()
        config = DEFAULT_CONFIG
        print(f"--- Starting new training run ---")

    weights_dir = os.path.join(exp_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    best_model_path = os.path.join(weights_dir, 'best.keras')
    last_model_path = os.path.join(weights_dir, 'last.keras')

    # --- 5.2 Print & Save Config ---
    print(f"Saving results to: {exp_dir}")
    print("Using configuration:")
    print(yaml.dump(config))
    
    # Save hparams.yaml
    with open(os.path.join(exp_dir, 'hparams.yaml'), 'w') as f:
        yaml.dump(config, f)

    # --- 5.3 Create Dataset ---
    full_batched_dataset = create_dataset(
        BASE_PATH, 
        config['BATCH_SIZE'], 
        config['IMG_SIZE'], 
        config['MAX_SEQ_LEN']
    )
    
    if full_batched_dataset:
        total_batches = DATASET_SIZE // config['BATCH_SIZE']
        train_steps = int(total_batches * 0.9) # 90/10 split
        val_steps = total_batches - train_steps
        train_data = full_batched_dataset.take(train_steps)
        val_data = full_batched_dataset.skip(train_steps)

        print(f"\nTotal batches: {total_batches}")
        print(f"Training steps per epoch: {train_steps}")
        print(f"Validation steps per epoch: {val_steps}")
        
        # --- 5.4 Save Batch Previews (NEW) ---
        if not args.resume: # Only save on a new run
            save_batch_previews(train_data, exp_dir, "train", config['BATCH_SIZE'])
            save_batch_previews(val_data, exp_dir, "val", config['BATCH_SIZE'])

        # --- 5.5 Create Callbacks ---
        # 1. Save Best Model
        best_model_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path, monitor='val_loss', mode='min',
            save_best_only=True, save_weights_only=False)
        # 2. Save Last Model
        last_model_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=last_model_path, save_freq='epoch')
        # 3. EarlyStopping
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)
        # 4. CSVLogger (NEW)
        results_path = os.path.join(exp_dir, 'results.csv')
        csv_logger = tf.keras.callbacks.CSVLogger(results_path, append=args.resume)

        all_callbacks = [
            best_model_callback, 
            last_model_callback, 
            early_stopping_callback,
            csv_logger
        ]

        # --- 5.6 Build or Load Model ---
        if args.resume:
            print("Loading and re-compiling existing model...")
            model = keras.models.load_model(
                args.resume, 
                custom_objects={'custom_loss': custom_loss}
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config['LEARNING_RATE']),
                loss=custom_loss
            )
        else:
            print("Building new model...")
            model = build_and_compile_model(config)
        
        # Save model summary (NEW)
        summary_path = os.path.join(exp_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        model.summary()

        # --- 5.7 Train the Model! ---
        print("\n--- Starting model.fit() ---")
        history = model.fit(
            train_data,
            epochs=config['EPOCHS'],
            steps_per_epoch=train_steps,
            validation_data=val_data,
            validation_steps=val_steps,
            callbacks=all_callbacks
        )

        print("--- Training Complete ---")
        print(f"Best model saved to: {best_model_path}")
        print(f"Last model saved to: {last_model_path}")

        # --- 5.8 Plot and save the Training Results ---
        print("\nPlotting training and validation loss...")
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(exp_dir, "training_loss_plot.png")
        plt.savefig(plot_path)
        print(f"Training plot saved to: {plot_path}")
        
    else:
        print("Dataset could not be created. Check BASE_PATH.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the PNG-to-SVG model.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to the last.keras weights to resume training from. e.g., runs/train/exp/weights/last.keras')
    args = parser.parse_args()
    main(args)