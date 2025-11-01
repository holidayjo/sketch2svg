import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import random
# To display images in a Jupyter notebook
from IPython.display import display, SVG
import matplotlib.pyplot as plt

# --- 1. Configuration ---
# --- Model Hyperparameters (MUST MATCH TRAINING) ---
IMG_SIZE      = 256
MAX_SEQ_LEN   = 2000   # <-- Set to your new max length
LATENT_DIM    = 256 
RNN_UNITS     = 512
NORMALIZE_DIM = 255.0  # For scaling SVG output

# --- Paths ---
BASE_PATH  = 'samples/samples_for_test' 
MODEL_PATH = 'model_checkpoints/best_model_20251101_1109.keras'
OUTPUT_DIR = 'inference_output'
# ---------------------

# --- 2. Data Finder (from train.py) ---
# We need this to find a test image
def get_file_paths(base_path):
    """
    Recursively finds all matching PNG/NPY pairs in all subfolders.
    """
    png_files = []
    npy_files = []
    key_to_png = {}
    key_to_npy = {}

    for root, _, files in os.walk(base_path):
        for f in files:
            if f.endswith('.png'):
                key_id = os.path.splitext(f)[0]
                key_to_png[key_id] = os.path.join(root, f)
            elif f.endswith('.npy'):
                key_id = os.path.splitext(f)[0]
                key_to_npy[key_id] = os.path.join(root, f)

    for key_id in sorted(set(key_to_png.keys()) & set(key_to_npy.keys())):
        png_files.append(key_to_png[key_id])
        npy_files.append(key_to_npy[key_id])

    print(f"Found {len(png_files)} matching file pairs.")
    return png_files, npy_files

# --- 3. Image Preprocessing Helper ---
def load_test_image(png_path):
    """Loads and preprocesses a single PNG for inference."""
    img = tf.io.read_file(png_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return tf.expand_dims(img, axis=0) # Add batch dim

# --- 4. SVG Conversion Helper (Corrected) ---
def sequence_to_svg(sequence, img_size=NORMALIZE_DIM):
    """
    Converts a (dx, dy, p1, p2, p3) sequence back into an SVG file string.
    """
    min_x, min_y = 0, 0
    max_x, max_y = 0, 0
    current_x, current_y = 0, 0
    
    for point in sequence:
        dx, dy = point[0], point[1]
        current_x += dx
        current_y += dy
        min_x, min_y = min(min_x, current_x), min(min_y, current_y)
        max_x, max_y = max(max_x, current_x), max(max_y, current_y)

    width = max_x - min_x
    height = max_y - min_y
    side = max(width, height) + 20 # Add 20 padding
    if side <= 20: 
        side = NORMALIZE_DIM + 20
        min_x = -10
        min_y = -10

    viewBox = f"{min_x - 10} {min_y - 10} {side} {side}"

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewBox}" width="500">',
        '  <rect width="100%" height="100%" fill="white"></rect>',
        '  <path d="'
    ]
    
    current_x, current_y = 0, 0
    is_first_point_in_stroke = True 
    
    for point in sequence:
        dx, dy = point[0], point[1]
        p_end_stroke = point[3] > 0.5  # Check p2
        p_end_drawing = point[4] > 0.5 # Check p3

        current_x += dx
        current_y += dy
        
        if is_first_point_in_stroke:
            svg_lines[-1] += f'M {current_x:.2f} {current_y:.2f}'
            is_first_point_in_stroke = False
        else:
            svg_lines[-1] += f' L {current_x:.2f} {current_y:.2f}'

        if p_end_drawing:
            break 
        elif p_end_stroke:
            if not is_first_point_in_stroke:
                svg_lines[-1] += '" stroke="black" stroke-width="1" fill="none" />'
                svg_lines.append('  <path d="')
            is_first_point_in_stroke = True 

    if svg_lines[-1].strip() == '  <path d="':
        svg_lines.pop()
    elif not svg_lines[-1].endswith('/>'):
        svg_lines[-1] += '" stroke="black" stroke-width="1" fill="none" />'
    
    svg_lines.append('</svg>')
    return '\n'.join(svg_lines)

# --- 5. Custom Loss Function (REQUIRED FOR LOADING) ---
# This must be defined to load the model.
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

# --- 6. Main Inference Function ---
def generate_drawing(model_path, test_png_path):
    print("Loading trained model...")
    # Load the full model to get the weights
    full_model = keras.models.load_model(
        model_path,
        custom_objects={'custom_loss': custom_loss}
    )

    # Extract the encoder
    encoder = full_model.get_layer('encoder')

    # Extract the trained decoder to get its weights
    trained_decoder = full_model.get_layer('decoder')

    # --- Build a new, stateful inference decoder ---
    print("Building inference decoder from trained weights...")

    # 1. Get weights from the trained decoder
    state_projector_weights = trained_decoder.get_layer('state_projector').get_weights()
    lstm_weights = trained_decoder.get_layer('decoder_lstm').get_weights()
    output_layer_weights = trained_decoder.get_layer('decoder_output').get_weights()

    # 2. Define new layers for inference
    inf_encoder_state_in = layers.Input(shape=(LATENT_DIM,), name="inf_encoder_state_in")
    inf_point_input = layers.Input(shape=(1, 5), name="inf_point_input")
    inf_state_h_in = layers.Input(shape=(RNN_UNITS,), name="inf_state_h_in")
    inf_state_c_in = layers.Input(shape=(RNN_UNITS,), name="inf_state_c_in")

    # 3. Re-create the layer graph
    # Model to get initial state
    state_projector_layer = layers.Dense(RNN_UNITS, name="inf_state_projector")
    initial_h = state_projector_layer(inf_encoder_state_in)
    initial_c = state_projector_layer(inf_encoder_state_in)
    initial_state_model = keras.Model(inf_encoder_state_in, [initial_h, initial_c])

    # Model for the decoder step
    lstm_layer = layers.LSTM(RNN_UNITS, return_sequences=True, return_state=True, name="inf_lstm")
    lstm_output, state_h_out, state_c_out = lstm_layer(
        inf_point_input,
        initial_state=[inf_state_h_in, inf_state_c_in]
    )
    output_layer = layers.Dense(5, activation='linear', name="inf_output_layer")
    output_point = output_layer(lstm_output)
    decoder_step_model = keras.Model(
        [inf_point_input, inf_state_h_in, inf_state_c_in],
        [output_point, state_h_out, state_c_out]
    )

    # 4. Set the weights on the new layers
    state_projector_layer.set_weights(state_projector_weights)
    lstm_layer.set_weights(lstm_weights)
    output_layer.set_weights(output_layer_weights)

    print("Inference model built.")

    # --- Now we can run inference ---
    
    test_image = load_test_image(test_png_path)
    print(f"Loaded test image: {test_png_path}")

    # Run the Encoder
    encoder_output = encoder(test_image)

    # Run the Decoder Autoregressively
    hidden_state, cell_state = initial_state_model(encoder_output)
    current_point = tf.zeros((1, 1, 5)) # Start with a "START" token
    generated_sequence = []

    print("Generating sequence (point by point)...")
    for t in range(MAX_SEQ_LEN):
        predicted_point, hidden_state, cell_state = decoder_step_model(
            [current_point, hidden_state, cell_state]
        )
        
        point_squeezed = tf.squeeze(predicted_point, axis=[0, 1])
        generated_sequence.append(point_squeezed.numpy())
        
        current_point = predicted_point # Feed the output back as the next input

        # Check for the "End of Drawing" token (p3)
        end_drawing_prob = tf.sigmoid(predicted_point[0, 0, 4]).numpy()
        if end_drawing_prob > 0.5: 
            print(f"Model predicted <END> token at step {t}. Stopping.")
            break

    print("Sequence generation finished.")
    return np.array(generated_sequence)

# --- 7. Run Everything ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all possible test images
    png_files, _ = get_file_paths(BASE_PATH)
    
    if not png_files:
        print(f"Error: No PNG files found in {BASE_PATH}. Cannot run inference.")
    else:
        # Pick one random image to test
        test_img_path = random.choice(png_files)
        test_img_name = os.path.basename(test_img_path)
        base_name = os.path.splitext(test_img_name)[0]
        
        # Generate the sequence
        predicted_sequence = generate_drawing(MODEL_PATH, test_img_path)

        # Convert sequence to SVG
        svg_string = sequence_to_svg(predicted_sequence)

        # Save the SVG to your new path
        output_svg_path = os.path.join(OUTPUT_DIR, f"predicted_{base_name}.svg")
        with open(output_svg_path, 'w') as f:
            f.write(svg_string)

        print(f"\n--- Success! ---")
        print(f"Predicted drawing saved to: {output_svg_path}")

        # Display results (for Jupyter)
        try:
            print("\n--- Displaying Results (for Jupyter Notebook) ---")
            print("Original PNG:")
            plt.imshow(plt.imread(test_img_path), cmap='gray')
            plt.show()
            
            print("Generated SVG:")
            display(SVG(data=svg_string))
        except Exception as e:
            print(f"\nCould not display images: {e}")
            print("To view results, open the saved PNG and SVG files.")