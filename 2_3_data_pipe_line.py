import tensorflow as tf
import os
import numpy as np

# --- Configuration ---
# 1. Point to your two data folders
BASE_PATH = '/content/sample_data' 
PNG_DIR   = os.path.join(BASE_PATH, 'ambulance_png')
NPY_DIR   = os.path.join(BASE_PATH, 'ambulance_processed')

# 2. Set model hyperparameters
BATCH_SIZE = 32
IMG_SIZE = 256  # The 256x256 size we created
MAX_SEQ_LEN = 250 # Max points in a drawing. We'll pad to this.
# ---------------------

def get_file_paths(png_dir, npy_dir):
    """
    Finds all matching PNG/NPY pairs and returns two sorted lists.
    """
    png_files = []
    npy_files = []
    
    # We use the key_id (filename without extension) to match
    key_ids = [f.split('.')[0] for f in os.listdir(png_dir) if f.endswith('.png')]
    
    for key_id in key_ids:
        png_path = os.path.join(png_dir, f"{key_id}.png")
        npy_path = os.path.join(npy_dir, f"{key_id}.npy")
        
        # Ensure both files actually exist
        if os.path.exists(png_path) and os.path.exists(npy_path):
            png_files.append(png_path)
            npy_files.append(npy_path)
            
    print(f"Found {len(png_files)} matching file pairs.")
    return png_files, npy_files

def load_npy_wrapper(npy_path_tensor):
    """
    Wraps np.load to handle the EagerTensor from tf.py_function.
    """
    # 1. Get the raw bytes from the EagerTensor
    npy_path_bytes = npy_path_tensor.numpy()
    
    # 2. Decode the bytes into a standard Python string
    npy_path_str = npy_path_bytes.decode('utf-8')
    
    # 3. Now, call np.load with the clean string path
    sequence = np.load(npy_path_str)
    
    # 4. Cast to float32 to match the output signature
    return sequence.astype(np.float32)

def load_and_preprocess(png_path, npy_path):
    """
    The main parsing function for tf.data.
    Loads and preprocesses one (PNG, NPY) pair.
    """
    
    # --- 1. Load and process the PNG (Input / X) ---
    # Read the file
    img = tf.io.read_file(png_path)
    # Decode the PNG to a tensor
    img = tf.image.decode_png(img, channels=1) # 1 channel (grayscale)
    # Resize (just in case, but they should be 256)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    # Normalize pixel values from [0, 255] to [-1, 1]
    # This is a common practice that helps models train better.
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    
    # --- 2. Load and process the NPY (Target / Y) ---
    # We must use tf.py_function to load with numpy
    # [sequence] = tf.py_function(np.load, [npy_path], [tf.float32])
    [sequence] = tf.py_function(load_npy_wrapper, [npy_path], [tf.float32])
    
    # --- 3. Pad the sequence ---
    seq_len = tf.shape(sequence)[0]
    # Pad with zeros at the end
    # [[0, 0], [MAX_SEQ_LEN - seq_len, 0], [0, 0]]
    # This pads the first dimension (time) but not the second (features)
    padding = [[0, MAX_SEQ_LEN - seq_len], [0, 0]]
    sequence = tf.pad(sequence, padding, 'CONSTANT')
    
    # Set the final shape, which is now known
    sequence.set_shape([MAX_SEQ_LEN, 5])
    
    # The model's target will be the sequence "shifted by one"
    # Target (Y): The sequence to be predicted
    # Input (X_decoder): The same sequence, one step behind
    
    # Decoder Input: (e.g., [START, p1, p2, p3, ...])
    decoder_input = sequence
    # Target Output: (e.g., [p1, p2, p3, END, ...])
    decoder_target = tf.roll(sequence, shift=-1, axis=0)

    # Our model needs two inputs (image, decoder_input) and one output (target)
    return (img, decoder_input), decoder_target

# --- Main execution ---
def create_dataset(png_dir, npy_dir, batch_size):
    """Builds the final tf.data.Dataset object."""
    
    # 1. Get the lists of file paths
    png_files, npy_files = get_file_paths(png_dir, npy_dir)
    if not png_files:
        print("No files found. Exiting.")
        return None

    # 2. Create a dataset of the file paths
    path_ds = tf.data.Dataset.from_tensor_slices((png_files, npy_files))
    
    # 3. Use .map() to load and preprocess the data
    #    num_parallel_calls=tf.data.AUTOTUNE lets TF load in parallel
    dataset = path_ds.map(load_and_preprocess, 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    # 4. Shuffle, batch, and prefetch for performance
    dataset = (
        dataset
        .shuffle(buffer_size=1024)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    return dataset

# --- Run it ---
if __name__ == "__main__":
    train_dataset = create_dataset(PNG_DIR, NPY_DIR, BATCH_SIZE)
    
    if train_dataset:
        print("\n--- Dataset Created Successfully ---")
        # Let's inspect one batch to confirm shapes
        for (img_batch, seq_in_batch), seq_out_batch in train_dataset.take(1):
            print(f"Image Batch Shape (X_img):   {img_batch.shape}")
            print(f"Sequence Input Shape (X_dec): {seq_in_batch.shape}")
            print(f"Sequence Target Shape (Y):   {seq_out_batch.shape}")
            
        print("\nData pipeline is ready.")