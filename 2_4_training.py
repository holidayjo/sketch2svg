import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- 0. Re-import dataset code (if in a new cell) ---
# Make sure these variables are defined before building the model
# BATCH_SIZE = 32
# IMG_SIZE = 256
# MAX_SEQ_LEN = 250
# train_dataset = create_dataset(PNG_DIR, NPY_DIR, BATCH_SIZE)


# --- 1. Define Model Hyperparameters ---
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
SEQ_SHAPE = (MAX_SEQ_LEN, 5) # 5 features: dx, dy, p1, p2, p3

# Latent dim is the "thought vector" size
LATENT_DIM = 256 
# RNN units for the decoder
RNN_UNITS = 512 


# --- 2. Build the Encoder (The "Eye") ---
# This turns the (256, 256, 1) image into a (LATENT_DIM) vector

def build_encoder():
    # Input for the PNG image
    image_input = layers.Input(shape=IMG_SHAPE, name="image_input")
    
    # We'll use a simple CNN. A pre-trained MobileNet/ResNet is also a great choice.
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(image_input)
    x = layers.BatchNormalization()(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.BatchNormalization()(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.BatchNormalization()(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.BatchNormalization()(x)
    
    # Flatten the feature map and get the final "thought vector"
    x = layers.Flatten()(x)
    x = layers.Dense(LATENT_DIM, activation='relu', name="encoder_output")(x)
    
    return keras.Model(image_input, x, name="encoder")


# --- 3. Build the Decoder (The "Hand") ---
# This takes the "thought vector" and draws the sequence (MAX_SEQ_LEN, 5)

def build_decoder():
    # Input for the sequence data (e.g., [START, p1, p2, ...])
    sequence_input = layers.Input(shape=SEQ_SHAPE, name="sequence_input")
    
    # Input for the "thought vector" from the Encoder
    encoder_state = layers.Input(shape=(LATENT_DIM,), name="encoder_state")

    # Use the encoder state as the initial hidden state for the RNN
    # We need two states (h and c) for an LSTM
    decoder_lstm = layers.LSTM(RNN_UNITS, return_sequences=True, name="decoder_lstm")
    
    # The initial_state expects a list: [hidden_state, cell_state]
    # We'll just re-use the encoder's output for both.
    decoder_outputs = decoder_lstm(sequence_input, initial_state=[encoder_state, encoder_state])
    
    # Final output layer to predict the 5 features
    # (dx, dy, p1, p2, p3)
    # We use 'linear' activation because dx/dy are not bounded
    output = layers.Dense(5, activation='linear', name="decoder_output")(decoder_outputs)
    
    return keras.Model([sequence_input, encoder_state], output, name="decoder")


# --- 4. Define the Custom Loss Function ---
# We must treat (dx, dy) and (p1, p2, p3) differently
def custom_loss(y_true, y_pred):
    # Split the 5-element vector
    # True values
    true_dx_dy = y_true[:, :, 0:2]  # (batch, seq_len, 2)
    true_p = y_true[:, :, 2:5]      # (batch, seq_len, 3)
    
    # Predicted values
    pred_dx_dy = y_pred[:, :, 0:2]
    pred_p = y_pred[:, :, 2:5]
    
    # Loss 1: Mean Squared Error for (dx, dy)
    # This is a regression task
    loss_dx_dy = tf.keras.losses.mean_squared_error(true_dx_dy, pred_dx_dy)
    
    # Loss 2: Cross-Entropy for (p1, p2, p3)
    # This is a classification task
    # We apply softmax to the predictions to turn them into probabilities
    loss_p = tf.keras.losses.categorical_crossentropy(true_p, pred_p, from_logits=True)
    
    # We need to create a mask to ignore padded steps (where all 5 inputs are 0)
    # tf.reduce_any checks if any value in the 5-vector is non-zero
    mask = tf.cast(tf.reduce_any(y_true != 0.0, axis=-1), tf.float32)
    
    # Apply the mask to both losses
    loss_dx_dy = loss_dx_dy * mask
    loss_p = loss_p * mask
    
    # Return the average loss per time-step
    total_loss = loss_dx_dy + loss_p
    return tf.reduce_mean(total_loss)


# --- 5. Combine and Compile the Full Model ---
def build_and_compile_model():
    # Build the two sub-models
    encoder = build_encoder()
    decoder = build_decoder()

    # Define the two inputs for the full model
    image_input = layers.Input(shape=IMG_SHAPE, name="image_input")
    sequence_input = layers.Input(shape=SEQ_SHAPE, name="sequence_input")
    
    # Connect the graph
    encoder_output = encoder(image_input)
    decoder_output = decoder([sequence_input, encoder_output])
    
    # Create the final model
    model = keras.Model(inputs=[image_input, sequence_input], 
                        outputs=decoder_output, 
                        name="png_to_svg_model")
    
    # Compile the model with our custom loss and an optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=custom_loss
    )
    
    return model

# --- Run it ---
if __name__ == "__main__":
    model = build_and_compile_model()
    
    # Print a summary of the model
    model.summary()
    
    print("\nModel is built, compiled, and ready for training.")