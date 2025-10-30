import json
import os
import numpy as np
import cairosvg
import time

# --- Configuration ---
# 1. The input file you're processing
NDJSON_FILE = 'samples/quick_draw_sample/ambulance_10000.ndjson'

# 2. Output directory for PNGs (model's X_train)
PNG_OUTPUT_DIR = 'samples/quick_draw_dataset/ambulance_png'
# 3. Output directory for NPYs (model's Y_train)
NPY_OUTPUT_DIR = 'samples/quick_draw_dataset/ambulance_processed'

# 4. PNG settings
PNG_SIZE = 256  # 256x256 is a standard size for CNN inputs
STROKE_WIDTH = 1
PADDING = 10    # Padding for the SVG viewBox

# 5. NPY settings
NORMALIZE_DIM = 255.0
# ---------------------

#####################################################################
# SVG & PNG GENERATION FUNCTIONS
#####################################################################

def create_svg_path(stroke, stroke_width):
    """Converts a single stroke list into an SVG path string."""
    x_coords, y_coords = stroke[0], stroke[1]
    
    path_data = f"M {x_coords[0]} {y_coords[0]}"
    for i in range(1, len(x_coords)):
        path_data += f" L {x_coords[i]} {y_coords[i]}"
        
    return (
        f'  <path d="{path_data}" '
        f'stroke="black" stroke-width="{stroke_width}" fill="none" '
        f'stroke-linecap="round" stroke-linejoin="round" />'
    )

def ndjson_to_svg_data(strokes, word, padding, stroke_width):
    """
    Converts ndjson strokes into a string of SVG data.
    This version creates a square viewBox and NO background.
    """
    all_x, all_y = [], []
    svg_paths = []

    for stroke in strokes:
        all_x.extend(stroke[0])
        all_y.extend(stroke[1])
        svg_paths.append(create_svg_path(stroke, stroke_width))

    # --- Calculate a SQUARE Bounding Box ---
    orig_min_x = min(all_x)
    orig_min_y = min(all_y)
    orig_max_x = max(all_x)
    orig_max_y = max(all_y)
    
    orig_width = orig_max_x - orig_min_x
    orig_height = orig_max_y - orig_min_y
    
    center_x = orig_min_x + orig_width / 2
    center_y = orig_min_y + orig_height / 2
    
    side_length = max(orig_width, orig_height) + (padding * 2)
    
    viewBox_min_x = center_x - (side_length / 2)
    viewBox_min_y = center_y - (side_length / 2)
    
    viewBox = f"{viewBox_min_x} {viewBox_min_y} {side_length} {side_length}"
    # --- End Bounding Box ---

    # Assemble the SVG string (NO background <rect> for transparency)
    svg_content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{viewBox}" width="{PNG_SIZE}" height="{PNG_SIZE}">\n'
        f'  <title>{word}</title>\n'
        + '\n'.join(svg_paths)
        + '\n</svg>'
    )
    
    return svg_content

#####################################################################
# NPY (SEQUENCE) GENERATION FUNCTIONS
#####################################################################

def normalize_drawing(strokes, normalize_dim):
    """
    Normalizes a drawing to fit within a [0, normalize_dim] box.
    """
    all_x, all_y = [], []
    for stroke in strokes:
        all_x.extend(stroke[0])
        all_y.extend(stroke[1])

    min_x, min_y = min(all_x), min(all_y)
    max_x, max_y = max(all_x), max(all_y)

    max_dim = max(max_x - min_x, max_y - min_y)
    if max_dim == 0:
        max_dim = 1e-5 

    normalized_strokes = []
    for stroke in strokes:
        x_coords = [(x - min_x) * normalize_dim / max_dim for x in stroke[0]]
        y_coords = [(y - min_y) * normalize_dim / max_dim for y in stroke[1]]
        normalized_strokes.append([x_coords, y_coords])
        
    return normalized_strokes

def convert_to_sequence(strokes):
    """
    Converts normalized strokes into the (dx, dy, p1, p2, p3) sequence.
    """
    sequence = []
    last_x, last_y = 0.0, 0.0
    num_strokes = len(strokes)
    
    for i, (x_coords, y_coords) in enumerate(strokes):
        for j, (x, y) in enumerate(zip(x_coords, y_coords)):
            
            is_end_stroke = (j == len(x_coords) - 1)
            is_end_drawing = is_end_stroke and (i == num_strokes - 1)
            
            p1 = 1.0  # pen_down
            p2 = 1.0 if is_end_stroke else 0.0
            p3 = 1.0 if is_end_drawing else 0.0
            
            dx = x - last_x
            dy = y - last_y
            
            sequence.append([dx, dy, p1, p2, p3])
            
            last_x, last_y = x, y

    return np.array(sequence, dtype=np.float32)

#####################################################################
# MAIN SCRIPT EXECUTION
#####################################################################

def main():
    if not os.path.exists(NDJSON_FILE):
        print(f"Error: Input file not found at '{NDJSON_FILE}'")
        return

    # Create both output directories
    os.makedirs(PNG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(NPY_OUTPUT_DIR, exist_ok=True)
    
    print(f"Input file: '{NDJSON_FILE}'")
    print(f"PNG output: '{PNG_OUTPUT_DIR}'")
    print(f"NPY output: '{NPY_OUTPUT_DIR}'")
    print("Starting paired data generation...")
    
    start_time = time.time()
    line_number = 0
    success_count = 0
    error_count = 0

    with open(NDJSON_FILE, 'r') as f:
        for line in f:
            line_number += 1
            
            try:
                # 1. Load the line
                data = json.loads(line)
                key_id = data['key_id']
                word = data['word']
                strokes = data['drawing'] # Raw strokes
                
                # Define file paths
                png_path = os.path.join(PNG_OUTPUT_DIR, f"{key_id}.png")
                npy_path = os.path.join(NPY_OUTPUT_DIR, f"{key_id}.npy")

                # --- Path A: Generate PNG ---
                svg_data_string = ndjson_to_svg_data(strokes, word, PADDING, STROKE_WIDTH)
                
                # Convert SVG string to PNG
                cairosvg.svg2png(
                    bytestring=svg_data_string.encode('utf-8'),
                    write_to=png_path,
                    output_width=PNG_SIZE,
                    output_height=PNG_SIZE
                )
                
                # --- Path B: Generate NPY ---
                # 2. Normalize strokes for NPY
                normalized_strokes = normalize_drawing(strokes, NORMALIZE_DIM)
                
                # 3. Convert to (dx, dy, p) sequence
                sequence_data = convert_to_sequence(normalized_strokes)
                
                # 4. Save as .npy file
                np.save(npy_path, sequence_data)
                
                success_count += 1
                
                if line_number % 500 == 0:
                    print(f"  ...Processed {line_number} drawings...")

            except Exception as e:
                print(f"Error processing line {line_number} (key: {data.get('key_id', 'N/A')}): {e}")
                error_count += 1

    end_time = time.time()
    print("\n--- Paired Data Generation Complete ---")
    print(f"Successfully processed: {success_count} drawings")
    print(f"Failed/Skipped:       {error_count} drawings")
    print(f"Total time:           {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()