import json
import os
import numpy as np
import cairosvg
import time

# --- Configuration ---
# 1. Directory containing your .ndjson files
NDJSON_INPUT_DIR = '/mnt/Documents/Dad/dataset/quick_draw/raw'

# 2. Base directory for all processed output
BASE_OUTPUT_DIR = 'samples/quick_draw_dataset'

# 3. List of .ndjson files to process
# Add the filenames of all categories you want to process here.
FILES_TO_PROCESS = [
    'The Eiffel Tower.ndjson',
    'cooler.ndjson',
    'calendar.ndjson',
    'firetruck.ndjson',
    'harp.ndjson']

# 4. PNG settings
PNG_SIZE     = 256  # 256x256 is a standard size for CNN inputs
STROKE_WIDTH = 1
PADDING      = 10   # Padding for the SVG viewBox

# 5. NPY settings
NORMALIZE_DIM = 255.0
# ---------------------

#####################################################################
# SVG & PNG GENERATION FUNCTIONS (Unchanged)
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
    svg_paths    = []

    for stroke in strokes:
        all_x.extend(stroke[0])
        all_y.extend(stroke[1])
        svg_paths.append(create_svg_path(stroke, stroke_width))

    # --- Calculate a SQUARE Bounding Box ---
    orig_min_x = min(all_x)
    orig_min_y = min(all_y)
    orig_max_x = max(all_x)
    orig_max_y = max(all_y)
    
    orig_width  = orig_max_x - orig_min_x
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
        '''
        Taking only x and y coordinates. 
        but doesn't take sequence which is stroke[2].
        '''
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
    sequence       = []
    last_x, last_y = 0.0, 0.0
    num_strokes    = len(strokes)
    
    for i, (x_coords, y_coords) in enumerate(strokes):
        for j, (x, y) in enumerate(zip(x_coords, y_coords)):
            
            is_end_stroke  = (j == len(x_coords) - 1)
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
# MAIN SCRIPT EXECUTION (Refactored for multiple files)
#####################################################################
def main():
    total_start_time    = time.time()
    total_success_count = 0
    total_error_count   = 0
    
    print(f"Starting paired data generation for {len(FILES_TO_PROCESS)} file(s)...")
    print(f"Base Output Directory: '{BASE_OUTPUT_DIR}'")
    
    for file_name in FILES_TO_PROCESS:
        ndjson_file_path = os.path.join(NDJSON_INPUT_DIR, file_name)
        
        print(f"\n--- Processing file: '{file_name}' ---")

        if not os.path.exists(ndjson_file_path):
            print(f"  Error: Input file not found at '{ndjson_file_path}'")
            print("  Skipping this file.")
            continue

        # --- Generate dynamic output paths ---
        # "The Eiffel Tower.ndjson" -> "the_eiffel_tower"
        class_name_raw       = os.path.splitext(file_name)[0]
        class_name_formatted = class_name_raw.replace(' ', '_').lower()
        
        png_output_dir = os.path.join(BASE_OUTPUT_DIR, f"{class_name_formatted}_png")
        npy_output_dir = os.path.join(BASE_OUTPUT_DIR, f"{class_name_formatted}_processed")
        
        # Create both output directories
        os.makedirs(png_output_dir, exist_ok=True)
        os.makedirs(npy_output_dir, exist_ok=True)
        
        print(f"  PNG output: '{png_output_dir}'")
        print(f"  NPY output: '{npy_output_dir}'")
        
        file_start_time    = time.time()
        file_line_number   = 0
        file_success_count = 0
        file_error_count   = 0

        with open(ndjson_file_path, 'r') as f:
            for line in f:
                file_line_number += 1
                data = {} # Initialize data dict to access key_id in case of error
                
                try:
                    # 1. Load the line
                    data    = json.loads(line)
                    key_id  = data['key_id']
                    word    = data['word']
                    strokes = data['drawing'] # Raw strokes
                    
                    # Define file paths
                    png_path = os.path.join(png_output_dir, f"{key_id}.png")
                    npy_path = os.path.join(npy_output_dir, f"{key_id}.npy")

                    # --- Path A: Generate PNG ---
                    svg_data_string = ndjson_to_svg_data(strokes, word, PADDING, STROKE_WIDTH)
                    
                    # Convert SVG string to PNG
                    cairosvg.svg2png(
                        bytestring    = svg_data_string.encode('utf-8'),
                        write_to      = png_path,
                        output_width  = PNG_SIZE,
                        output_height = PNG_SIZE
                    )
                    
                    # --- Path B: Generate NPY ---
                    # 2. Normalize strokes for NPY
                    normalized_strokes = normalize_drawing(strokes, NORMALIZE_DIM)
                    
                    # 3. Convert to (dx, dy, p) sequence
                    sequence_data = convert_to_sequence(normalized_strokes)
                    
                    # 4. Save as .npy file
                    np.save(npy_path, sequence_data)
                    
                    file_success_count += 1
                    
                    if file_line_number % 1000 == 0:
                        print(f"    ...Processed {file_line_number} drawings...")

                except Exception as e:
                    print(f"  Error processing line {file_line_number} (key: {data.get('key_id', 'N/A')}): {e}")
                    file_error_count += 1

        file_end_time = time.time()
        print(f"  Finished processing '{file_name}' in {file_end_time - file_start_time:.2f}s")
        print(f"  Successfully processed: {file_success_count} drawings")
        print(f"  Failed/Skipped:       {file_error_count} drawings")
        
        total_success_count += file_success_count
        total_error_count   += file_error_count

    total_end_time = time.time()
    print("\n--- Paired Data Generation Complete ---")
    print(f"Total drawings processed: {total_success_count}")
    print(f"Total errors:             {total_error_count}")
    print(f"Total time:               {total_end_time - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()