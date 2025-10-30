import json
import os
import cairosvg
import time

# --- Configuration ---
# 1. The full input file
NDJSON_FILE = '/media/holidayj/Documents/data/quick_draw_dataset/raw/airplane.ndjson'
# 2. The new directory to create and save files in
OUTPUT_DIR = '/media/holidayj/Documents/data/quick_draw_dataset/airplane'

# 3. Drawing style configuration
STROKE_WIDTH = 1  # Kept the thin line preference
PADDING = 20
PNG_WIDTH = 1000
PNG_HEIGHT = 1000
# ---------------------


def create_svg_path(stroke):
    """
    Converts a single stroke list [ [x_coords], [y_coords] ]
    into an SVG path string.
    """
    x_coords = stroke[0]
    y_coords = stroke[1]
    
    path_data = f"M {x_coords[0]} {y_coords[0]}"
    for i in range(1, len(x_coords)):
        path_data += f" L {x_coords[i]} {y_coords[i]}"

    return (
        f'  <path d="{path_data}" '
        f'stroke="black" stroke-width="{STROKE_WIDTH}" fill="none" '
        f'stroke-linecap="round" stroke-linejoin="round" />'
    )


def ndjson_to_svg(ndjson_line):
    """
    Converts a full ndjson line (as a string) into SVG content
    and returns the SVG, word, and key_id.
    """
    try:
        data = json.loads(ndjson_line)
        key_id = data['key_id'] # This is the unique ID
        word = data.get('word', 'drawing')
        strokes = data['drawing']

        all_x = []
        all_y = []
        svg_paths = []

        for stroke in strokes:
            all_x.extend(stroke[0])
            all_y.extend(stroke[1])
            svg_paths.append(create_svg_path(stroke))

        # --- Calculate a SQUARE Bounding Box (viewBox) ---
        orig_min_x = min(all_x)
        orig_min_y = min(all_y)
        orig_max_x = max(all_x)
        orig_max_y = max(all_y)
        
        orig_width = orig_max_x - orig_min_x
        orig_height = orig_max_y - orig_min_y
        
        center_x = orig_min_x + orig_width / 2
        center_y = orig_min_y + orig_height / 2
        
        side_length = max(orig_width, orig_height) + (PADDING * 2)
        
        viewBox_min_x = center_x - (side_length / 2)
        viewBox_min_y = center_y - (side_length / 2)
        
        viewBox = f"{viewBox_min_x} {viewBox_min_y} {side_length} {side_length}"
        # --- End of Bounding Box logic ---

        svg_content = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{viewBox}" width="{PNG_WIDTH}">\n'
            f'  <title>{word}</title>\n'
            + '\n'.join(svg_paths)
            + '\n</svg>'
        )
        
        # Return the SVG data and the unique key_id for the filename
        return svg_content, key_id

    except Exception as e:
        # Return None if the line is corrupt
        return None, None


# --- Main script execution ---
def main():
    
    # 1. Check if source file exists
    if not os.path.exists(NDJSON_FILE):
        print(f"Error: Input file not found at '{NDJSON_FILE}'")
        print("Please check the path and filename (e.g., 'airplabe' typo?).")
        return

    # 2. Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory ensured: '{OUTPUT_DIR}'")

    print(f"Starting conversion of '{NDJSON_FILE}'...")
    start_time = time.time()
    
    # Counters for progress
    line_number = 0
    success_count = 0
    error_count = 0

    # 3. Open and loop through the entire file
    with open(NDJSON_FILE, 'r') as f:
        for line in f:
            line_number += 1
            
            # --- Process the line ---
            svg_data, key_id = ndjson_to_svg(line)
            
            # Skip if the line was corrupt
            if not svg_data:
                print(f"Skipping corrupt line: {line_number}")
                error_count += 1
                continue

            # --- Define output paths ---
            # We use the key_id as the filename for uniqueness
            svg_path = os.path.join(OUTPUT_DIR, f"{key_id}.svg")
            png_path = os.path.join(OUTPUT_DIR, f"{key_id}.png")

            try:
                # --- 4. Write the SVG file ---
                with open(svg_path, 'w') as svg_f:
                    svg_f.write(svg_data)

                # --- 5. Convert SVG to 1000x1000 PNG ---
                cairosvg.svg2png(
                    url=svg_path,
                    write_to=png_path,
                    output_width=PNG_WIDTH,
                    output_height=PNG_HEIGHT
                )
                success_count += 1

                # --- 6. Print progress periodically ---
                if line_number % 1000 == 0:
                    print(f"  ...Processed {line_number} drawings...")

            except Exception as e:
                print(f"Error processing drawing {key_id} on line {line_number}: {e}")
                error_count += 1

    # --- 7. Final Report ---
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n--- Conversion Complete ---")
    print(f"Successfully converted: {success_count} drawings")
    print(f"Failed/Skipped:       {error_count} drawings")
    print(f"Total time:           {total_time:.2f} seconds")

if __name__ == "__main__":
    main()                                                                      