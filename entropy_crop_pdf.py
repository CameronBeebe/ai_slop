import fitz
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import argparse
import os

def calculate_entropy(arr):
    """Calculate the Shannon entropy of a 1D array of pixel values."""
    # Ensure array is flattened and handle empty arrays
    arr = np.asarray(arr).ravel()
    if len(arr) == 0:
        return 0
        
    # Use bincount for integer values (faster than histogram)
    counts = np.bincount(arr, minlength=256)
    # Convert to probabilities
    probs = counts / len(arr)
    # Remove zero probabilities and calculate entropy
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def calculate_scanning_entropy(arr1, arr2):
    """
    Calculate the entropy of transitions between two arrays.
    Uses a simplified approach for better performance.
    """
    # Calculate absolute differences for faster computation
    differences = np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))
    
    # Normalize differences to 0-255 range
    if np.max(differences) > 0:
        differences = np.round((differences * 255 / np.max(differences))).astype(np.uint8)
    else:
        differences = np.zeros_like(differences, dtype=np.uint8)
    
    # Calculate entropy of differences
    counts = np.bincount(differences.ravel(), minlength=256)
    probs = counts / len(differences.ravel())
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0

def analyze_page_entropy(gray_img):
    """
    Analyze entropy patterns using single vertical and horizontal scans.
    Returns entropy profiles for both directions.
    """
    height, width = gray_img.shape
    
    # Calculate vertical scan (top to bottom)
    vertical_entropy = np.zeros(height)
    vertical_scanning = np.zeros(height)
    chunk_size = 50
    for i in range(0, height, chunk_size):
        end = min(i + chunk_size, height)
        chunk = gray_img[i:end, :]
        for j in range(end - i):
            vertical_entropy[i + j] = calculate_entropy(chunk[j])
            if i + j < height - 1:
                vertical_scanning[i + j] = calculate_scanning_entropy(
                    gray_img[i + j, :], 
                    gray_img[i + j + 1, :]
                )
    vertical_scanning[-1] = vertical_scanning[-2]  # Copy last value
    
    # Calculate horizontal scan (left to right)
    horizontal_entropy = np.zeros(width)
    horizontal_scanning = np.zeros(width)
    for i in range(0, width, chunk_size):
        end = min(i + chunk_size, width)
        chunk = gray_img[:, i:end]
        for j in range(end - i):
            horizontal_entropy[i + j] = calculate_entropy(chunk[:, j])
            if i + j < width - 1:
                horizontal_scanning[i + j] = calculate_scanning_entropy(
                    gray_img[:, i + j],
                    gray_img[:, i + j + 1]
                )
    horizontal_scanning[-1] = horizontal_scanning[-2]  # Copy last value
    
    # Calculate gradients
    vertical_gradient = np.gradient(vertical_entropy)
    horizontal_gradient = np.gradient(horizontal_entropy)

    # --- New: Calculate trailing average of scanning entropy and its gradient ---
    # Use a window that is 2% of the dimension, with a minimum of 10 pixels
    vert_ma_window = max(10, int(height * 0.02))
    horz_ma_window = max(10, int(width * 0.02))

    def moving_average(data, window_size):
        """Calculates moving average using convolution."""
        return np.convolve(data, np.ones(window_size), 'same') / window_size

    vertical_scanning_avg = moving_average(vertical_scanning, vert_ma_window)
    horizontal_scanning_avg = moving_average(horizontal_scanning, horz_ma_window)

    vertical_scanning_avg_gradient = np.gradient(vertical_scanning_avg)
    horizontal_scanning_avg_gradient = np.gradient(horizontal_scanning_avg)
    
    # Normalize all measures to [0, 1] range for plotting
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)
    
    return {
        'vertical': {
            'entropy': normalize(vertical_entropy),
            'scanning': normalize(vertical_scanning),
            'gradient': vertical_gradient,  # Return raw gradient
            'gradient_normalized': normalize(vertical_gradient), # For plotting
            'scanning_avg': normalize(vertical_scanning_avg),
            'scanning_avg_gradient': vertical_scanning_avg_gradient,
            'scanning_avg_gradient_normalized': normalize(vertical_scanning_avg_gradient),
        },
        'horizontal': {
            'entropy': normalize(horizontal_entropy),
            'scanning': normalize(horizontal_scanning),
            'gradient': horizontal_gradient, # Return raw gradient
            'gradient_normalized': normalize(horizontal_gradient), # For plotting
            'scanning_avg': normalize(horizontal_scanning_avg),
            'scanning_avg_gradient': horizontal_scanning_avg_gradient,
            'scanning_avg_gradient_normalized': normalize(horizontal_scanning_avg_gradient),
        }
    }

def find_crop_boundaries(entropy_data, gradient_threshold=0.05, conservative_gradient_threshold=0.005):
    """
    Determine crop boundaries by finding the largest peaks in the entropy gradient.
    This version uses the raw gradient to distinguish between positive (margin-to-content)
    and negative (content-to-margin) transitions.
    It also finds a second, more conservative boundary if the primary crop is large.
    """
    boundaries = {}
    conservative_boundaries = {}

    # --- Vertical Scan (Top and Bottom) ---
    vertical_gradient = entropy_data['vertical']['gradient']
    height = len(vertical_gradient)
    
    # Find Top Boundary: Search for the max positive gradient in the top half
    top_search_area = vertical_gradient[:height//2]
    top_peak_index = np.argmax(top_search_area)
    top_peak_value = top_search_area[top_peak_index]
    
    if top_peak_value > gradient_threshold:
        boundaries['top'] = top_peak_index + 1
        print(f"Found top boundary at {top_peak_index}, adjusting inward to {boundaries['top']} (gradient: {top_peak_value:.3f})")
    else:
        boundaries['top'] = 1
        print(f"No significant top boundary (max grad: {top_peak_value:.3f}). Defaulting to 1.")

    # Find Bottom Boundary: Search for the min negative gradient in the bottom half
    bottom_search_area = vertical_gradient[height//2:]
    bottom_peak_offset = np.argmin(bottom_search_area)
    bottom_peak_index = bottom_peak_offset + height // 2
    bottom_peak_value = bottom_search_area[bottom_peak_offset]
    
    if bottom_peak_value < -gradient_threshold:
        boundaries['bottom'] = bottom_peak_index - 1
        print(f"Found bottom boundary at {bottom_peak_index}, adjusting inward to {boundaries['bottom']} (gradient: {bottom_peak_value:.3f})")
    else:
        boundaries['bottom'] = height - 1
        print(f"No significant bottom boundary (min grad: {bottom_peak_value:.3f}). Defaulting to {height - 1}.")

    # --- Horizontal Scan (Left and Right) ---
    horizontal_gradient = entropy_data['horizontal']['gradient']
    width = len(horizontal_gradient)

    # Find Left Boundary: Search for the max positive gradient in the left half
    left_search_area = horizontal_gradient[:width//2]
    left_peak_index = np.argmax(left_search_area)
    left_peak_value = left_search_area[left_peak_index]

    if left_peak_value > gradient_threshold:
        boundaries['left'] = left_peak_index + 1
        print(f"Found left boundary at {left_peak_index}, adjusting inward to {boundaries['left']} (gradient: {left_peak_value:.3f})")
    else:
        boundaries['left'] = 1
        print(f"No significant left boundary (max grad: {left_peak_value:.3f}). Defaulting to 1.")

    # Find Right Boundary: Search for the min negative gradient in the right half
    right_search_area = horizontal_gradient[width//2:]
    right_peak_offset = np.argmin(right_search_area)
    right_peak_index = right_peak_offset + width // 2
    right_peak_value = right_search_area[right_peak_offset]
    
    if right_peak_value < -gradient_threshold:
        boundaries['right'] = right_peak_index - 1
        print(f"Found right boundary at {right_peak_index}, adjusting inward to {boundaries['right']} (gradient: {right_peak_value:.3f})")
    else:
        boundaries['right'] = width - 1
        print(f"No significant right boundary (min grad: {right_peak_value:.3f}). Defaulting to {width - 1}.")

    # --- Conservative Boundary Search ---
    # If the primary crop for a side is > 5% of the page dimension,
    # search for a more conservative boundary within that margin area.
    print("\nSearching for conservative boundaries...")

    # Top Conservative Boundary
    if boundaries['top'] > height * 0.05:
        grad = entropy_data['vertical']['scanning_avg_gradient']
        search_area = grad[0:boundaries['top']]
        peak_index = np.argmax(search_area)
        peak_value = search_area[peak_index]

        if peak_value > conservative_gradient_threshold:
            conservative_boundaries['top'] = peak_index + 1
            print(f"Found conservative top boundary at {peak_index}, adjusting to {conservative_boundaries['top']}")
        else:
            conservative_boundaries['top'] = boundaries['top']
            print("No secondary top peak found, using primary boundary.")
    else:
        conservative_boundaries['top'] = boundaries['top']

    # Bottom Conservative Boundary
    if boundaries['bottom'] < height * 0.95:
        grad = entropy_data['vertical']['scanning_avg_gradient']
        search_area = grad[boundaries['bottom']:]
        peak_offset = np.argmin(search_area)
        peak_value = search_area[peak_offset]
        
        if peak_value < -conservative_gradient_threshold:
            conservative_boundaries['bottom'] = boundaries['bottom'] + peak_offset - 1
            print(f"Found conservative bottom boundary at {boundaries['bottom'] + peak_offset}, adjusting to {conservative_boundaries['bottom']}")
        else:
            conservative_boundaries['bottom'] = boundaries['bottom']
            print("No secondary bottom peak found, using primary boundary.")
    else:
        conservative_boundaries['bottom'] = boundaries['bottom']

    # Left Conservative Boundary
    if boundaries['left'] > width * 0.05:
        grad = entropy_data['horizontal']['scanning_avg_gradient']
        search_area = grad[0:boundaries['left']]
        peak_index = np.argmax(search_area)
        peak_value = search_area[peak_index]

        if peak_value > conservative_gradient_threshold:
            conservative_boundaries['left'] = peak_index + 1
            print(f"Found conservative left boundary at {peak_index}, adjusting to {conservative_boundaries['left']}")
        else:
            conservative_boundaries['left'] = boundaries['left']
            print("No secondary left peak found, using primary boundary.")
    else:
        conservative_boundaries['left'] = boundaries['left']

    # Right Conservative Boundary
    if boundaries['right'] < width * 0.95:
        grad = entropy_data['horizontal']['scanning_avg_gradient']
        search_area = grad[boundaries['right']:]
        peak_offset = np.argmin(search_area)
        peak_value = search_area[peak_offset]

        if peak_value < -conservative_gradient_threshold:
            conservative_boundaries['right'] = boundaries['right'] + peak_offset - 1
            print(f"Found conservative right boundary at {boundaries['right'] + peak_offset}, adjusting to {conservative_boundaries['right']}")
        else:
            conservative_boundaries['right'] = boundaries['right']
            print("No secondary right peak found, using primary boundary.")
    else:
        conservative_boundaries['right'] = boundaries['right']


    # --- Final Validation ---
    # Ensure boundaries are logical (top < bottom, left < right)
    if boundaries['top'] >= boundaries['bottom']:
        print(f"Warning: Invalid vertical boundaries detected (top: {boundaries['top']}, bottom: {boundaries['bottom']}). Resetting to page edges.")
        boundaries['top'] = 0
        boundaries['bottom'] = height
        
    if boundaries['left'] >= boundaries['right']:
        print(f"Warning: Invalid horizontal boundaries detected (left: {boundaries['left']}, right: {boundaries['right']}). Resetting to page edges.")
        boundaries['left'] = 0
        boundaries['right'] = width
        
    if conservative_boundaries.get('top', 0) >= conservative_boundaries.get('bottom', height):
        print(f"Warning: Invalid conservative vertical boundaries. Resetting to page edges.")
        conservative_boundaries['top'] = 0
        conservative_boundaries['bottom'] = height
        
    if conservative_boundaries.get('left', 0) >= conservative_boundaries.get('right', width):
        print(f"Warning: Invalid conservative horizontal boundaries. Resetting to page edges.")
        conservative_boundaries['left'] = 0
        conservative_boundaries['right'] = width
        
    return boundaries, conservative_boundaries

def draw_entropy_overlay(page, entropy_data, metrics_to_plot, dpi, crop_rect=None, conservative_crop_rect=None):
    """
    Draws the selected entropy curves and the proposed crop box directly onto the PDF page.
    """
    colors = {
        'entropy': (0, 0, 1),   # Blue
        'scanning': (0, 1, 0),  # Green
        'gradient': (1, 0, 0)   # Red
    }
    scaling = dpi / 72.0
    page_rect = page.rect

    # --- Draw Conservative Crop Box ---
    if conservative_crop_rect and not conservative_crop_rect.is_empty:
        corner_size = 15
        corner_color = (0, 1, 0) # Green for conservative
        corner_width = 1.5
        
        # Top-Left
        tl = conservative_crop_rect.tl
        page.draw_line(tl, tl + (corner_size, 0), color=corner_color, width=corner_width)
        page.draw_line(tl, tl + (0, corner_size), color=corner_color, width=corner_width)
        
        # Top-Right
        tr = conservative_crop_rect.tr
        page.draw_line(tr, tr - (corner_size, 0), color=corner_color, width=corner_width)
        page.draw_line(tr, tr + (0, corner_size), color=corner_color, width=corner_width)

        # Bottom-Left
        bl = conservative_crop_rect.bl
        page.draw_line(bl, bl + (corner_size, 0), color=corner_color, width=corner_width)
        page.draw_line(bl, bl - (0, corner_size), color=corner_color, width=corner_width)

        # Bottom-Right
        br = conservative_crop_rect.br
        page.draw_line(br, br - (corner_size, 0), color=corner_color, width=corner_width)
        page.draw_line(br, br - (0, corner_size), color=corner_color, width=corner_width)


    # --- Draw Proposed Crop Box Corners ---
    if crop_rect and not crop_rect.is_empty:
        corner_size = 20  # points
        corner_color = (1, 0, 1)  # Magenta
        corner_width = 1.5
        
        # Top-Left
        tl = crop_rect.tl
        page.draw_line(tl, tl + (corner_size, 0), color=corner_color, width=corner_width)
        page.draw_line(tl, tl + (0, corner_size), color=corner_color, width=corner_width)
        
        # Top-Right
        tr = crop_rect.tr
        page.draw_line(tr, tr - (corner_size, 0), color=corner_color, width=corner_width)
        page.draw_line(tr, tr + (0, corner_size), color=corner_color, width=corner_width)

        # Bottom-Left
        bl = crop_rect.bl
        page.draw_line(bl, bl + (corner_size, 0), color=corner_color, width=corner_width)
        page.draw_line(bl, bl - (0, corner_size), color=corner_color, width=corner_width)

        # Bottom-Right
        br = crop_rect.br
        page.draw_line(br, br - (corner_size, 0), color=corner_color, width=corner_width)
        page.draw_line(br, br - (0, corner_size), color=corner_color, width=corner_width)
        
    # --- Draw Vertical Scan (Left Edge) ---
    plot_area_width = page_rect.width * 0.05  # Use 5% of page width for the plot
    vertical_data = entropy_data['vertical']

    for metric in metrics_to_plot:
        metric_key = 'gradient_normalized' if metric == 'gradient' else metric
        if metric_key not in vertical_data:
            continue
            
        data = vertical_data[metric_key]
        points = [
            fitz.Point(value * plot_area_width, i / scaling)
            for i, value in enumerate(data)
        ]
        
        for i in range(len(points) - 1):
            page.draw_line(points[i], points[i+1], color=colors[metric], width=0.5)

    # --- Draw Horizontal Scan (Bottom Edge) ---
    plot_area_height = page_rect.height * 0.05 # Use 5% of page height for the plot
    horizontal_data = entropy_data['horizontal']

    for metric in metrics_to_plot:
        metric_key = 'gradient_normalized' if metric == 'gradient' else metric
        if metric_key not in horizontal_data:
            continue

        data = horizontal_data[metric_key]
        points = [
            fitz.Point(i / scaling, page_rect.height - (value * plot_area_height))
            for i, value in enumerate(data)
        ]

        for i in range(len(points) - 1):
            page.draw_line(points[i], points[i+1], color=colors[metric], width=0.5)

    # --- Draw Legend ---
    legend_items = [m for m in metrics_to_plot if m in colors]
    if not legend_items:
        return

    # Create a white background for the legend
    legend_rect = fitz.Rect(10, 10, 100, 10 + len(legend_items) * 15 + 5)
    page.draw_rect(legend_rect, color=None, fill=(1, 1, 1), overlay=True)
    page.draw_rect(legend_rect, color=(0,0,0), width=0.5, overlay=True)


    for i, metric in enumerate(legend_items):
        color = colors[metric]
        # Draw color swatch
        swatch_y = legend_rect.y0 + 5 + (i * 15)
        page.draw_line(
            fitz.Point(legend_rect.x0 + 5, swatch_y + 5),
            fitz.Point(legend_rect.x0 + 15, swatch_y + 5),
            color=color,
            width=2,
            overlay=True
        )
        # Insert text
        page.insert_text(
            fitz.Point(legend_rect.x0 + 20, swatch_y + 9),
            metric.capitalize(),
            fontsize=8,
            overlay=True
        )

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Crop PDF borders based on entropy changes or overlay entropy curves for analysis.")
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument("output_pdf", help="Path to the output PDF file")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendering pages (default: 300)")
    parser.add_argument("--plot-dir", default="entropy_plots", help="Directory for entropy plots (generated unless --overlay is used)")
    parser.add_argument("--overlay", nargs='+', choices=['entropy', 'scanning', 'gradient'], 
                        help="Overlay entropy curves on the output PDF instead of cropping. "
                             "When specified, no cropping is performed. "
                             "Can specify one or more metrics to plot.")
    parser.add_argument("--conservative", action="store_true", 
                        help="Use conservative cropping boundaries if a large crop is detected.")
    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input_pdf):
        print(f"Error: Input file '{args.input_pdf}' does not exist.")
        exit(1)

    # Create plot directory if not overlaying
    if not args.overlay:
        os.makedirs(args.plot_dir, exist_ok=True)

    # Open the PDF
    doc = fitz.open(args.input_pdf)
    out_doc = fitz.open()  # Create new PDF for output
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"\nProcessing page {page_num + 1}")
        
        # Get original page dimensions
        orig_width = page.rect.width
        orig_height = page.rect.height
        
        # Render page as an image
        pix = page.get_pixmap(dpi=args.dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        # Convert to grayscale
        gray = img if pix.n == 1 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        print(f"Image dimensions: {pix.width} x {pix.height} pixels")
        
        # Analyze entropy patterns
        entropy_data = analyze_page_entropy(gray)
        
        if args.overlay:
            # --- Overlay Mode ---
            # Find crop boundaries to visualize them
            boundaries, conservative_boundaries = find_crop_boundaries(entropy_data)
            scaling = args.dpi / 72.0
            crop_rect = fitz.Rect(
                boundaries['left'] / scaling,
                boundaries['top'] / scaling,
                boundaries['right'] / scaling,
                boundaries['bottom'] / scaling
            )
            crop_rect = crop_rect & page.mediabox
            
            conservative_crop_rect = fitz.Rect(
                conservative_boundaries['left'] / scaling,
                conservative_boundaries['top'] / scaling,
                conservative_boundaries['right'] / scaling,
                conservative_boundaries['bottom'] / scaling
            )
            conservative_crop_rect = conservative_crop_rect & page.mediabox

            if crop_rect.is_empty or crop_rect.width <= 0 or crop_rect.height <= 0:
                print("Warning: Proposed crop is a degenerate rectangle. Will not be drawn.")
                crop_rect = None

            print("Proposed crop box (points):")
            if crop_rect:
                print(f"  (x0, y0, x1, y1): ({crop_rect.x0:.1f}, {crop_rect.y0:.1f}, {crop_rect.x1:.1f}, {crop_rect.y1:.1f})")
            else:
                print("  None")
            
            print("Conservative crop box (points):")
            if conservative_crop_rect:
                 print(f"  (x0, y0, x1, y1): ({conservative_crop_rect.x0:.1f}, {conservative_crop_rect.y0:.1f}, {conservative_crop_rect.x1:.1f}, {conservative_crop_rect.y1:.1f})")
            else:
                print("  None")

            # Create a new page in out_doc that is a copy of the original
            out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            out_page.show_pdf_page(out_page.rect, doc, page_num)
            
            # Draw the entropy curves and crop box on the new page
            draw_entropy_overlay(out_page, entropy_data, args.overlay, args.dpi, crop_rect, conservative_crop_rect)
            print(f"Added entropy overlays and crop box outline for page {page_num + 1}")

        else:
            # --- Cropping Mode ---
            # Plot entropy data for debugging
            plot_entropy_curves(page_num + 1, entropy_data, args.plot_dir)
            
            # Find crop boundaries
            boundaries, conservative_boundaries = find_crop_boundaries(entropy_data)

            # Choose which boundaries to use
            final_boundaries = conservative_boundaries if args.conservative else boundaries
            
            # Convert pixel coordinates to PDF points
            scaling = args.dpi / 72.0  # DPI to points conversion
            left = final_boundaries['left'] / scaling
            top = final_boundaries['top'] / scaling
            right = final_boundaries['right'] / scaling
            bottom = final_boundaries['bottom'] / scaling
            
            # Create crop rectangle
            crop_rect = fitz.Rect(left, top, right, bottom)
            
            # Ensure crop_rect is within MediaBox
            mediabox = page.mediabox
            crop_rect = crop_rect & mediabox
            
            # Final check for degenerate rectangles after intersection
            if crop_rect.is_empty or crop_rect.width <= 0 or crop_rect.height <= 0:
                print("Warning: Crop resulted in a degenerate rectangle. Using original page dimensions.")
                crop_rect = mediabox
            
            # Print detailed cropping information
            print(f"Crop coordinates (points):")
            print(f"  Left: {crop_rect.x0:.1f} (was {page.rect.x0:.1f})")
            print(f"  Top: {crop_rect.y0:.1f} (was {page.rect.y0:.1f})")
            print(f"  Right: {crop_rect.x1:.1f} (was {page.rect.x1:.1f})")
            print(f"  Bottom: {crop_rect.y1:.1f} (was {page.rect.y1:.1f})")
            print(f"Final dimensions: {crop_rect.width:.1f} x {crop_rect.height:.1f} points")
            
            # Create new page with crop applied
            new_page = out_doc.new_page(width=crop_rect.width, height=crop_rect.height)
            
            # Copy content from original page to new page with crop
            new_page.show_pdf_page(
                new_page.rect,  # target rectangle
                doc,            # source document
                page_num,       # source page number
                clip=crop_rect  # source rectangle
            )

    # Save the cropped PDF
    out_doc.save(args.output_pdf)
    out_doc.close()
    doc.close()
    
    if args.overlay:
        print(f"\nPDF with overlays saved to '{args.output_pdf}'.")
    else:
        print(f"\nCropped PDF saved to '{args.output_pdf}'. Entropy plots are in '{args.plot_dir}'.")

def plot_entropy_curves(page_num, entropy_data, output_dir):
    """
    Generate separate plot files for vertical and horizontal scans for each page,
    with plots separated into logical groups.
    """
    # --- Plot Vertical Scan Data ---
    fig_vert, (ax_vert1, ax_vert2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig_vert.suptitle(f'Page {page_num} Vertical Scan Analysis', fontsize=16)
    
    vertical = entropy_data['vertical']
    x_vert = np.arange(len(vertical['entropy']))
    
    # Top subplot: Entropy values
    ax_vert1.plot(x_vert, vertical['entropy'], 'b-', label='Line Entropy', alpha=0.7)
    ax_vert1.plot(x_vert, vertical['scanning'], 'g-', label='Scanning Entropy', alpha=0.7)
    ax_vert1.plot(x_vert, vertical['scanning_avg'], 'c-', label='Scanning Entropy (Avg)', alpha=0.6)
    ax_vert1.set_title("Entropy Values")
    ax_vert1.set_ylabel('Normalized Value')
    ax_vert1.grid(True, alpha=0.3)
    ax_vert1.legend()
    
    # Bottom subplot: Gradient values
    ax_vert2.plot(x_vert, vertical['gradient_normalized'], 'r--', label='Normalized Gradient', alpha=0.5)
    ax_vert2.plot(x_vert, vertical['scanning_avg_gradient_normalized'], 'm--', label='Scanning Avg Gradient (Norm)', alpha=0.5)
    ax_vert2.set_title("Gradient Values")
    ax_vert2.set_xlabel('Position (pixels, Top to Bottom)')
    ax_vert2.set_ylabel('Normalized Value')
    ax_vert2.grid(True, alpha=0.3)
    ax_vert2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plot_path_vert = os.path.join(output_dir, f"page_{page_num}_vertical_analysis.png")
    plt.savefig(plot_path_vert, dpi=150)
    plt.close(fig_vert)

    # --- Plot Horizontal Scan Data ---
    fig_horz, (ax_horz1, ax_horz2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig_horz.suptitle(f'Page {page_num} Horizontal Scan Analysis', fontsize=16)
    
    horizontal = entropy_data['horizontal']
    x_horz = np.arange(len(horizontal['entropy']))
    
    # Top subplot: Entropy values
    ax_horz1.plot(x_horz, horizontal['entropy'], 'b-', label='Line Entropy', alpha=0.7)
    ax_horz1.plot(x_horz, horizontal['scanning'], 'g-', label='Scanning Entropy', alpha=0.7)
    ax_horz1.plot(x_horz, horizontal['scanning_avg'], 'c-', label='Scanning Entropy (Avg)', alpha=0.6)
    ax_horz1.set_title("Entropy Values")
    ax_horz1.set_ylabel('Normalized Value')
    ax_horz1.grid(True, alpha=0.3)
    ax_horz1.legend()
    
    # Bottom subplot: Gradient values
    ax_horz2.plot(x_horz, horizontal['gradient_normalized'], 'r--', label='Normalized Gradient', alpha=0.5)
    ax_horz2.plot(x_horz, horizontal['scanning_avg_gradient_normalized'], 'm--', label='Scanning Avg Gradient (Norm)', alpha=0.5)
    ax_horz2.set_title("Gradient Values")
    ax_horz2.set_xlabel('Position (pixels, Left to Right)')
    ax_horz2.set_ylabel('Normalized Value')
    ax_horz2.grid(True, alpha=0.3)
    ax_horz2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plot_path_horz = os.path.join(output_dir, f"page_{page_num}_horizontal_analysis.png")
    plt.savefig(plot_path_horz, dpi=150)
    plt.close(fig_horz)

if __name__ == "__main__":
    main()