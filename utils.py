import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import os

SYSTEM_PROMPT = (
    "You are given {num_frames} frames from a video. The frame indices are {selected_frame_idxs}."
)

PROMPT_TEMPLATES = [
        "Point to {label}\nPlease say 'This isn't in the video.' if it is not in the video.",
        "Point to all occurrences of \"{label}\"",
        "Point to any {label} in the video",
        "Point to any {label} in the video.",
        "Point: Where are the {label}",
        "Show me where the {label} are",
        "Can you show me where the {label} are?",
        "Show me where the {label} are",
        "Show me where {label} is",
        "Show me where {label} is.",
        "If there are any {label} in the video? Show me where they are.",
        "Where are {label}?",
        "Generate a list of points showing where {label} are.",
        "Find \"{label}\".",
        "Find \"{label}\".",
        "Locate all {label}.",
        "Locate {label}.",
        "Locate every {label}.",
        "Locate {label}.",
        "Locate {label}.",
        "Object: {label}\nInstruction: Point to the object.",
        "find {label}",
        "find {label}.",
        "Point to every {label}",
        "find any {label} in the picture",
        "Find {label}",
        "Find any {label}",
        "Point to {label}",
        "Point to {label}",
        "Look for {label} in the video and show me where they are.",
        "Help me find an object in the video by pointing to them.\nObject: {label}.",
        "I am looking for {label}, where can they be found in the video?",
        "Can you see any {label} in the video? Point to them.",
        "Point out each {label} in the video.",
        "Point out every {label} in the video.",
        "Point to {label} in the video.",
        "Locate each {label} in the video.",
        "Can you point out all {label} in this video?",
        "Please find {label} and show me where they are.",
        "If there are any {label} present, indicate their positions.",
        "If there is {label} present, indicate its positions.",
        "show me all visible {label}",
    ]

def extract_caption(caption):
    # Remove common question words to extract the key label
    label = re.sub(r"^(Where is|Where are|Find|Show me|Locate|Can you see|Segment the)", "", caption, flags=re.IGNORECASE).strip()
    return label

def get_points_in_xml_format(points, caption):
    lines = ["<points"]
    
    for t, xy_list in points.items():
        line = f' t="{int(t)}"'
        xy_list = sorted(xy_list, key=lambda p: (p["x"], p["y"]))
        for i, xy in enumerate(xy_list):
            if len(xy_list) == 1:  
                x, y = xy["x"], xy["y"]
                line += f' x="{x:.1f}" y="{y:.1f}"'
            else:
                x, y = xy["x"], xy["y"]
                line += f' x{i+1}="{x:.1f}" y{i+1}="{y:.1f}"'
        lines.append(line)

    lines.append(f' alt="{caption}">{caption}</points>')
    output = "".join(lines)
    return output


def add_points(predictor, input_points, input_labels, inference_state, f=0, device='cuda'):
    for i in range(len(input_points)):
        input_point = np.array([input_points[i]])
        input_label = np.array([input_labels[i]])
        ann_frame_idx = f # Frame index to interact/start with.
        ann_object_id = i # Give a unique object ID to the object, an integer.

        with torch.inference_mode(), torch.autocast(str(device), dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_object_id,
                points=input_point,
                labels=input_label
            )
    return out_obj_ids, out_mask_logits

def draw_point_and_show(image_path=None, points=None):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    for point in points:
        image = cv2.circle(
            image, 
            (point[0], point[1]), 
            radius=5, 
            color=(0, 255, 0), 
            thickness=5,
            lineType=cv2.LINE_AA
        )

    plt.imshow(image[..., ::-1])
    plt.axis('off')
    plt.show()



def get_coords(output_string, image=Image.new("RGB", (100, 100), color=(255, 0, 0))):
    w, h = image.size
    coords_dict = {}
    
    # Pattern matches t="frame", and optionally x and y attributes.
    pattern = r't="(\d+)"(?:\s+x="([\d.]+)"\s+y="([\d.]+)")?'
    matches = re.findall(pattern, output_string)
    
    for frame_str, x_str, y_str in matches:
        frame = int(frame_str)
        if x_str and y_str:
            # Convert the x and y percentages to absolute pixel coordinates
            x_val = int(float(x_str) / 100 * w)
            y_val = int(float(y_str) / 100 * h)
            coords_dict[frame] = (x_val, y_val)
        else:
            # If one or both coordinates are missing, set empty tuple
            coords_dict[frame] = ()
    
    return coords_dict

def compute_mse_points(pred_text, gt_text, image=Image.new("RGB", (100, 100), color=(255, 0, 0))):
    w, h = image.size
    
    def extract_coords(text):
        text = str(text)
        coords = {}
        # Iterate over each <points ...>...</points> block.
        for block_match in re.finditer(r'<points\s+(.*?)>(.*?)</points>', text, flags=re.DOTALL | re.IGNORECASE):
            attributes, content = block_match.groups()
            has_none = "There are none" in content
            # Extract all frame entries from the attributes.
            for t_str, x_str, y_str in re.findall(r't="(\d+)"(?:\s+x="([\d.]+)"\s+y="([\d.]+)")?', attributes, flags=re.IGNORECASE):
                frame = int(t_str)
                if has_none and (not x_str or not y_str):
                    coords[frame] = "none"
                else:
                    try:
                        if x_str and y_str:
                            x_val = float(x_str)
                            y_val = float(y_str)
                            coords[frame] = (int(x_val/100 * w), int(y_val/100 * h))
                        else:
                            coords[frame] = ()
                    except ValueError:
                        coords[frame] = ()
        return coords

    pred_coords = extract_coords(pred_text)
    gt_coords = extract_coords(gt_text)
    all_frames = set(pred_coords.keys()).union(gt_coords.keys())
    
    mse_list = []
    for frame in all_frames:
        p = pred_coords.get(frame, ())
        g = gt_coords.get(frame, ())
        if p == "none" and g == "none":
            mse_list.append(0)
        elif not p or not g:
            mse_list.append(100 * 100)
        else:
            mse_list.append(((p[0]-g[0])**2 + (p[1]-g[1])**2)/2.0)
    return sum(mse_list)/len(mse_list) if len(mse_list) == 2 else 100*100


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap('tab10')
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    ax.axis('off')

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def load_png_as_array(directory):
    boolean_arrays = []
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            file_path = os.path.join(directory, filename)
            # Open the image and convert to grayscale
            image = Image.open(file_path).convert('L')
            # Convert image to a NumPy array
            image_array = np.array(image)
            # Convert to a boolean array (nonzero pixels -> True, zero pixels -> False)
            boolean_array = image_array > 0
            boolean_arrays.append(boolean_array)
    return boolean_arrays

# Function to recursively convert ndarrays to lists
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    return obj


import shutil
from natsort import natsorted  # Ensures correct sorting of filenames

def save_reversed_frames(input_dir, output_dir):
    # Get all frame filenames and sort them naturally (e.g., frame_1.png, frame_2.png, ...)
    frame_files = natsorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not frame_files:
        print("Error: No image frames found in the directory.")
        return
    
    # Reverse the order of frames
    frame_files.reverse()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save reversed frames by copying them to the output directory
    for idx, frame_file in enumerate(frame_files):
        src_path = os.path.join(input_dir, frame_file)
        # Preserve the original file extension
        _, ext = os.path.splitext(frame_file)
        dest_filename = f"{idx:05d}{ext}"
        dest_path = os.path.join(output_dir, dest_filename)
        shutil.copy2(src_path, dest_path)

    print("Reversed frames saved successfully!")


def save_images_folder(np_images):
    # Save each image in the list as a JPEGx
    if not os.path.exists('temp'):
        os.makedirs('temp')
    for idx, img in enumerate(np_images):
        save_path = os.path.join('temp', f"{idx}.jpg")
        img_pil = Image.fromarray(img.astype(np.uint8))  # Convert NumPy array to PIL Image
        img_pil.save(save_path, "JPEG")



def plot_metric(metric_values, metric_name, folder='plots'):
    filename = f"{folder}/{metric_name}.png"
    os.makedirs(folder, exist_ok=True)    
    plt.figure(figsize=(10, 6))
    plt.plot(metric_values, marker='o', linestyle='-', label=metric_name)
    plt.xlabel("Steps")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def pil_to_np(images):
    image_arrays = []
    for image in images:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image_arrays.append(np.array(image))
    images = image_arrays
    return images