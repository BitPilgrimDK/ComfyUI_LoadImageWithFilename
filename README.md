# ComfyUI LoadImageWithFilename (BUG right now dont use)

Custom nodes for loading and saving images while preserving filenames and paths.

## Features

### LoadImageWithFilename
- Load a single image from any folder path
- Browse button to select image (preserves original path)
- Outputs: image, mask, filename, save_path, file_extension
- save_path = the folder where the image is located

### LoadImageFolder
- Load images from a folder
- Takes folder_path as input
- **Returns ONE image at a time** (alphabetically first)
- Keeps original height and width intact
- Use ComfyUI's queue system to process more images
  - Each queue run loads the next image in the folder
- Outputs: image, mask, filenames, save_path, file_extensions

### SaveImageWithFilename
- Save images with custom filenames
- Accepts filenames input to name each image
- output_path - folder to save to (leave empty for default)
- overwrite toggle - True to overwrite, False to rename
- file_extensions - preserves original format (.jpg, .png, etc.)

## Installation

1. Place this folder in ComfyUI `custom_nodes` directory
2. Restart ComfyUI
3. Nodes appear in "image" category

## Usage

### LoadImageWithFilename
1. Enter folder path or click Browse button
2. Enter image filename
3. Connect outputs to other nodes

### LoadImageFolder
1. Enter folder path (e.g., `C:\Users\You\Images\textures`)
2. Run workflow - loads first image alphabetically
3. Queue more runs to process remaining images
4. Each run loads the next image

### SaveImageWithFilename
1. Connect images from processing nodes
2. Connect filenames from LoadImageWithFilename/LoadImageFolder
3. Set output_path (or leave empty for default)
4. Set overwrite = True to save to original location

## Example Workflow

```
LoadImageFolder (folder_path: "C:\textures")
    ↓
[Your processing nodes - upscale, etc.]
    ↓
SaveImageWithFilename (output_path: from LoadImageFolder, overwrite: True)
```

Queue multiple runs to process all images in the folder one by one.
