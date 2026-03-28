import os
import hashlib
import torch
import numpy as np
import json
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import folder_paths
import node_helpers


class LoadImageWithFilename:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Click folder icon to browse",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "save_path", "file_extension")
    FUNCTION = "load_image"

    def load_image(self, image, unique_id=None):
        if not image:
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "", "", "")

        # Check if it's a full path (starts with drive letter or /)
        is_full_path = os.path.isabs(image) or (len(image) > 2 and image[1] == ":")

        if is_full_path and os.path.isfile(image):
            # It's a full path - use directly
            image_path = image
        elif os.path.isfile(image):
            # It's a full path (could be unix style)
            image_path = image
        else:
            # It's a filename in the input folder
            image_path = folder_paths.get_annotated_filepath(image)

        if not os.path.isfile(image_path):
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "", "", "")

        folder_path = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        file_extension = os.path.splitext(filename)[1].lower()

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            elif i.mode == "P" and "transparency" in i.info:
                mask = (
                    np.array(i.convert("RGBA").getchannel("A")).astype(np.float32)
                    / 255.0
                )
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Extract filename from the image path
        filename = os.path.basename(image_path)

        return (output_image, output_mask, filename, folder_path, file_extension)

    @classmethod
    def IS_CHANGED(s, image):
        if not image:
            return ""

        # Check if it's a full path or needs to check input folder
        if os.path.isfile(image):
            image_path = image
        else:
            try:
                image_path = folder_paths.get_annotated_filepath(image)
            except:
                return ""

        if os.path.isfile(image_path):
            m = hashlib.sha256()
            with open(image_path, "rb") as f:
                m.update(f.read())
            return m.digest().hex()
        return ""

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not image:
            return "Image is required"
        return True


class LoadImageFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Folder containing images",
                    },
                ),
            },
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filenames", "save_path", "file_extensions")
    FUNCTION = "load_folder"

    def load_folder(self, folder_path):
        # If no path provided, return empty tensors
        if not folder_path or not os.path.exists(folder_path):
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "", "", "")

        # Check if it's a directory
        if not os.path.isdir(folder_path):
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "", "", "")

        # Get all image files in the folder, sorted alphabetically
        image_files = []
        for file in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                try:
                    with Image.open(file_path) as img:
                        image_files.append(file)
                except:
                    continue

        if not image_files:
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "", folder_path, "")

        # Load the first image only
        filename = image_files[0]
        file_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()

        try:
            img = node_helpers.pillow(Image.open, file_path)

            output_images = []
            output_masks = []

            excluded_formats = ["MPO"]

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == "I":
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")

                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]

                if "A" in i.getbands():
                    mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                    mask = 1.0 - torch.from_numpy(mask)
                elif i.mode == "P" and "transparency" in i.info:
                    mask = (
                        np.array(i.convert("RGBA").getchannel("A")).astype(np.float32)
                        / 255.0
                    )
                    mask = 1.0 - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))

            if len(output_images) > 1 and img.format not in excluded_formats:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]

            print(f"[LoadImageFolder] Loaded: {filename}")
            print(
                f"[LoadImageFolder] {len(image_files) - 1} images remaining. Queue more runs to process all."
            )

            return (output_image, output_mask, [filename], folder_path, [ext])

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "", folder_path, "")

    @classmethod
    def IS_CHANGED(s, folder_path):
        if not folder_path or not os.path.exists(folder_path):
            return "INVALID"

        # Hash based on folder contents
        m = hashlib.sha256()
        try:
            for file in sorted(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, "rb") as f:
                            m.update(f.read())
                    except:
                        continue
        except:
            return "INVALID"
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, folder_path):
        if not folder_path:
            return "Empty path provided"
        if not os.path.exists(folder_path):
            return "Path does not exist: {}".format(folder_path)
        if not os.path.isdir(folder_path):
            return "Not a directory: {}".format(folder_path)
        return True


class SaveImageWithFilename:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filenames": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Single filename or comma-separated list of filenames. If empty, will use default naming.",
                    },
                ),
                "file_extensions": (
                    "STRING",
                    {
                        "default": ".png",
                        "tooltip": "File extension(s) - connect from load node or provide comma-separated list (.png, .jpg, etc).",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, saves with the provided filename (overwrites existing). If disabled, appends number to avoid overwrite.",
                    },
                ),
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Output directory. If empty, uses ComfyUI default output folder.",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images with specified filenames to your ComfyUI output directory."

    def save_images(
        self,
        images,
        filenames="",
        file_extensions=".png",
        overwrite=False,
        output_path="",
        filename_prefix="ComfyUI",
        prompt=None,
        extra_pnginfo=None,
    ):
        filename_prefix += self.prefix_append

        # Determine output directory
        if output_path and os.path.isdir(output_path):
            output_dir = output_path
            use_default_output = False
        else:
            output_dir = self.output_dir
            use_default_output = True

        # Parse filenames - handle both string and list inputs
        filename_list = []
        if filenames:
            # Handle both string and list inputs
            if isinstance(filenames, str):
                # Split by comma and strip whitespace
                filename_list = [f.strip() for f in filenames.split(",") if f.strip()]
            elif isinstance(filenames, list):
                # Already a list, just use it
                filename_list = [str(f).strip() for f in filenames if f]
            else:
                # Convert to string and try to split
                filename_str = str(filenames)
                filename_list = [
                    f.strip() for f in filename_str.split(",") if f.strip()
                ]

        # Parse file extensions
        ext_list = []
        if file_extensions:
            if isinstance(file_extensions, str):
                ext_list = [e.strip() for e in file_extensions.split(",") if e.strip()]
            elif isinstance(file_extensions, list):
                ext_list = [str(e).strip() for e in file_extensions if e]
            else:
                ext_str = str(file_extensions)
                ext_list = [e.strip() for e in ext_str.split(",") if e.strip()]

        # Default to .png if no extensions provided
        if not ext_list:
            ext_list = [".png"]

        # If no filenames provided or not enough filenames, use default naming
        if not filename_list or len(filename_list) < len(images):
            # Use default naming for remaining images
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(
                    filename_prefix,
                    output_dir,
                    images[0].shape[1],
                    images[0].shape[0],
                )
            )

            results = list()
            for batch_number, image in enumerate(images):
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                # Use provided filename if available, otherwise use default naming
                # Get extension for this batch item
                ext = ext_list[batch_number] if batch_number < len(ext_list) else ".png"

                # Determine save format based on extension
                save_format = "PNG"
                save_kwargs = {
                    "pnginfo": metadata,
                    "compress_level": self.compress_level,
                }
                if ext.lower() in [".jpg", ".jpeg"]:
                    save_format = "JPEG"
                    save_kwargs = {"quality": 95}
                    if metadata:
                        # JPEG doesn't support PNG metadata, add as comment
                        pass

                if batch_number < len(filename_list):
                    # Use the provided filename
                    provided_filename = filename_list[batch_number]
                    # Remove extension if present and add correct extension
                    base_name = os.path.splitext(provided_filename)[0]
                    file = f"{base_name}{ext}"
                else:
                    # Use default naming
                    filename_with_batch_num = filename.replace(
                        "%batch_num%", str(batch_number)
                    )
                    file = f"{filename_with_batch_num}_{counter:05}_{ext[1:]}"

                img.save(
                    os.path.join(full_output_folder, file),
                    format=save_format,
                    **save_kwargs,
                )
                results.append(
                    {"filename": file, "subfolder": subfolder, "type": self.type}
                )
        else:
            # Use provided filenames for all images
            results = list()
            for batch_number, image in enumerate(images):
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                # Use the provided filename
                # Get extension for this batch item
                ext = ext_list[batch_number] if batch_number < len(ext_list) else ".png"

                # Determine save format based on extension
                save_format = "PNG"
                save_kwargs = {
                    "pnginfo": metadata,
                    "compress_level": self.compress_level,
                }
                if ext.lower() in [".jpg", ".jpeg"]:
                    save_format = "JPEG"
                    save_kwargs = {"quality": 95}

                provided_filename = filename_list[batch_number]
                # Remove extension if present and add correct extension
                base_name = os.path.splitext(provided_filename)[0]
                file = f"{base_name}{ext}"

                # Save to output directory
                if overwrite:
                    # Overwrite mode: save directly with filename
                    img.save(
                        os.path.join(output_dir, file),
                        format=save_format,
                        **save_kwargs,
                    )
                else:
                    # No overwrite: find unique filename
                    save_path = os.path.join(output_dir, file)
                    if os.path.exists(save_path):
                        # Add counter to avoid overwrite
                        counter = 1
                        while os.path.exists(save_path):
                            file = f"{base_name}_{counter:03d}{ext}"
                            save_path = os.path.join(output_dir, file)
                            counter += 1
                    img.save(
                        save_path,
                        format=save_format,
                        **save_kwargs,
                    )

                results.append({"filename": file, "subfolder": "", "type": self.type})

        return {"ui": {"images": results}}

    @classmethod
    def IS_CHANGED(
        s,
        images,
        filenames,
        file_extensions,
        overwrite,
        output_path,
        filename_prefix,
        **kwargs,
    ):
        return hashlib.sha256(str(images).encode()).hexdigest()


class CropImageByMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to crop"}),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "The mask to use for cropping. White pixels (1.0) will be excluded, black pixels (0.0) will be included."
                    },
                ),
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "crop_by_mask"
    DESCRIPTION = "Crops an image based on its mask, keeping only areas with black pixels (0.0) in the mask and excluding white pixels (1.0)."

    def crop_by_mask(self, image, mask):
        # Ensure mask is binary (0.0 or 1.0)
        if mask.dim() == 2:
            # Single mask
            binary_mask = (mask > 0.5).float()
            return self._crop_single_image(image, binary_mask)
        else:
            # Batch of masks
            cropped_images = []
            cropped_masks = []

            for i in range(mask.shape[0]):
                single_mask = mask[i]
                single_image = image[i] if image.shape[0] > 1 else image[0]

                binary_mask = (single_mask > 0.5).float()
                cropped_img, cropped_mask = self._crop_single_image(
                    single_image, binary_mask
                )

                cropped_images.append(cropped_img)
                cropped_masks.append(cropped_mask)

            # Stack results
            if len(cropped_images) > 1:
                return (
                    torch.stack(cropped_images, dim=0),
                    torch.stack(cropped_masks, dim=0),
                )
            else:
                return (cropped_images[0].unsqueeze(0), cropped_masks[0].unsqueeze(0))

    def _crop_single_image(self, image, mask):
        # Convert to numpy for easier manipulation
        img_np = image.cpu().numpy()
        mask_np = mask.cpu().numpy()

        # Find the largest rectangle that contains only black pixels (0.0) and no white pixels (1.0)
        # Use a more efficient approach based on finding the largest rectangle in a binary matrix
        # where 0 = black (keep), 1 = white (exclude)

        # Create a binary matrix where 0 = black pixels (valid), 1 = white pixels (invalid)
        binary_matrix = (mask_np == 1).astype(
            int
        )  # 1 for white pixels, 0 for black pixels

        # Find the largest rectangle of 0s (black pixels)
        max_area, best_coords = self._largest_rectangle_of_zeros(binary_matrix)

        if max_area == 0:
            # No valid rectangle found, return original
            return image, mask

        y_min, y_max, x_min, x_max = best_coords

        # Crop the image and mask
        cropped_img = img_np[y_min : y_max + 1, x_min : x_max + 1]
        cropped_mask = mask_np[y_min : y_max + 1, x_min : x_max + 1]

        # Convert back to torch tensors
        cropped_img_tensor = torch.from_numpy(cropped_img).float()
        cropped_mask_tensor = torch.from_numpy(cropped_mask).float()

        return cropped_img_tensor, cropped_mask_tensor

    def _largest_rectangle_of_zeros(self, matrix):
        """Find the largest rectangle containing only 0s in a binary matrix."""
        if not matrix.size:
            return 0, (0, 0, 0, 0)

        rows, cols = matrix.shape
        max_area = 0
        best_coords = (0, 0, 0, 0)

        # For each cell, find the largest rectangle with this cell as the top-left corner
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 0:  # Only start from 0s
                    # Find the maximum width and height for this starting position
                    max_width = cols - j
                    max_height = rows - i

                    # Find the actual width (how many consecutive 0s to the right)
                    width = 0
                    for k in range(j, cols):
                        if matrix[i][k] == 0:
                            width += 1
                        else:
                            break

                    # Find the maximum height for this width
                    height = 0
                    for k in range(i, rows):
                        # Check if this row has 0s from j to j+width-1
                        if j + width <= cols and np.all(matrix[k][j : j + width] == 0):
                            height += 1
                        else:
                            break

                    area = width * height
                    if area > max_area:
                        max_area = area
                        best_coords = (i, i + height - 1, j, j + width - 1)

        return max_area, best_coords

    @classmethod
    def IS_CHANGED(s, image, mask, **kwargs):
        return hashlib.sha256(str(image).encode() + str(mask).encode()).hexdigest()
