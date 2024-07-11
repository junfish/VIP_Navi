import os
from PIL import Image, ImageDraw, ImageFont
import json

base_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/"

line_spacing = 4  # Adjustable line spacing

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def load_image_paths(multiscene_file_path):
    filenames = []
    with open(multiscene_file_path, 'r', encoding='utf-8') as file:
        # Skip the first two lines (header lines)
        lines = file.readlines()[2:]
        for line in lines:
            # Extract the filename, which is the first element in the comma-separated values
            filename = line.split(',')[0].strip()
            filenames.append(filename)
    return filenames

def calculate_text_height_and_wrap(text, font, max_width,line_spacing):
    """Wrap text to fit within a specified width and calculate the height needed."""
    temp_img = Image.new('RGB', (max_width, 100))  # Temp image for drawing
    draw = ImageDraw.Draw(temp_img)
    words = text.split()
    lines = []
    current_line = ''
    total_height = 0

    for word in words:
        test_line = f"{current_line} {word}" if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]  # right - left
        if line_width > max_width:
            lines.append(current_line)
            current_line = word
            total_height += bbox[3] - bbox[1] + line_spacing  # bottom - top + spacing
        else:
            current_line = test_line
    if current_line:  # add last line
        lines.append(current_line)
        bbox = draw.textbbox((0, 0), current_line, font=font)
        total_height += bbox[3] - bbox[1]  # bottom - top

    return total_height, lines

def calculate_text_width_and_wrap(text, font, max_height, line_spacing):
    """Wrap text to fit within a specified height and calculate the width needed."""
    temp_img = Image.new('RGB', (100, max_height))  # Temp image for drawing
    draw = ImageDraw.Draw(temp_img)
    words = text.split()
    lines = []
    current_line = ''
    current_height = 0
    max_line_width = 0

    for word in words:
        test_line = f"{current_line} {word}" if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_height = bbox[3] - bbox[1]  # bottom - top
        line_width = bbox[2] - bbox[0]  # right - left

        if line_height > max_height:
            lines.append(current_line)
            current_line = word
            current_height = line_height + line_spacing  # Reset current height
            max_line_width = max(max_line_width, line_width)  # Update max line width if needed
        else:
            current_line = test_line
            current_height += line_height + line_spacing
            max_line_width = max(max_line_width, line_width)

    if current_line:  # add last line
        lines.append(current_line)
        bbox = draw.textbbox((0, 0), current_line, font=font)
        max_line_width = max(max_line_width, bbox[2] - bbox[0])  # Update max line width if needed

    return max_line_width, lines


def combine_text_image(text, image_path, output_base_path, font_size=50):

    # Determine the output directory based on the image path
    if "Basement" in image_path:
        output_folder = os.path.join(output_base_path, "Basement")
    elif "Level_2" in image_path:
        output_folder = os.path.join(output_base_path, "Level_2")
    elif "Level_1" in image_path:
        output_folder = os.path.join(output_base_path, "Level_1")
    elif "Lower_Level" in image_path:
        output_folder = os.path.join(output_base_path, "Lower_Level")
    else:
        output_folder = output_base_path

    # image_path = os.path.join(base_path, image_path)
    image_path = base_path + 'test/Level_1/' + image_path
    output_folder = output_base_path + 'Level_1/'

    # Load the image
    with Image.open(image_path) as img:
        # Load font
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            print("Failed to load specified font. Using default font.")
            font = ImageFont.load_default()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Create draw object for the original image
        draw = ImageDraw.Draw(img)

        # Add "left" text to the left of the original image
        margin = 10
        text_left = "left"
        bbox_left = draw.textbbox((margin, img.height / 2), text_left, font=font)
        adjusted_bbox_left = (bbox_left[0], bbox_left[1] - margin, bbox_left[2] + margin, bbox_left[3] + margin)
        draw.rectangle(adjusted_bbox_left, fill="white")
        draw.text((adjusted_bbox_left[0] + margin, adjusted_bbox_left[1] + margin), text_left, font=font, fill=(0, 0, 0))

        # Add "right" text to the right of the original image
        text_right = "right"
        temp_position = (img.width - margin, img.height / 2)  # Temp position for width calculation
        bbox_right_temp = draw.textbbox(temp_position, text_right, font=font)
        text_width = bbox_right_temp[2] - bbox_right_temp[0]
        text_x_right = img.width - text_width - 2 * margin  # Place text taking into account margins
        bbox_right = draw.textbbox((text_x_right, img.height / 2), text_right, font=font)
        adjusted_bbox_right = (bbox_right[0] - margin, bbox_right[1] - margin, bbox_right[2] + margin, bbox_right[3] + margin)
        draw.rectangle(adjusted_bbox_right, fill="white")
        draw.text((text_x_right + margin, adjusted_bbox_right[1] + margin), text_right, font=font, fill=(0, 0, 0))

        # Handle text wrapping based on image dimensions
        # if img.height > img.width:
        #     # Calculate the text dimensions
        #     max_text_width = (img.height - img.width) - 40# Margin
        #     text_height, lines = calculate_text_height_and_wrap(text, font, max_text_width, line_spacing)
        #
        #     # Create new image with space for text to the right
        #     new_image_width = img.width + max_text_width + 40
        #     new_img = Image.new("RGB", (new_image_width, img.height), "white")
        #     new_img.paste(img, (0, 0))
        #
        #     # Draw text onto the new image
        #     draw = ImageDraw.Draw(new_img)
        #     x_position = img.width + 20  # Start drawing to the right of the image
        #     y_position = 20  # Start drawing with some top margin
        #     for line in lines:
        #         draw.text((x_position, y_position), line, font=font, fill=(0, 0, 0))
        #         bbox = draw.textbbox((x_position, y_position), line, font=font)
        #         y_position += bbox[3] - bbox[1] + line_spacing  # Move to next line with spacing
        #
        # else:
        #     max_text_width = img.width - 40 # Margin
        #     text_height, lines = calculate_text_height_and_wrap(text, font, max_text_width, line_spacing)
        #
        #     # Create new image with space for text below
        #     new_image_height = img.height + text_height + 40
        #     new_img = Image.new("RGB", (img.width, new_image_height), "white")
        #     new_img.paste(img, (0, 0))
        #
        #     # Draw text onto the new image
        #     draw = ImageDraw.Draw(new_img)
        #     y_position = img.height + 20  # Start drawing below the image
        #     for line in lines:
        #         draw.text((20, y_position), line, font=font, fill=(0, 0, 0))
        #         bbox = draw.textbbox((20, y_position), line, font=font)
        #         y_position += bbox[3] - bbox[1] + line_spacing  # Move to next line with spacing

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, f'prompt_{base_name}.jpg')

        # Save the new image
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        img.save(output_path)

def main(prompt_path, images_txt_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = load_image_paths(images_txt_path)

    prompt = read_text_from_file(prompt_path)

    for img_index, image_path in enumerate(image_paths):
        print(img_index, image_path)
        combine_text_image(prompt, image_path, output_folder, 50)


if __name__ == "__main__":
    prompt_path = os.path.join(base_path, "opencv", "prompt.txt")
    images_txt_path = os.path.join(base_path, "test/Level_1/image_test_all.txt")
    output_folder = os.path.join(base_path, "gpt-4/test/")

    main(prompt_path, images_txt_path, output_folder)