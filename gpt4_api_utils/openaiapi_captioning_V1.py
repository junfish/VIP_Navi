import os
import base64
import requests
import csv
import logging


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_caption(base64_image, api_key):
    # Base prompt for direct description
    # custom_prompt = ("I need three paragraphs of scene descriptions. The first paragraph title is Concise. The second paragraph title is DetailedForLateBlind. The third paragraph title is DetailedForEarlyBlind. \n "
    #                  "1. The first description should be concise and within 15 words.\n "
    #                  "2. The second description should be detailed and 150 words, and generated for people with vision impairments (low vision or late blindness). Follow these rules:\n "
    #                  "a) Use clear and concise language: Choose words carefully to provide clear, concise descriptions, using descriptive adjectives and adverbs for relevant information.\n "
    #                  "b) Provide directional and distance information: Include information about the layout of the space, who and where any people are, and what they are doing. Give a reliable description of how to navigate the space, "
    #                  "describing any safety hazards in detail. Include information on the direction and distance of points of interest. Use clock face references (e.g., “to your right at 3 o’clock”) to give a sense of orientation. "
    #                  "Note that the words left and right are denoted on the image as a reference. Use common reference objects to describe distances, sizes, and other measurements.\n "
    #                  "c) Describe surroundings in stages: Detail the environment in sections, starting with the immediate area, then describing nearby objects or obstacles, and finally providing information about the destination or the route ahead.\n "
    #                  "3. The third description should also be detailed and 150 words, but generated for people who have been blind since birth. Follow the same rules as the second description, with one additional consideration:\n "
    #                  "d) Avoid describing color information: Do not include color details that are difficult to imagine for a user who has been blind since birth.\n")

    custom_prompt = ("Please give me **three** paragraphs of scene descriptions. The first paragraph title is Concise. The second paragraph title is DetailedForLateBlind. The third paragraph title is DetailedForEarlyBlind. \n"
                     "\n"
                     "1. The first description should be concise and within 15 words. \n"
                     "\n"
                     "2. The second description should be detailed and 150 words, and generated for people with vision impairments (low vision or late blindness). Please follow these rules: \n"
                     "  (a). *Use clear and concise language*: Choose words carefully to provide clear, concise descriptions, using descriptive adjectives and adverbs for relevant information. \n"
                     "  (b). *Provide directional and distance information*: Include information about the layout of the space, who and where any people are, and what they are doing. Give a reliable description of how to navigate the space, describing any safety hazards in detail. Include information on the direction and distance of points of interest. Use clock face references (e.g., \"to your right at 3 o\'clock\") to give a sense of orientation. Note that the words \"left\" and \"right\" are denoted on the image as a reference. Use common reference objects to describe distances, sizes, and other measurements. \n"
                     "  (c). *Describe surroundings in stages*: Detail the environment in sections, starting with the immediate area, then describing nearby objects or obstacles, and finally providing information about the destination or the route ahead. \n"
                     "\n"
                     "3. The third description should also be detailed and 150 words, but generated for people who have been blind since birth. Please follow the same rules as the second description, with one additional consideration: \n"
                     "  (d). *Avoid describing color information*: Do not include color details that are difficult to imagine for a user who has been blind since birth.")

    print(custom_prompt)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": custom_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        # The max_tokens parameter only applies to the output tokens.
        "max_tokens": 1000
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        if 'choices' in response_json and response_json['choices'] and 'message' in response_json['choices'][0]:
            caption = response_json['choices'][0]['message']['content'].strip()
            return caption
    except requests.RequestException as e:
        print(f"API request failed: {e}")
    return "Failed to get caption"


def process_images(input_path, output_csv, api_key, batch_size=100):
    # Load already processed images
    processed_images = set()
    if os.path.exists(output_csv):
        with open(output_csv, mode='r', newline='') as file:
            reader = csv.reader(file)
            try:
                next(reader)  # Skip header row
            except StopIteration:
                pass  # Handle the case where the CSV is empty

            for row in reader:
                if row and len(row) > 1:  # Ensure the row has enough columns
                    processed_images.add(row[1])  # Add image file name to the set

    directory_to_process = input_path
    all_files = []
    for root, _, files in os.walk(directory_to_process):
        for file_name in filter(lambda f: f.lower().endswith(('.png', '.jpg', '.jpeg')), files):
            all_files.append(os.path.join(root, file_name))

    total_files = len(all_files)
    logging.info(f"Total images to process: {total_files}")

    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.path.getsize(output_csv) == 0:
            writer.writerow(['caption', 'image_file'])  # Write header row if the file is empty

        for i in range(0, total_files, batch_size):
            batch_files = all_files[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1}/{(total_files + batch_size - 1) // batch_size}")

            for image_path in batch_files:
                image_file_name = os.path.basename(image_path)
                if image_file_name in processed_images:
                    logging.info(f"Skipping already processed image: {image_file_name}")
                    continue
                try:
                    base64_image = encode_image(image_path)
                    caption = get_caption(base64_image, api_key)
                    caption = caption.replace('"', '')
                    if caption == "Failed to get caption":
                        logging.info(f"Failed: {image_file_name}")
                    else:
                        logging.info(f"Processed: {image_file_name}")
                    writer.writerow([caption, image_file_name])
                    processed_images.add(image_file_name)  # Add to the set of processed images
                except Exception as e:
                    logging.error(f"Error processing {image_path}: {e}")
                    writer.writerow(["Error processing image", image_file_name])

            # Flush the writer buffer to the disk after each batch
            file.flush()
            os.fsync(file.fileno())


# Example usage
# Configure logging to write to a file
log_file_path = '/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/processing.log'
logging.basicConfig(filename = log_file_path, level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_path = '/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/gpt-4/Basement'
output_csv = '/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/gpt-4/Basement_output.csv'
api_key = 'sk-proj-Aqze1HM6nH9Sak2cpJT1T3BlbkFJJCZIRYESwnocXxUp621F'
process_images(input_path, output_csv, api_key)
