import os
import base64
import requests
import csv
import logging
import random
# from imgcat import imgcat

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
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}","detail": "low"}}
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
            # Extract token information
            if 'usage' in response_json:
                total_tokens = response_json['usage'].get('total_tokens', 'N/A')
                prompt_tokens = response_json['usage'].get('prompt_tokens', 'N/A')
                completion_tokens = response_json['usage'].get('completion_tokens', 'N/A')

                # Log token usage
                logging.info(
                    f"Tokens used - Total: {total_tokens}, Prompt: {prompt_tokens}, Completion: {completion_tokens}")
            return caption

    except requests.RequestException as e:
        print(f"API request failed: {e}")
    return "Failed to get caption"


def process_images(input_path, output_csv, api_key, batch_size=100, start_id_range=(1000, 10000)):
    # Load already processed images and determine the next image ID
    processed_images = set()
    max_image_id = 0
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
                    try:
                        image_id = int(row[0])
                        if image_id > max_image_id:
                            max_image_id = image_id
                    except ValueError:
                        pass

    # Generate a random starting ID within the specified range
    if max_image_id == 0:
        next_image_id = random.randint(*start_id_range)
    else:
        next_image_id = max_image_id + 1

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
            writer.writerow(['image_id', 'image_file', 'caption'])  # Write header row if the file is empty

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
                    writer.writerow([next_image_id, image_file_name, caption])
                    processed_images.add(image_file_name)  # Add to the set of processed images
                    next_image_id += 1  # Increment the image ID for the next image
                except Exception as e:
                    logging.error(f"Error processing {image_path}: {e}")
                    writer.writerow([f"Error processing image {next_image_id}", image_file_name])
                    next_image_id += 1  # Increment the image ID even on error to maintain unique IDs

            # Flush the writer buffer to the disk after each batch
            file.flush()
            os.fsync(file.fileno())


# Example usage
# Configure logging to write to a file

# floor_name = 'Basement'
# floor_name = 'Lower_Level'
# floor_name = 'Level_1'
floor_name = 'Level_1'

log_file_path = '/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/gpt-4/test/processing_test_'+ floor_name + '.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_path = '/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/gpt-4/test/' + floor_name
output_csv = '/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/gpt-4/test/' + floor_name + '_output.csv'
# api_key_yifan = 'sk-proj-Aqze1HM6nH9Sak2cpJT1T3BlbkFJJCZIRYESwnocXxUp621F'
# api_key_jun = 'sk-proj-B8L08U3CKcQULN3E4b71T3BlbkFJRRaGw1TeUdF09Oh13qpK'
# api_key_lower = 'sk-proj-VuKk0sV4hmtKUV6L2o8iT3BlbkFJyPGB7HNwiIhuED22WQNs'
api_key_level_1 = 'sk-proj-NLy3EYyCdMzARuPP64YZT3BlbkFJetSp34k9jKE0kjTnd4lS'
# api_key_level_2 = 'sk-proj-xDAWPqh6xVI86fnhEzQAT3BlbkFJWuaPR5hhN63nHiU6vXEZ'
process_images(input_path, output_csv, api_key_level_1)
