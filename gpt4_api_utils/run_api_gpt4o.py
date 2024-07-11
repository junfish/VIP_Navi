from openai import OpenAI
'''
print("Please give me **three** paragraphs of scene descriptions. \n"
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
'''

import base64
import requests

# OpenAI API Key
api_key = "sk-proj-B8L08U3CKcQULN3E4b71T3BlbkFJRRaGw1TeUdF09Oh13qpK"
# "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"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/gpt-4/Basement/prompt_DJI_20240103_121558_frame_059.5s.jpg"
# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o", # gpt-4-vision-preview
  "messages": [
    {
      "role": "user",
      "content": [
        {
         "type": "text", "text": "Please give me **three** paragraphs of scene descriptions. \n"
                                 "\n"
                                 "1. The first description should be concise and within 15 words. \n"
                                 "\n"
                                 "2. The second description should be detailed and 150 words, and generated for people with vision impairments (low vision or late blindness). Please follow these rules: \n"
                                 "  (a). *Use clear and concise language*: Choose words carefully to provide clear, concise descriptions, using descriptive adjectives and adverbs for relevant information. \n"
                                 "  (b). *Provide directional and distance information*: Include information about the layout of the space, who and where any people are, and what they are doing. Give a reliable description of how to navigate the space, describing any safety hazards in detail. Include information on the direction and distance of points of interest. Use clock face references (e.g., \"to your right at 3 o\'clock\") to give a sense of orientation. Note that the words \"left\" and \"right\" are denoted on the image as a reference. Use common reference objects to describe distances, sizes, and other measurements. \n"
                                 "  (c). *Describe surroundings in stages*: Detail the environment in sections, starting with the immediate area, then describing nearby objects or obstacles, and finally providing information about the destination or the route ahead. \n"
                                 "\n"
                                 "3. The third description should also be detailed and 150 words, but generated for people who have been blind since birth. Please follow the same rules as the second description, with one additional consideration: \n"
                                 "  (d). *Avoid describing color information*: Do not include color details that are difficult to imagine for a user who has been blind since birth."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 1000
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())
print("\n")
print(response.json()["choices"][0]["message"]["content"])