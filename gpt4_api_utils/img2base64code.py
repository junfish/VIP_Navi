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
image_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/20231219_170815_proj/DJI_20231219_170815/DJI_20231219_170815_frame_000.6s.jpg"
# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
         "type": "text", "text": "Please help me to annotate images with natural language descriptions for vision impaired people (VIPs). It is important to be as descriptive and clear as possible. For each image, please give me two versions (concise & detailed version) of descriptions for two VIP groups (VIPs who are early-blind & late-blind), respectively."
                                 "My requirements of two versions are:"
                                 ""
                                 "(a) Concise version contains only one sentence (no more than 20 words). Use specific nouns and adjectives to give environment summarization, such as “a quiet lab where some people are reading books.”"
                                 "(b) Detailed version contains the descriptions of three different levels, as outlined below. Each level represents a progressively higher need within the visually impaired population. Please ensure that the information for each level is contained within its own separate paragraph."
                                 ""
                                 "  Level 1. Spatial information needs (perceptual insight cravings). VIPs need two types of spatial information: shape information and layout information, to get a high-level overview of a space. Your descriptions should help VIPs gather shape and layout information in a manner that facilitates active engagement with the environment."
                                 "  Level 2. Independent exploration needs (self-directed learning desires). VIPs face difficulties in making navigation decisions based on spatial information collected via their non-visual senses and your descriptions. Additionally, acquiring appropriate orientation and mobility (O&M) training and maintaining their O&M skills can be a struggle for them, which negatively impacting their confidence to explore independently. You should try your best to afford VIPs precise and reliable spatial information and should ensure that VIPs make accurate navigation decisions based on this information. You can serve as an O&M education assistance tool to give VIPs the confidence to explore unfamiliar environments independently."
                                 "  Level 3. Collaborative exploration needs (social interaction needs). Social pressures pose a major challenge to VIPs receiving help and to non-VIPs providing help when exploring environments collaboratively. VIPs want verbal assistance in a comprehensible format when exploring collaboratively but find it challenging to communicate this preference to others. You should normalize VIPs’ exploration behaviors by introducing social norms for exploration assistance in the current situation. If possible, you could scaffold and facilitate collaborative exploration by telling VIPs how to distribute requests for collaboration and what kind of translations may be needed for VIPs."
                                 ""
                                 "Here are some tips for you to refine my requirements above:"
                                 ""
                                 "1. Object Identification and Description: Clearly identify each object in the image. Use specific nouns and adjectives to describe the objects, such as “round red apple” or “high laboratory shelves.”"
                                 "2. Spatial Relationships: Describe the spatial relationships between objects. Use terms like “to the left of,” “behind,” “in front of,” “next to,” and “above” to explain how objects are situated in relation to each other."
                                 "3. Approximate Measurements: Provide approximate measurements to help the listener understand distances. Use units like steps, arm’s length, or common objects as references, for example, “about three steps away” or “an arm’s length apart.”"
                                 "4. Textures and Surfaces: Mention the texture or surface of objects if relevant. For instance, “the desk has a smooth, laminate surface” or “the corridor is lined with polished tiles.”"
                                 "5. Colors and Contrasts: Even though the individual may not perceive colors, describing them can help build a richer picture or convey information to those who have partial vision or can perceive some colors. Contrasts can also be useful, like “the bright fluorescent lights against the neutral-colored ceiling.”"
                                 "6. Layout Description: Offer a general layout of the scene. For example, “The room is rectangular, with a door on the shorter wall and a window on the longer wall.”"
                                 "7. Directional Orientation: Use cardinal directions if the person is familiar with them, like “the conference room is to the east,” or relative directions like “the sun sets in front of you when you face the window.”"
                                 "8. Consistency: Keep descriptions consistent throughout the annotations. If you start by describing objects from left to right, maintain that order. You can also start by describing the immediate area, followed by objects in the middle distance, and then far away items."
                                 "9. Interactive Elements: Point out any interactive elements or objects that can be used, such as “a button to press at the crosswalk” or “to your right, there’s a window that can be opened for fresh air.”"
                                 "10. Avoid Subjectivity: Keep descriptions objective and avoid personal interpretations or non-evident assumptions about the scene. When uncertain about quantities or functionalities, employ phrases such as “Some chairs feature a green accent” or “these numbered, circular floor markers with arrows possibly serve for wayfinding or organizing seating.”"
                                 "11. Clarity and Brevity: Be clear and concise. While detail is important, unnecessary information can clutter the mental image. For instance, instead of “a rectangular, mahogany coffee table with intricate carvings,” use “a long, wooden coffee table in front of the couch.”"
                                 "12. Safety and Navigation Cues: Highlight potential obstacles or safety hazards (like low-hanging objects, uneven surfaces) and include navigational cues if possible. Use sentences like “Please be careful of the small step down as you enter the room. There’s a low coffee table in the center, about two feet high.” or “If you keep walking straight for about six steps, you’ll reach the kitchen door. You can feel a wooden chair right before the entrance.”"
                                 ""
                                 "By considering these tips, you can create descriptions in natural language that are informative and useful for building a mental map for VIPs."
                                 ""
                                 "Requirements of two groups of VIPs are:"
                                 ""
                                 "(a) Early-blind VIPs are a group of people who are blind by birth or developed vision impairments early in life. They have learned to trust their non-visual senses to collect spatial information over time, and have no concept or sense of colors. Please describe using texture or tactile feel, such as “the bench is crafted from unvarnished timber, giving it a textured, coarse feel,” rather than using color or contrast descriptions like “an aisle carpeted in dark gray.”"
                                 "(b) Late-blind VIPs are a group of people who are blind after they have a basic understanding of our world; they know things similar to normal people. They prefer receive various information in the descriptions with as much visual detail as possible."
                                 ""
                                 "Below are specialized tips for early-blind VIPs. Prioritize these over the initial 12 tips, as they align more closely with the unique experiences of early-blind VIPs. In cases of conflicting advice, please give these tips precedence:"
                                 ""
                                 "1. Emphasize Non-Visual Senses: Since early-blind individuals may have limited or no visual memories, focus on descriptions that leverage other senses like touch, sound, smell, and taste based on the visuals in image. For example, “the carpet underfoot looks thick and soft, with a wavy texture; the walls seem to be smooth and cool to the touch.”"
                                 "2. Spatial Layout and Orientation: Provide detailed information about the layout and orientation of spaces and objects, as this group relies heavily on spatial awareness for navigation and understanding their environment. Use language like “walk straight for about ten steps to reach the sofa.”"
                                 "3. Texture and Shape Over Color: Describe the texture and shape of objects, as these will be more meaningful than color descriptions, such as “the shape of the object in front of you is similar to a large loaf of bread.”"
                                 "4. Consistent Terminology: Use consistent terminology and reference points to avoid confusion, as early-blind individuals may have their unique ways of conceptualizing their environment. If you use steps as a measurement, as in “the table is four steps away from you,” continue using steps for other distances instead of suddenly switching to feet or meters."
                                 "5. Contextual Descriptions: Offer context for objects and environments that may not be immediately apparent, like the purpose of a certain object or the typical activities in a space. For example, “you are now in an office, a place for work and study; the object with a flat top and drawers you can feel is a desk, used for writing and placing computers.”"
                                 ""
                                 "For each results of four descriptions (i.e., concise & detailed versions for early-blind & late-blind VIPs, respectively), please give me hints to distinguish them in your generated text."
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
  "max_tokens": 3000
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())