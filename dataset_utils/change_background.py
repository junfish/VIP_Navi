from PIL import Image

def make_background_transparent(image_path, output_path, background_color):
    # Load the image
    img = Image.open(image_path)
    img = img.convert("RGBA")  # Convert to RGBA to add alpha channel

    # Get pixel data
    data = img.getdata()

    # Replace specified color with a transparent background
    new_data = []
    for item in data:
        # print(item[:3])
        # Change all white (also consider other colors) pixels to transparent
        if item[:3][0] >= background_color[0] and item[:3][1] >= background_color[1] and item[:3][2] >= background_color[2]:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    # Update image data
    img.putdata(new_data)

    # Save the modified image
    img.save(output_path, "PNG")

# Example usage
image_path = '/Users/jasonyu/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/NaVIP/poster/cdf_orientation.png'
output_path = '/Users/jasonyu/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/NaVIP/poster/cdf_orientation_trans.png'
background_color = (240, 240, 240)  # Change this to the background color of your image
make_background_transparent(image_path, output_path, background_color)