from PIL import Image

def extract_lsb(image_path):
    # Wrap the filename in quotes
    img = Image.open(image_path)
    img = img.convert('RGB') # Ensures we have R, G, and B channels
    pixels = img.load()
    binary_data = ""
    
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            r, g, b = pixels[x, y]
            # Extract the LSB of each color channel
            binary_data += str(r & 1)
            binary_data += str(g & 1)
            binary_data += str(b & 1)
            
    # Convert binary to string
    all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    
    # Use a try-except to handle non-text data gracefully
    decoded_text = ""
    for byte in all_bytes:
        try:
            char = chr(int(byte, 2))
            # Optional: Stop at a null terminator if the message is short
            if char == '\x00': 
                break
            decoded_text += char
        except ValueError:
            continue
            
    return decoded_text

# Make sure to use the actual filename you downloaded
print(extract_lsb('ddd.png'))