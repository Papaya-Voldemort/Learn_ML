import cv2
import subprocess

# 1. Choose your custom characters for jp2a!
# From darkest/shadows to lightest/highlights
CUSTOM_CHARS = " .:-=+*#%@"

# Set the grid width (More columns = a sharper, higher-resolution face)
ASCII_WIDTH = 120 

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Live jp2a ASCII Camera Active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for a natural mirror effect
    frame = cv2.flip(frame, 1)

    # Encode the raw camera frame into a JPEG image inside computer memory (RAM)
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        continue
    
    # Convert encoded image to a stream of bytes that jp2a can read
    jpeg_bytes = encoded_image.tobytes()

    # Build the jp2a command line arguments
    # We pass '-' at the end to tell jp2a to read from the piped byte stream
    jp2a_command = [
        "jp2a",
        f"--width={ASCII_WIDTH}",
        f"--chars={CUSTOM_CHARS}",
        "--background=dark", # Change to light if your terminal has a white background
        "-"
    ]

    try:
        # Run jp2a as a sub-process and pipe the JPEG data straight into it
        process = subprocess.Popen(
            jp2a_command, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Send the image bytes and capture the text output
        stdout, _ = process.communicate(input=jpeg_bytes)
        ascii_frame = stdout.decode('utf-8')

        # Clear the terminal window and print the new frame instantly
        print("\033[H" + ascii_frame, end="")

    except FileNotFoundError:
        print("\nError: jp2a is not installed on your system or not in your PATH.")
        break

    # Standard exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nCamera turned off.")
