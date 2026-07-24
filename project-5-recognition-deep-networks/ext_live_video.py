"""
    Drew Hill & Abhiram Banda
    CS5330 Pattern Recognition & Computer Vision

    ext_live_video.py

    Uses the trained model to read handwritten numbers from a live video stream 
    instead of from saved photos.
"""
import sys
import torch
import cv2
import numpy as np
from handwritten import load, convert

def get_center_frame(frame, size):
    """
        Gets a box from the middle of the video frame. This is where number gets
        read from.

        frame: numpy array - one frame of video.
        size: int - how wide the square should be.
        returns the top left corner and a greyscale box.
    """
    height, width = frame.shape[:2]

    # Finds the top left corner of the square in the middle of the frame.
    top = (height - size) // 2
    left = (width - size) // 2

    box = frame[top:top + size, left:left + size]
    grey_box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)

    return top, left, grey_box

def guess_number(model, grey_box):
    """
        Runs the number form the square in center of screen through the model and 
        gets a guess. Returns nothing if nothing is found.

        model: model - the trained model.
        grey_box: numpy array - the gray square from the middle of the frame.
        returns the guessed number, or none if the square looks blank.
    """
    # Skips blank squares.
    if grey_box.min() > 100:
        return None

    with torch.no_grad():
        tensor = convert(grey_box)
        out = model(tensor)

        guess = out.argmax(1).item()

    return guess


def run_video(model, size, source):
    """
        Opens the live video from mobile phone or webcam and reads numbers until 
        q is pressed. Draws a box on screen showing where to hold the number.

        model: model - the trained model.
        size: int - how wide the reading box should be.
        source: str - the address of the phone camera or 0 for webcam.
    """
    cap = cv2.VideoCapture(source)

    # Exits program if camera wont open.
    if not cap.isOpened():
        print(f"Could not open {source}")
        return

    numbers_saved = []

    while True:
        # Gets a single frame of video .
        worked, frame = cap.read()

        if not worked:
            print("Video stream not found")
            break

        # Rotates mobile frame
        if len(source) > 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        top, left, grey_box = get_center_frame(frame, size)
        guess = guess_number(model, grey_box)

        # Draws the box showing where to hold the number.
        cv2.rectangle(frame, (left, top), (left + size, top + size), (0, 255, 0), 2)

        # Writes the guess above the box.
        if guess is None:
            text = "Number Not Found"
            color = (0, 0, 255)
            font_size = .5

        else:
            text = f"Guess: {guess}"
            color = (0, 255, 0)
            font_size = 1.0

        cv2.putText(frame, text, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)

        cv2.imshow("Live number reader", frame)

      
        key = cv2.waitKey(1) & 0xFF

        # Exits program if user presses q and prints out how many numbers were saved.
        if key == ord("q"):
            arrr = np.array(numbers_saved)
            count = np.bincount(arrr, minlength=10)

            print("\nNumbers Saved: ")
            for number in range(10):
                print(f"{number}: {count[number]}\n")

            break

        # Saves a number.
        if key == ord('s'):
            numbers_saved.append(guess)
            print(f"{guess} saved\n")

    cap.release()
    cv2.destroyAllWindows()


def main(argv):
    """
        Loads the trained model and reads handwritten numbers from a video stream.
    """
    model = load("model_weights.pth")

    # Loads the phone stream address if one was given if not uses the webcam.
    if len(argv) > 1:
        src = f"http://{argv[1]}:8080/stream.mjpg"

    else:
        source = 0

    run_video(model, 200, src)
    return

if __name__ == "__main__":
    main(sys.argv)