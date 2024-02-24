import cv2
import mediapipe as mp
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

def initialize_hands_model():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.75, min_tracking_confidence=0.75, max_num_hands=2)
    return hands

def initialize_drawing_utility():
    mpDraw = mp.solutions.drawing_utils
    return mpDraw

def start_webcam():
    cap = cv2.VideoCapture(0)
    return cap

def initialize_webdriver():
    driver = webdriver.Firefox()  # replace with the path to your WebDriver if necessary
    driver.get("https://youtu.be/XKHEtdqhLK8")  # replace with your YouTube video URL
    return driver

def process_image(hands, img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    return results

def draw_landmarks(mpDraw, img, handLms):
    mpDraw.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)

def get_finger_positions(handLms):
    index_tip = handLms.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
    thumb_tip = handLms.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y
    return index_tip, thumb_tip

def is_hand_open(index_tip, thumb_tip):
    return index_tip < thumb_tip

def play_or_pause_video(driver, is_open, prev_open):
    if prev_open != is_open:
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.SPACE)
    return is_open

def get_index_tip_x(handLms):
    return handLms.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x

def detect_swipe(driver, index_tip_x, prev_index_tip_x, last_swipe_time):
    if prev_index_tip_x is not None and time.time() - last_swipe_time > 1:  # 1 second cooldown
        if index_tip_x - prev_index_tip_x > 0.1:
            # Swipe right detected!
            # Rewind the video
            driver.find_element(By.TAG_NAME, 'body').send_keys('j')
            last_swipe_time = time.time()
        elif prev_index_tip_x - index_tip_x > 0.1:
            # Swipe left detected!
            # Fast forward the video
            driver.find_element(By.TAG_NAME, 'body').send_keys('l')
            last_swipe_time = time.time()
    return last_swipe_time

def main():
    hands = initialize_hands_model()
    mpDraw = initialize_drawing_utility()
    cap = start_webcam()
    driver = initialize_webdriver()

    prev_open = False
    prev_index_tip_x = None
    last_swipe_time = 0  # Time when the last swipe was detected

    while True:
        # Read frame from the webcam
        success, img = cap.read()

        # Process the image and get the hand landmarks
        results = process_image(hands, img)

        # Check if any hand is detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Draw hand landmarks on the image
                draw_landmarks(mpDraw, img, handLms)

                # Get the position of index finger tip and thumb tip
                index_tip, thumb_tip = get_finger_positions(handLms)

                # Check if the hand is open or closed
                is_open = is_hand_open(index_tip, thumb_tip)

                # Check if the hand just opened or closed
                prev_open = play_or_pause_video(driver, is_open, prev_open)

                # Get the position of index finger tip in each frame
                index_tip_x = get_index_tip_x(handLms)

                # Check if the index finger tip has moved significantly
                last_swipe_time = detect_swipe(driver, index_tip_x, prev_index_tip_x, last_swipe_time)

                prev_index_tip_x = index_tip_x

        # Display the image
        cv2.imshow("Image", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()