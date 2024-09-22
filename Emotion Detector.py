import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model for emotion recognition
model = load_model("emotion_detection_model1.h5")

# Define the emotions labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
video_capture = cv2.VideoCapture(1)

while True:
    # Read the current frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region of interest
        roi = gray[y:y + h, x:x + w]

        # Resize the face ROI to match the input size of the model
        roi = cv2.resize(roi, (48, 48))

        # Normalize the face ROI
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)

        # Perform emotion recognition
        predictions = model.predict(roi)[0]

        # Get the predicted emotion label
        emotion_label = emotions[np.argmax(predictions)]

        # Draw the bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 100, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Emotion Recognizer', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()


