import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Minimum Detection Confidence Threshold = 0.75 for optimal facial detection
detector = FaceDetector(minDetectionCon=0.75)

while True:
    success, image = cap.read()
    # Find faces in an image and return the bounding box information and draw around the detected face if draw=True
    image, boundingBoxes = detector.findFaces(image, draw=False)

    # If bounding box(es) is/are not empty/null
    if boundingBoxes:
        # Give different names for each available bounding box with 'i' as the unique identifier
        for i, boundingBox in enumerate(boundingBoxes):
            x, y, w, h = boundingBox["bbox"]
            # To avoid errors when an image is very close to the camera and gets out of bounds
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            # Cropping image with: 'y' as starting height and 'y + h' as ending height... Alternatively, 'x' is the
            # same for width
            croppedFace = image[y:y + h, x:x + w]
            # Blur cropped face(s), (35, 35) must be odd numbers, and it's the degree of blur needed on the face
            blurredFace = cv2.blur(croppedFace, (35, 35))
            image[y:y + h, x:x + w] = blurredFace  # Replace cropped face with the blurred to hide the detected face
            # Display detected cropped face(s)
            # cv2.imshow(f"Cropped face {i}", image)

    cv2.imshow("Image", image)
    cv2.waitKey(1)  # 1 millisecond delay
