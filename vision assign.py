import cv2

# Load the face cascade classifier with the correct path
face_cascade_path = r'C:\Users\Administrator\Downloads\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise IOError(f"Could not load Haarcascade xml file from path: {face_cascade_path}")

# Path to the image
start = r'C:\Users\Administrator\Downloads\image vision.jpg'

# Read the image
img = cv2.imread(start)
if img is None:
    raise IOError(f"Could not read the image file from path: {start}")

# Convert the image to grayscale
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(grey, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)

# Display the image with rectangles around detected faces
cv2.imshow('img', img)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 800, 600)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Path to the image for DeepFace analysis
pathimg = r'C:\Users\Administrator\Downloads\image vision.jpg'

# Analyze gender and age using DeepFace
try:
    result_gender = DeepFace.analyze(pathimg, actions=['gender'])
    gender = result_gender['gender']
    print(f"Predicted gender is: {gender}")

    result_age = DeepFace.analyze(pathimg, actions=['age'])
    age = result_age['age']
    print(f"Predicted age is: {age} years")
except Exception as e:
    print(f"An error occurred: {e}")
