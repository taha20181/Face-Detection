import cv2

# class for feature detection using Haar Cascades
class Detection:

    # class Constructor defining cascades
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml") # face cascade
        self.eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml") # eye cascade
        self.smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml") # smile cascade

    # function to detect face and features from an Image
    def fromImage(self):

        # read image
        image = cv2.imread("human.jpg")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detecting faces and features by following lines
        faces = self.face_cascade.detectMultiScale(gray_image, 1.1, 5)
        eyes = self.eye_cascade.detectMultiScale(gray_image, 1.1, 5)
        smiles = self.smile_cascade.detectMultiScale(gray_image, 1.5, 25)

        # drawing bounding boxes around found faces
        for x, y, w, h in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # drawing bounding boxes around found eyes
        for x, y, w, h in eyes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # drawing bounding boxes around found smile
        for x, y, w, h in smiles:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Detection Screen", image) # Display image

        cv2.waitKey()
        cv2.destroyAllWindows()

    

    # function to detect face and feature from Video stream
    def fromVideo(self):
        video = cv2.VideoCapture(0)
        while True:
            x, frame = video.read()

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            # detecting faces and features by following lines
            face = self.face_cascade.detectMultiScale(frame, 1.1, 5)
            eyes = self.eye_cascade.detectMultiScale(gray_image, 1.1, 5)
            smiles = self.smile_cascade.detectMultiScale(gray_image, 1.5, 25)

            # drawing bounding boxes around found faces from video frame by frame
            for x, y, w, h in face:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # drawing bounding boxes around found eyes from the video frame by frame
            for x, y, w, h in eyes:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # drawing bounding boxes around found smile from the video frame by frame
            for x, y, w, h in smiles:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Video', frame) # displaying frames

            if cv2.waitKey(25) & 0xFF == ord('q'): # stop by pressing 'q' on the keyboard
                break

        video.release() # video released, stop capturing
        cv2.destroyAllWindows()


# creating class instance
detect = Detection()

# calling the function
detect.fromImage()
