import cv2
import pytesseract
import numpy as np
from PIL import Image
import mediapipe as mp

class AirWriter():
    def __init__(self, mode = False, maxHands = 2, modelComplexity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        
        self.paintWindow = np.zeros((471, 636, 3), dtype=np.uint8) + 255
        cv2.namedWindow('Air Paint', cv2.WINDOW_AUTOSIZE)


    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def recognize_text(self):
        # Use pytesseract to do OCR
        text = "pytesseract.image_to_string(self.paintWindow)"

        cv2.imwrite('paintWindow_image.png', self.paintWindow)

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        # Open the image file
        image = Image.open('paintWindow_image.png')

        # Use Tesseract OCR to extract text
        text = pytesseract.image_to_string(self.paintWindow)
        
        # Print the extracted text
        if text:
            print(f"Extracted Text: {text}")
            self.clearCanvas()

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])

            fingers = self.fingersUp()

            if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                self.clearCanvas()
                cv2.circle(img, (self.lmList[4][1], self.lmList[4][2]), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (self.lmList[8][1], self.lmList[8][2]), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (self.lmList[12][1], self.lmList[12][2]), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (self.lmList[16][1], self.lmList[16][2]), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (self.lmList[20][1], self.lmList[20][2]), 10, (0, 0, 255), cv2.FILLED)

            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                self.recognize_text()
                cv2.circle(img, (self.lmList[8][1], self.lmList[8][2]), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (self.lmList[12][1], self.lmList[12][2]), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (self.lmList[16][1], self.lmList[16][2]), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (self.lmList[20][1], self.lmList[20][2]), 10, (0, 255, 0), cv2.FILLED)

            elif fingers[0] == 1 and fingers[1] == 1:
                cv2.circle(self.paintWindow, (self.lmList[8][1], self.lmList[8][2]), 10, (0, 0, 0), cv2.FILLED)
                cv2.circle(img, (self.lmList[4][1], self.lmList[4][2]), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (self.lmList[8][1], self.lmList[8][2]), 10, (255, 0, 0), cv2.FILLED)

            return self.lmList

    def clearCanvas(self):
        self.paintWindow[:] = 255  

    def fingersUp(self):
        fingers = []
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers


def main():     

    cap = cv2.VideoCapture(0)
    detector = AirWriter()

    while True:             
        success, img = cap.read()         
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        cv2.imshow("Camera", img)
        cv2.imshow("Air Paint", detector.paintWindow)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()