import cv2
import mediapipe as mp
# img = cv2.imread('D:/DeepLearning/src/images_Object_Detection/shape.jpg')
# cap = cv2.VideoCapture('D:/DeepLearning/src/images_Object_Detection/night.mp4')
# cap = cv2.VideoCapture('hand2.mp4')
# mh = mp.solutions.hands
# hands = mh.Hands()
# mpdraw = mp.solutions.drawing_utils
# while True:
#     ret, img = cap.read()
#     if ret:
#         imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         result = hands.process(imgRGB)
#         print(result.multi_hand_landmarks)
#         cv2.imshow('img',img)
#         if result.multi_hand_landmarks:
#             for handlms in result.multi_hand_landmarks:
#                 mpdraw.draw_landmarks(img,handlms)
#     if cv2.waitKey(1) == ord('q'):
#         break

# while True:
#     ret, frame = cap.read()#ret 是否成功取得下一偵 frame下一偵圖片
#     # frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
#     # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# #     frame = cv2.GaussianBlur(frame,(3,3),0)
#     if ret:
#         cv2.imshow('video',frame)
#     else:
#         break
#     if cv2.waitKey(10) == ord('q'):
#         break
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('hand2.mp4')
mpHands = mp.solutions.hands #使用手追蹤的模型
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)#呼叫模型
mpDraw = mp.solutions.drawing_utils #畫手的landmark
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)#設定細節
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)
pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB) #執行模型

        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                #第三個參數是要把點和點連接起來
                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)

                    # cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

                    # if i == 4:
                    #     cv2.circle(img, (xPos, yPos), 20, (166, 56, 56), cv2.FILLED)
                    # print(i, xPos, yPos)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break