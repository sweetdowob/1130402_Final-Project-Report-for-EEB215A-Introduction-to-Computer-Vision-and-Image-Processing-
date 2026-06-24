import cv2
from ultralytics import YOLO


model = YOLO("best.pt")
# 數值決定哪台攝影機
cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
    print("攝影機打開失敗，請檢查權限或連線。")
    exit()

# 開始偵測
while True:
    success, frame = cap.read()
    # 防呆
    if success == False:
        break

    # conf代表只抓取信心分數大於conf值的消防栓
    results = model.predict(frame, conf=0.5, show=False)
    result_frame = results[0].plot()

    # 顯示畫面
    cv2.imshow("Live Demo (按Q可以立刻關閉)", result_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()