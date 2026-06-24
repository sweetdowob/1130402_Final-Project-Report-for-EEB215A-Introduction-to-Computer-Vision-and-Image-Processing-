import cv2
from ultralytics import YOLO


video_name = "影片放這"
output = "result.mp4"
model = YOLO("best.pt")
cap = cv2.VideoCapture(video_name)

if cap.isOpened() == False:
    print("影片打開失敗，請檢查檔名或路徑。")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, fps, (width, height))

#開始偵測
while True:
    success, frame = cap.read()
    # 防呆
    if success == False:
        break
    
    # conf代表只抓取信心分數大於conf值的消防栓
    results = model.predict(frame, conf=0.5)
    result_frame = results[0].plot()
    out.write(result_frame)

    cv2.imshow("畫面(按Q可以立刻關閉)", result_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()