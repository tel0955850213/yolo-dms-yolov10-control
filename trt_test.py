from ultralytics import YOLO
import cv2
import time

# 載入剛剛轉好的 TensorRT 引擎
model = YOLO("best.engine", task="detect") 

# 開啟 USB 相機 (0 通常是預設的 USB 鏡頭)
cap = cv2.VideoCapture(0)

print("啟動 TensorRT 推理...")

while cap.isOpened():
    success, frame = cap.read()
    if success:
        start_time = time.time()
        
        # 執行推理 (加上 FP16 半精度加速)
        results = model(frame, conf=0.5, half=True)
        
        # 計算 FPS
        fps = 1.0 / (time.time() - start_time)
        
        # 繪製結果與 FPS
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("TensorRT Rescue Monitor", annotated_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
