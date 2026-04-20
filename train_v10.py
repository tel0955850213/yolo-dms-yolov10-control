from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 載入 YOLOv10 nano 模型 (對照組)
    model = YOLO("yolov10n.pt")

    # 2. 開始訓練！
    print("🚀 準備啟動 YOLOv10 訓練...")
    model.train(
        data="C:/Users/Lin/Desktop/yolov10/data.yaml", 
        epochs=4,           # 先跑 100 輪
        imgsz=640,            # 圖片大小 640
        batch=16,             # 每次看 16 張圖
        device=0,             # 預設使用你電腦裡的 NVIDIA 顯卡
        project="DMS_Thesis", # 會在你桌面自動建這個資料夾存結果
        name="YOLOv10_Control",
        workers=2             # Windows 建議設 2 或 4 比較穩
    )