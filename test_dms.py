import cv2
import time
import json
import requests
import threading
from ultralytics import YOLO

# ==========================================
# 參數設定區
# ==========================================
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1480445491991806094/m2x-orlNQurasWk6cZ-Jq7HJSCMhZ081NQr5q6QQUZLMjRsGOgUGV3z0jZWt4LEapSn0"
MODEL_PATH = "best.pt"  # 填入你剛剛練好的模型檔名

# 觸發條件 (秒)
CONFIRM_TIME_SLEEP = 1.5  # 閉眼超過 1.5 秒判定為打瞌睡
CONFIRM_TIME_PHONE = 1.0  # 偵測到手機超過 1 秒判定為分心
COOLDOWN_TIME = 10.0      # 警報冷卻時間 (避免 Discord 被洗頻)

# 狀態追蹤變數
sleep_start_time = 0
phone_start_time = 0
last_alert_time = 0

# ==========================================
# Discord 推播模組
# ==========================================
def send_discord_alert(alert_type, frame):
    global last_alert_time
    
    if alert_type == "sleep":
        msg = "🚨 **【DMS 警報】駕駛打瞌睡！** 偵測到閉眼超過 1.5 秒，極度危險！"
    elif alert_type == "phone":
        msg = "⚠️ **【DMS 警報】駕駛分心！** 偵測到使用手機，請注意前方路況！"
    elif alert_type == "yawn":
        msg = "🥱 **【DMS 提示】駕駛疲勞！** 偵測到頻繁打呵欠，建議停車休息。"
    else:
        msg = "🚨 **【DMS 警報】偵測到異常駕駛行為！**"

    def _send():
        try:
            # 存一張截圖當作證據
            cv2.imwrite("dms_alert.jpg", frame)
            with open("dms_alert.jpg", "rb") as f:
                payload = {"content": msg}
                requests.post(DISCORD_WEBHOOK_URL, data={"payload_json": json.dumps(payload)}, files={"file": f}, timeout=10)
            print(f"✅ [Discord] {alert_type} 警報發送成功！")
        except Exception as e:
            print(f"❌ [Discord] 發送失敗: {e}")

    threading.Thread(target=_send).start()

# ==========================================
# 主程式
# ==========================================
def main():
    global sleep_start_time, phone_start_time, last_alert_time
    
    print("🧠 正在載入 DMS AI 模型...")
    model = YOLO(MODEL_PATH)
    
    print("📷 開啟筆電攝影機...")
    cap = cv2.VideoCapture(0)  # 0 代表筆電內建鏡頭

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 進行 AI 推論 (信心度設 0.5 過濾雜訊)
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot() # 畫上 Bounding Box
        
        # 當前影格的狀態旗標
        current_eyeclose = False
        current_phone = False
        current_yawn = False
        
        # 分析 AI 抓到的物件
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            if class_name == "eyeclose":
                current_eyeclose = True
            elif class_name == "phone":
                current_phone = True
            elif class_name == "yawn":
                current_yawn = True

        # --------------------------------------------------
        # 邏輯 1：打瞌睡判定 (閉眼計時)
        # --------------------------------------------------
        if current_eyeclose:
            if sleep_start_time == 0:
                sleep_start_time = time.time()
            elif time.time() - sleep_start_time >= CONFIRM_TIME_SLEEP:
                if time.time() - last_alert_time > COOLDOWN_TIME:
                    print("🚨 觸發警報：駕駛打瞌睡！")
                    send_discord_alert("sleep", annotated_frame)
                    last_alert_time = time.time()
        else:
            sleep_start_time = 0 # 眼睛一睜開，計時器馬上歸零！

        # --------------------------------------------------
        # 邏輯 2：分心判定 (手機計時)
        # --------------------------------------------------
        if current_phone:
            if phone_start_time == 0:
                phone_start_time = time.time()
            elif time.time() - phone_start_time >= CONFIRM_TIME_PHONE:
                if time.time() - last_alert_time > COOLDOWN_TIME:
                    print("⚠️ 觸發警報：駕駛使用手機！")
                    send_discord_alert("phone", annotated_frame)
                    last_alert_time = time.time()
        else:
            phone_start_time = 0

        # --------------------------------------------------
        # 邏輯 3：打呵欠判定 (直接觸發)
        # --------------------------------------------------
        if current_yawn:
            if time.time() - last_alert_time > COOLDOWN_TIME:
                print("🥱 觸發警報：駕駛打呵欠！")
                send_discord_alert("yawn", annotated_frame)
                last_alert_time = time.time()

        # UI 顯示計時狀態 (方便你 Debug)
        if sleep_start_time > 0:
            cv2.putText(annotated_frame, f"Closing Eye: {time.time()-sleep_start_time:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if phone_start_time > 0:
            cv2.putText(annotated_frame, f"Phone detected: {time.time()-phone_start_time:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        cv2.imshow("DMS Camera Test", annotated_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()