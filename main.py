import cv2
import mediapipe
import pyautogui
import math
import time

face_mesh_landmarks = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Specify the desired window width and height
window_width = 800
window_height = 600

# Open the default camera (try device 10 first, then fallback to 0)
cap = cv2.VideoCapture(10)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Direction key control variables
last_key_time = 0
key_cooldown = 0.3  # 防止按键过快，300ms冷却时间

# While loop to continuously get frames from the camera
while True:
    ret, image = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break 

    image = cv2.flip(image, 1)
    window_h, window_w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = face_mesh_landmarks.process(rgb_image)
    all_face_landmark_points = processed_image.multi_face_landmarks
    
    if all_face_landmark_points:
        one_face_landmark_points = all_face_landmark_points[0].landmark
        
        # Left eye blink detection (commented out)
        # left_eye = [one_face_landmark_points[145], one_face_landmark_points[159]]
        # for landmark_point in left_eye:
        #     x = int(landmark_point.x * window_w)
        #     y = int(landmark_point.y * window_h)
        #     cv2.circle(image, (x, y), 6, (0, 255, 0))

        # if (left_eye[0].y - left_eye[1].y) < 0.015:
        #     pyautogui.click()
        #     pyautogui.sleep(0.5)
        #     print("Mouse Clicked")

        # ===== 左眼方向键控制 =====
        # 获取左眼关键点
        iris_center = one_face_landmark_points[473]  # 左眼瞳孔中心
        eye_left = one_face_landmark_points[263]     # 左眼外眼角
        eye_right = one_face_landmark_points[362]    # 左眼内眼角

        # 计算眼睛宽度
        eye_width = math.sqrt((eye_right.x - eye_left.x)**2 + (eye_right.y - eye_left.y)**2)

        # 计算眼睛的角度（263到362的连线角度）
        angle = math.atan2(eye_right.y - eye_left.y, eye_right.x - eye_left.x)

        # 计算眼睛中心点（水平和垂直）
        eye_center_x = (eye_left.x + eye_right.x) / 2
        eye_center_y = (eye_left.y + eye_right.y) / 2

        # 根据你的设计：虚拟眼睛高度 = 眼睛宽度的一半
        eye_height = eye_width / 2

        # 计算旋转后的坐标系中瞳孔的位置
        # 将瞳孔坐标转换到以眼睛中心为原点、沿眼睛角度的坐标系
        dx = iris_center.x - eye_center_x
        dy = iris_center.y - eye_center_y

        # 旋转到眼睛的局部坐标系
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a

        # 计算瞳孔相对位置（归一化到 0-1）
        horizontal_ratio = (local_x + eye_width / 2) / eye_width if eye_width > 0 else 0.5
        vertical_ratio = (local_y + eye_height / 2) / eye_height if eye_height > 0 else 0.5

        # 可视化调试（绘制关键点和旋转的边界框）
        left_x = int(eye_left.x * window_w)
        left_y = int(eye_left.y * window_h)
        right_x = int(eye_right.x * window_w)
        right_y = int(eye_right.y * window_h)
        center_x = int(eye_center_x * window_w)
        center_y = int(eye_center_y * window_h)

        # 绘制眼角点 263 和 362（红色）
        cv2.circle(image, (left_x, left_y), 5, (0, 0, 255), -1)  # 263 外眼角
        cv2.circle(image, (right_x, right_y), 5, (0, 0, 255), -1)  # 362 内眼角

        # 计算旋转矩形的4个顶点
        # 矩形中心在眼睛中心，宽度是eye_width，高度是eye_height
        half_width = eye_width / 2
        half_height = eye_height / 2

        # 在局部坐标系中的4个顶点（未旋转）
        corners_local = [
            (-half_width, -half_height),  # 左上
            (half_width, -half_height),   # 右上
            (half_width, half_height),    # 右下
            (-half_width, half_height)    # 左下
        ]

        # 旋转并转换到图像坐标
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        corners_image = []
        for lx, ly in corners_local:
            # 旋转
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            # 转换到图像坐标
            ix = int((eye_center_x + rx) * window_w)
            iy = int((eye_center_y + ry) * window_h)
            corners_image.append((ix, iy))

        # 绘制旋转的矩形框（黄色）
        for i in range(4):
            cv2.line(image, corners_image[i], corners_image[(i + 1) % 4], (255, 255, 0), 2)

        # 绘制虚拟 eye_top 点（矩形上边的中点，红色）
        top_center_x = int((eye_center_x - half_height * sin_a) * window_w)
        top_center_y = int((eye_center_y + half_height * cos_a) * window_h)
        cv2.circle(image, (top_center_x, top_center_y), 5, (0, 0, 255), -1)  # eye_top

        # 绘制瞳孔中心（黄色）
        iris_x = int(iris_center.x * window_w)
        iris_y = int(iris_center.y * window_h)
        cv2.circle(image, (iris_x, iris_y), 3, (0, 255, 255), -1)

        # 方向判断阈值
        threshold_inner = 0.35  # 内侧阈值
        threshold_outer = 0.65  # 外侧阈值

        # 判断方向并发送按键
        current_time = time.time()
        if current_time - last_key_time > key_cooldown:
            direction = None

            if horizontal_ratio < threshold_inner:
                direction = 'right'
                pyautogui.press('right')
            elif horizontal_ratio > threshold_outer:
                direction = 'left'
                pyautogui.press('left')
            elif vertical_ratio < threshold_inner:
                direction = 'down'
                pyautogui.press('down')
            elif vertical_ratio > threshold_outer:
                direction = 'up'
                pyautogui.press('up')

            if direction:
                last_key_time = current_time
                print(f"Direction: {direction} (H:{horizontal_ratio:.2f}, V:{vertical_ratio:.2f})")
                # 在图像上显示方向
                cv2.putText(image, f"DIR: {direction.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize the image to the desired window size
    resized_image = cv2.resize(image, (window_width, window_height))
  
    cv2.imshow("Android_cam", resized_image)
  
    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
