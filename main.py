import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), sigmaX=1)

def apply_finite_difference_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=np.float32)

    edge_x = cv2.filter2D(gray, -1, sobel_x)
    edge_y = cv2.filter2D(gray, -1, sobel_y)
    magnitude = cv2.magnitude(edge_x.astype(np.float32), edge_y.astype(np.float32))
    return cv2.convertScaleAbs(magnitude)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = cv2.VideoCapture(1)
print(" Align your face. Press SPACE to capture.")

user_avg = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame = apply_gaussian_blur(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    frame_h, frame_w = frame.shape[:2]
    box_width = 500
    box_height = 700
    start_x = (frame_w - box_width) // 2
    start_y = (frame_h - box_height) // 2
    end_x = start_x + box_width
    end_y = start_y + box_height

    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    cv2.putText(frame, "ALIGN FACE WITHIN BOX", (start_x + 50, end_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            def get_avg_color(ids):
                colors = []
                for idx in ids:
                    if idx >= len(face_landmarks.landmark):
                        continue
                    pt = face_landmarks.landmark[idx]
                    x, y = int(pt.x * w), int(pt.y * h)
                    if y >= h or x >= w or y < 0 or x < 0:
                        continue
                    b, g, r = frame[y, x]
                    colors.append(np.array([r, g, b], dtype=np.float32))
                return np.mean(colors, axis=0) if colors else np.array([0, 0, 0], dtype=np.float32)

            cheek_points = [205, 425]
            eye_points = [145, 374]
            forehead_points = [9, 10]

            avg_skin = get_avg_color(cheek_points)
            avg_eye = get_avg_color(eye_points)
            avg_hair = get_avg_color(forehead_points)
            user_avg = (avg_skin + avg_eye + avg_hair) / 3

            for i in cheek_points + eye_points + forehead_points:
                pt = face_landmarks.landmark[i]
                x, y = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow("Face Scanner", frame)

    key = cv2.waitKey(1)
    if key == 27:  
        break
    elif key == 32 and user_avg is not None: 
        # Show edge-detected version
        edges = apply_finite_difference_edge_detection(frame)
        cv2.imshow("Edge Detection (Sobel)", edges)

        # Show capture success message
        cv2.putText(frame, "Captured!", (180, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Face Scanner", frame)
        cv2.waitKey(1000)
        break

cap.release()
cv2.destroyAllWindows()

season_refs = {
    "Winter": np.array([180, 160, 200]),
    "Summer": np.array([200, 180, 170]),
    "Autumn": np.array([220, 190, 150]),
    "Spring": np.array([250, 220, 180])
}

def match_season(user_color, refs):
    return min(refs.keys(), key=lambda s: np.linalg.norm(user_color - refs[s]))

matched = match_season(user_avg, season_refs)
print(f"\n🥰 You are most likely a {matched} girly!")

season_palettes = {
    "Winter": [[255,255,255],[0,0,0],[0,56,168],[116,12,164],[200,0,90],[0,140,140]],
    "Summer": [[230,240,250],[180,200,230],[220,170,200],[190,200,220],[145,180,200],[160,160,180]],
    "Autumn": [[120,70,20],[190,120,45],[220,170,50],[160,80,50],[170,120,90],[100,60,30]],
    "Spring": [[255,230,180],[255,210,140],[255,160,130],[230,150,200],[160,220,150],[200,240,180]]
}

palette = np.array(season_palettes[matched], dtype=np.uint8)
strip = np.zeros((100, 600, 3), dtype=np.uint8)
for i, color in enumerate(palette):
    strip[:, i*100:(i+1)*100] = color

avg_img = np.full((100, 100, 3), user_avg.astype(np.uint8), dtype=np.uint8)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(strip)
plt.title(f"{matched} Palette")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(avg_img)
plt.title("Your Average Color")
plt.axis('off')
plt.tight_layout()
plt.show()
