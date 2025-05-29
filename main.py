import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Webcam capture
cap = cv2.VideoCapture(1)
print("📸 Align your face. Press SPACE to capture.")

user_avg = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # Draw placeholder box
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
            # Landmark indices: forehead=10, cheek=234 & 454, eye=468, hair=152 (top)
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
            eye_points = [145, 374]  # NEW better eye color landmarks
            forehead_points = [9, 10]

            avg_skin = get_avg_color(cheek_points)
            avg_eye = get_avg_color(eye_points)
            avg_hair = get_avg_color(forehead_points)
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

            # Draw dots for debug
            for i in [10, 234, 454, 159, 386]:
                if i < len(face_landmarks.landmark):
                    pt = face_landmarks.landmark[i]
                    x, y = int(pt.x * w), int(pt.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                else:
                    print(f"⚠️ Landmark index {i} not available.")

    cv2.imshow("Face Scanner", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32 and user_avg is not None:  # SPACE
        # Show capture success message
        cv2.putText(frame, "Captured!", (180, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Face Scanner", frame)
        cv2.waitKey(1000)  # Hold for 1 second
        break

cap.release()
cv2.destroyAllWindows()

# Check if user_avg is not None before calling match_season 
if user_avg is not None:
    # Season matching (UPDATED)
    season_refs = {
        "Spring": np.array([151, 183, 93]),
        "Summer": np.array([160, 156, 180]),
        "Autumn": np.array([139, 115, 50]),
        "Winter": np.array([153, 65, 136])
    }

    def match_season(user_color, refs):
        return min(refs.keys(), key=lambda s: np.linalg.norm(user_color - refs[s]))

    matched = match_season(user_avg, season_refs)
    print(f"\n🥰 You are most likely a {matched} girly!")

    # Show final palette (UPDATED)
    season_palettes = {
        "Spring": [[255,160,130],[0,180,180],[255,215,0],[193,154,107],[50,205,50]],
        "Summer": [[205,150,170],[100,149,237],[230,230,250],[72,60,50],[192,192,192]],
        "Autumn": [[204,85,0],[107,142,35],[0,128,128],[128,0,0],[255,219,88]],
        "Winter": [[255,0,0],[255,0,255],[0,71,171],[0,0,0],[255,255,255]]
    }

    palette = np.array(season_palettes[matched], dtype=np.uint8)
    strip = np.zeros((100, 600, 3), dtype=np.uint8)

    for i, color in enumerate(palette):
        strip[:, i*100:(i+1)*100] = color

    avg_img = np.full((100, 100, 3), user_avg.astype(np.uint8), dtype=np.uint8)

    import matplotlib.pyplot as plt
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

else:
    print("⚠️ Face not captured yet. Please align your face and press SPACE to capture.")
