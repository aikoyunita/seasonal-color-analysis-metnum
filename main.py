# Import required libraries
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Function to apply Gaussian blur for smoother input frames
def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), sigmaX=1)

# Function to apply Sobel operator (finite difference method) for edge detection
def apply_finite_difference_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    edge_x = cv2.filter2D(gray, -1, sobel_x)
    edge_y = cv2.filter2D(gray, -1, sobel_y)
    magnitude = cv2.magnitude(edge_x.astype(np.float32), edge_y.astype(np.float32))
    return cv2.convertScaleAbs(magnitude)

# Function to get average RGB color from specific landmark indexes
def get_avg_color(ids, face_landmarks, frame, h, w):
    colors = []
    for idx in ids:
        if idx >= len(face_landmarks.landmark):  # Prevent out-of-range error
            continue
        pt = face_landmarks.landmark[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        if 0 <= x < w and 0 <= y < h:
            b, g, r = frame[y, x]
            colors.append(np.array([r, g, b], dtype=np.float32))
    return np.mean(colors, axis=0) if colors else np.array([0, 0, 0], dtype=np.float32)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Start webcam capture
cap = cv2.VideoCapture(1)
ret, test_frame = cap.read()
if not ret:
    print("❌ Failed to read from webcam.")
    exit()

# Setup face alignment box in the center
frame_h, frame_w = test_frame.shape[:2]
box_w, box_h = 500, 700
start_x, start_y = (frame_w - box_w) // 2, (frame_h - box_h) // 2
end_x, end_y = start_x + box_w, start_y + box_h

# Instructions for user
print("📸 Align your face. Press SPACE to capture.")
print("🔁 Press 'E' to toggle Edge Detection (Numerical Methods feature).")

user_avg = None
show_edges = False

# Main loop for real-time face analysis
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the image for natural alignment
    frame = apply_gaussian_blur(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # Draw the face alignment box
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    cv2.putText(frame, "ALIGN FACE WITHIN BOX", (start_x + 50, end_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w = frame.shape[:2]

            # Selected facial points
            cheek_pts = [50, 205, 425, 280]  
            eye_pts = [468, 473]
            hair_pts = [54, 284]
            
            # Calculate average colors
            avg_cheek = get_avg_color(cheek_pts, face_landmarks, frame, h, w)
            avg_skin = get_avg_color(cheek_pts, face_landmarks, frame, h, w)
            avg_eye = get_avg_color(eye_pts, face_landmarks, frame, h, w)
            avg_hair = get_avg_color(hair_pts, face_landmarks, frame, h, w)

            # Final average color
            user_avg = (avg_skin + avg_eye + avg_hair) / 3

            # Visualize sampled points
            for i in cheek_pts + eye_pts + hair_pts:
                pt = face_landmarks.landmark[i]
                x, y = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
    else:
        user_avg = None
        cv2.putText(frame, "No face detected!", (start_x + 50, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Optionally apply edge detection
    display_frame = apply_finite_difference_edge_detection(frame.copy()) if show_edges else frame.copy()
    if show_edges:
        cv2.putText(display_frame, "NUMERICAL EDGE MODE", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Face Scanner", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('e'):
        show_edges = not show_edges
        print("🟢 Edge Detection ON" if show_edges else "⚪ Edge Detection OFF")
    elif key == 32 and user_avg is not None:  # SPACE
        captured_frame = frame.copy()
        cv2.putText(captured_frame, "Captured!", (180, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Face Scanner", captured_frame)
        cv2.waitKey(1000)
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()

# If face data is captured, determine season palette
if user_avg is not None:
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

    # Display the matched color palette
    palettes = {
        "Spring": [[255,160,130],[0,180,180],[255,215,0],[193,154,107],[50,205,50]],
        "Summer": [[205,150,170],[100,149,237],[230,230,250],[72,60,50],[192,192,192]],
        "Autumn": [[204,85,0],[107,142,35],[0,128,128],[128,0,0],[255,219,88]],
        "Winter": [[255,0,0],[255,0,255],[0,71,171],[0,0,0],[255,255,255]]
    }

    palette = np.array(palettes[matched], dtype=np.uint8)
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
else:
    print("⚠️ Face not captured yet. Please align your face and press SPACE to capture.")
