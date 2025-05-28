import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURABLE SETTINGS ---
SCAN_BOX_SIZE = 40
SCAN_BOX_THICKNESS = 5

# --- STEP 1: Display Face Scan Box Overlay (Optional UI Layer) ---
overlay = np.zeros((800, 600, 4), dtype=np.uint8)
color = (0, 255, 0, 255)  # Green in RGBA

# Corner brackets
corners = {
    "tl": (50, 50),
    "tr": (550, 50),
    "bl": (50, 750),
    "br": (550, 750)
}

for label, (x, y) in corners.items():
    if "t" in label:
        cv2.line(overlay, (x, y), (x + SCAN_BOX_SIZE if 'l' in label else x - SCAN_BOX_SIZE, y), color, SCAN_BOX_THICKNESS)
        cv2.line(overlay, (x, y), (x, y + SCAN_BOX_SIZE), color, SCAN_BOX_THICKNESS)
    else:
        cv2.line(overlay, (x, y), (x + SCAN_BOX_SIZE if 'l' in label else x - SCAN_BOX_SIZE, y), color, SCAN_BOX_THICKNESS)
        cv2.line(overlay, (x, y), (x, y - SCAN_BOX_SIZE), color, SCAN_BOX_THICKNESS)

cv2.putText(overlay, "ALIGN FACE WITHIN FRAME", (120, 780), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2, cv2.LINE_AA)

# --- STEP 2: Capture Snapshot from Webcam ---
cap = cv2.VideoCapture(0)
print("\n📸 Press SPACE to take a snapshot, or ESC to cancel.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Live Webcam", frame)
    key = cv2.waitKey(1)
    if key % 256 == 27:
        print("❌ Capture cancelled.")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key % 256 == 32:
        img = frame.copy()
        break

cap.release()
cv2.destroyAllWindows()

# --- STEP 3: Preprocess Image ---
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = cv2.resize(img_rgb, (600, 800))
img_rgb = cv2.GaussianBlur(img_rgb, (5, 5), 0)

# --- STEP 4: Select ROIs and Compute Averages ---
def get_avg_color(region_name):
    print(f"👉 Select your {region_name.upper()} region, then press ENTER or SPACE")
    roi = cv2.selectROI(f"{region_name.capitalize()} ROI", img_rgb)
    crop = img_rgb[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    return np.mean(crop.reshape(-1, 3), axis=0)

avg_skin = get_avg_color("skin")
avg_hair = get_avg_color("hair")
avg_eye = get_avg_color("eye")
cv2.destroyAllWindows()

user_avg = (avg_skin + avg_hair + avg_eye) / 3

# --- STEP 5: Define Season References ---
season_refs = {
    "Winter": np.array([180, 160, 200]),
    "Summer": np.array([200, 180, 170]),
    "Autumn": np.array([220, 190, 150]),
    "Spring": np.array([250, 220, 180])
}

# --- STEP 6: Match Season ---
def match_season(user_color, refs):
    return min(refs, key=lambda season: np.linalg.norm(user_color - refs[season]))

matched_season = match_season(user_avg, season_refs)
print(f"\n✨ You are most likely a {matched_season} palette girl!")

# --- STEP 7: Define Full Season Palettes ---
season_palettes = {
    "Winter": [[255, 255, 255], [0, 0, 0], [0, 56, 168], [116, 12, 164], [200, 0, 90], [0, 140, 140]],
    "Summer": [[230, 240, 250], [180, 200, 230], [220, 170, 200], [190, 200, 220], [145, 180, 200], [160, 160, 180]],
    "Autumn": [[120, 70, 20], [190, 120, 45], [220, 170, 50], [160, 80, 50], [170, 120, 90], [100, 60, 30]],
    "Spring": [[255, 230, 180], [255, 210, 140], [255, 160, 130], [230, 150, 200], [160, 220, 150], [200, 240, 180]]
}

# --- STEP 8: Display Result ---
palette_colors = np.array(season_palettes[matched_season], dtype=np.uint8)
palette_img = np.zeros((100, 600, 3), dtype=np.uint8)
for i, color in enumerate(palette_colors):
    palette_img[:, i*100:(i+1)*100] = color

avg_color_img = np.zeros((100, 100, 3), dtype=np.uint8)
avg_color_img[:, :] = user_avg.astype(np.uint8)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(palette_img)
plt.title(f"{matched_season} Palette")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(avg_color_img)
plt.title("Your Avg Tone")
plt.axis("off")

plt.tight_layout()
plt.show()