import cv2
from pathlib import Path
import time

# Output folders
BASE_DIR = Path("data/real_world/raw")

CARDBOARD_DIR = BASE_DIR / "cardboard"
PAPER_DIR = BASE_DIR / "paper"

CARDBOARD_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cardboard_count = len(list(CARDBOARD_DIR.glob("*.jpg")))
paper_count = len(list(PAPER_DIR.glob("*.jpg")))

print("Real-world image capture started.")
print()
print("Press C = capture cardboard")
print("Press P = capture paper")
print("Press Q = quit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam.")
        break

    # Instructions
    cv2.putText(
        frame,
        f"Cardboard: {cardboard_count} | Paper: {paper_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "C = Cardboard | P = Paper | Q = Quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Training Image Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # Cardboard
    if key == ord("c"):

        filename = CARDBOARD_DIR / f"cardboard_{cardboard_count:03d}.jpg"

        cv2.imwrite(str(filename), frame)

        cardboard_count += 1

        print(f"Saved cardboard image: {filename}")

        time.sleep(0.3)

    # Paper
    elif key == ord("p"):

        filename = PAPER_DIR / f"paper_{paper_count:03d}.jpg"

        cv2.imwrite(str(filename), frame)

        paper_count += 1

        print(f"Saved paper image: {filename}")

        time.sleep(0.3)

    # Quit
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print()
print("Capture completed.")
print(f"Cardboard images: {cardboard_count}")
print(f"Paper images: {paper_count}")