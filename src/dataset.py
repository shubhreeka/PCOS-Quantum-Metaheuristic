"""Pre-processing shared by training and the web app."""
import cv2


def clahe_rgb(img_rgb, clip=2.0, tiles=(8, 8)):
    """CLAHE on the grayscale ultrasound, returned as 3-channel RGB (uint8)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
