# app/capture/roi_crop.py

import cv2
import numpy as np


def select_roi(image: np.ndarray, window_name: str = "Select ROI") -> np.ndarray:
    """
    OpenCV UI로 ROI를 선택하여 Crop된 이미지를 반환

    Args:
        image (np.ndarray): BGR 이미지
        window_name (str): ROI 선택 창 이름

    Returns:
        np.ndarray: Crop된 BGR 이미지
    """
    roi = cv2.selectROI(
        window_name,
        image,
        showCrosshair=True,
        fromCenter=False
    )
    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w == 0 or h == 0:
        raise ValueError("ROI 선택이 취소되었거나 크기가 0입니다.")

    cropped = image[y:y + h, x:x + w]
    return cropped


if __name__ == "__main__":
    # 단독 테스트용
    from app.capture.screen_capture import capture_fullscreen_bgr

    screen = capture_fullscreen_bgr()
    cropped = select_roi(screen)

    cv2.imshow("Cropped Result", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()