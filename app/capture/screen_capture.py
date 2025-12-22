# app/capture/screen_capture.py

import numpy as np
import cv2
from mss import mss


def capture_fullscreen_bgr(monitor_index: int = 1) -> np.ndarray:
    """
    전체 화면을 캡처하여 OpenCV에서 사용하는 BGR 이미지로 반환

    Args:
        monitor_index (int): 캡처할 모니터 인덱스 (기본값: 1 = 주 모니터)

    Returns:
        np.ndarray: BGR 이미지
    """
    with mss() as sct:
        monitor = sct.monitors[monitor_index]
        img = np.array(sct.grab(monitor))  # BGRA
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return bgr


if __name__ == "__main__":
    # 단독 실행 테스트용
    screen = capture_fullscreen_bgr()
    cv2.imshow("Fullscreen Capture Test", screen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
