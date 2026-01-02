@echo off
chcp 65001 >nul
echo ======================================================================
echo PP-OCRv5 학습 시작 (RTX 4070Ti, venv_ocr)
echo ======================================================================
echo.

cd /d "%~dp0"

REM venv_ocr 가상환경 확인
set VENV_OCR=..\..\venv_ocr
if not exist "%VENV_OCR%\Scripts\python.exe" (
    echo [ERROR] venv_ocr 가상환경을 찾을 수 없습니다!
    echo   경로: %VENV_OCR%
    pause
    exit /b 1
)

echo [가상환경] venv_ocr 사용
echo [Python] %VENV_OCR%\Scripts\python.exe
echo.

REM 학습 실행
"%VENV_OCR%\Scripts\python.exe" train_v5.py

pause

