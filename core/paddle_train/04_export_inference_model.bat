@echo off
REM PaddleOCR Detection 모델 Inference 변환 스크립트
REM 학습된 체크포인트를 Inference 모델로 변환합니다.

echo ================================================
echo PaddleOCR Detection 모델 Inference 변환
echo ================================================
echo.

cd C:\Pyg\Tools\PaddleOCR

REM Inference 모델 변환
C:\Pyg\Projects\semi\yuzyproject-aimodels\venv_ocr\Scripts\python.exe tools/export_model.py ^
  -c C:\Pyg\Projects\semi\yuzyproject-aimodels\core\paddle_train\configs\det_ke_finetune.yml ^
  -o Global.pretrained_model="C:/Pyg/Projects/semi/yuzyproject-aimodels/output/det_ke_model/best_accuracy" ^
     Global.save_inference_dir="C:/Pyg/Projects/semi/yuzyproject-aimodels/output/det_ke_inference"

echo.
echo ================================================
echo 변환 완료
echo ================================================
echo Inference 모델 위치: C:\Pyg\Projects\semi\yuzyproject-aimodels\output\det_ke_inference
echo.

pause

