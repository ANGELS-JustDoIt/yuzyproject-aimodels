@echo off
REM PaddleOCR Detection 모델 Inference 변환 스크립트
REM 학습된 체크포인트를 Inference 모델로 변환합니다.
REM GPU 경합 방지를 위해 CPU 모드로 실행

REM GPU 완전 차단: venv_ocr에 paddlepaddle-gpu가 설치되어 있어도 CPU 모드로 강제
SET CUDA_VISIBLE_DEVICES=-1
REM FLAGS_selected_gpus는 빈 문자열이면 오류 발생하므로 삭제 (설정하지 않음)
SET PADDLE_INFERENCE_PLACE=cpu
SET USE_GPU=0
SET FLAGS_use_gpu=0
SET FLAGS_use_cuda=0
SET FLAGS_use_xpu=0
SET FLAGS_use_npu=0

echo ================================================
echo PaddleOCR Detection 모델 Inference 변환
echo GPU 경합 방지: CPU 모드로 실행
echo ================================================
echo.

cd C:\Pyg\Tools\PaddleOCR

REM Inference 모델 변환 (CPU 모드)
REM PaddlePaddle 2.6.2에서는 export_with_pir=False로 설정해야 AssertionError 방지
C:\Pyg\Projects\semi\yuzyproject-aimodels\venv_ocr\Scripts\python.exe tools/export_model.py ^
  -c C:\Pyg\Projects\semi\yuzyproject-aimodels\core\paddle_train\configs\det_ke_finetune.yml ^
  -o Global.pretrained_model="C:/Pyg/Projects/semi/yuzyproject-aimodels/output/det_ke_model/best_accuracy" ^
     Global.save_inference_dir="C:/Pyg/Projects/semi/yuzyproject-aimodels/output/det_ke_inference" ^
     Global.use_gpu=False ^
     Global.export_with_pir=False

echo.
echo ================================================
echo 변환 완료
echo ================================================
echo Inference 모델 위치: C:\Pyg\Projects\semi\yuzyproject-aimodels\output\det_ke_inference
echo.

pause

