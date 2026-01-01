# GitHub Push 가이드

이 문서는 대용량 파일 문제를 해결하고 GitHub에 코드만 올리는 방법을 설명합니다.

## 문제 상황

- GitHub에 파일을 올렸을 때 폴더 이름만 보이고 내용이 없음
- 대용량 파일 때문에 push가 실패함
- `code_syntax_dataset/`, `venv_ocr/`, 모델 파일 등이 너무 큼

## 해결 방법

### 1. `.gitignore` 업데이트 완료

다음 항목들이 `.gitignore`에 추가되어 Git 추적에서 제외됩니다:

- `venv_ocr/` - 가상환경 (매우 큼)
- `code_syntax_dataset/` - 데이터셋 (25,000 + 5,000 이미지)
- `output_root/` - 학습된 모델 파일
- `outputs/` - 임시 출력 파일
- `PaddleOCR/output/` - PaddleOCR 학습 결과
- `*.pdparams`, `*.pdopt`, `*.states` - 모델 체크포인트
- `server/` - 서버 모델 파일

### 2. 추가된 문서

#### `yuzyproject-aimodels/SETUP_GUIDE.md`

데이터셋 생성부터 학습 시작까지의 **전체 과정을 상세히 설명**하는 가이드:

- 환경 설정 (가상환경, PaddlePaddle 설치)
- 데이터셋 생성 (상세 코드 내용 포함)
- 학습 시작 및 모니터링
- 문제 해결 가이드

이 가이드를 따라하면 다른 컴퓨터에서도 동일한 환경을 만들 수 있습니다.

## 다음 단계: GitHub에 Push하기

### 1. 변경 사항 확인

```bash
cd C:\Pyg\Projects\semi
git status
```

다음 파일들이 수정되었습니다:
- `.gitignore` - 대용량 파일 제외 규칙 추가
- `yuzyproject-aimodels/SETUP_GUIDE.md` - 상세 설정 가이드 (새 파일)
- `yuzyproject-aimodels/README.md` - 업데이트

### 2. 대용량 파일이 추적되고 있는지 확인

```bash
# 이미 Git에 추가된 대용량 파일 확인
git ls-files | findstr /I "venv_ocr code_syntax_dataset output_root server"
```

만약 결과가 나온다면, 해당 파일들을 Git에서 제거해야 합니다:

```bash
# Git 추적에서 제거 (파일은 삭제하지 않음)
git rm -r --cached yuzyproject-aimodels/venv_ocr
git rm -r --cached yuzyproject-aimodels/code_syntax_dataset
git rm -r --cached yuzyproject-aimodels/output_root
git rm -r --cached yuzyproject-aimodels/server
git rm -r --cached PaddleOCR/output
```

### 3. 변경 사항 스테이징

```bash
# .gitignore 업데이트 추가
git add .gitignore

# 새 문서 추가
git add yuzyproject-aimodels/SETUP_GUIDE.md
git add yuzyproject-aimodels/README.md

# 다른 코드 파일들 (필요한 경우)
git add yuzyproject-aimodels/*.py
git add yuzyproject-aimodels/core/
# ... 필요한 파일들만 추가
```

### 4. 커밋

```bash
git commit -m "docs: 대용량 파일 제외 및 설정 가이드 추가

- .gitignore에 venv, dataset, model 파일 제외 규칙 추가
- SETUP_GUIDE.md: 데이터셋 생성 및 학습 전체 가이드 추가
- README.md 업데이트"
```

### 5. Push

```bash
git push origin gonida
```

## 데이터셋 생성 방법 (다른 컴퓨터에서)

다른 컴퓨터에서 프로젝트를 받은 후 데이터셋을 생성하려면:

1. **저장소 클론**
   ```bash
   git clone <repository-url>
   cd semi
   git checkout gonida
   ```

2. **상세 가이드 참조**
   - `yuzyproject-aimodels/SETUP_GUIDE.md` 파일을 열어서 전체 과정을 따라하세요.
   - 특히 [2. 데이터셋 생성](#2-데이터셋-생성) 섹션을 참조하세요.

3. **빠른 실행**
   ```bash
   cd yuzyproject-aimodels
   python prepare_code_syntax_dataset.py
   ```

## 확인 사항

### GitHub에 올라가지 말아야 할 항목

다음 항목들이 `.gitignore`에 포함되어 있는지 확인:

- ✅ `venv_ocr/`
- ✅ `code_syntax_dataset/`
- ✅ `output_root/`
- ✅ `outputs/`
- ✅ `server/`
- ✅ `PaddleOCR/output/`
- ✅ `*.pdparams`
- ✅ `*.pdopt`
- ✅ `*.states`

### GitHub에 올라가야 할 항목

- ✅ `.gitignore`
- ✅ `SETUP_GUIDE.md` (새 파일)
- ✅ `README.md`
- ✅ `prepare_code_syntax_dataset.py` (데이터셋 생성 스크립트)
- ✅ `core/` (코드 파일들)
- ✅ `start_training_clean.bat` (학습 실행 스크립트)
- ✅ 기타 코드 파일들 (`.py`, `.bat`, `.yml` 등)

## 문제 해결

### 문제: 이미 커밋된 대용량 파일 제거

만약 이미 Git에 커밋된 대용량 파일이 있다면:

```bash
# Git 히스토리에서 완전히 제거 (주의: 히스토리가 변경됩니다)
git filter-branch --force --index-filter \
  "git rm -rf --cached --ignore-unmatch yuzyproject-aimodels/venv_ocr yuzyproject-aimodels/code_syntax_dataset" \
  --prune-empty --tag-name-filter cat -- --all

# 또는 BFG Repo-Cleaner 사용 (더 빠름)
# https://rtyley.github.io/bfg-repo-cleaner/
```

### 문제: Push 거부됨

GitHub는 파일 크기 제한이 있습니다:
- 단일 파일: 100MB
- 경고: 50MB

너무 큰 파일이 있다면:
1. `.gitignore`에 추가되어 있는지 확인
2. Git 추적에서 제거 (`git rm --cached`)
3. 다시 커밋 및 push

## 요약

1. ✅ `.gitignore` 업데이트 완료
2. ✅ `SETUP_GUIDE.md` 작성 완료 (상세 가이드)
3. ✅ `README.md` 업데이트 완료
4. ⏭️ Git에서 대용량 파일 제거 (필요한 경우)
5. ⏭️ 변경 사항 커밋 및 push

**다음 단계**: 위의 "다음 단계: GitHub에 Push하기" 섹션을 따라하세요.

