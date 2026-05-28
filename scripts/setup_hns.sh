#!/usr/bin/env bash
# setup_hns.sh  –  HNS MAPPO LSTM 학습 환경 세팅 스크립트
# 사용법: bash scripts/setup_hns.sh
# 전제조건:
#   - Python 3.11
#   - MuJoCo 2.1.0 이 ~/.mujoco/mujoco210 에 설치되어 있을 것
#   - NVIDIA GPU + CUDA 드라이버

set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== [1/4] mujoco_worldgen 설치 ==="
pip install "Cython<3"
pip install -e "$REPO_DIR/marllib/patch/hns/mujoco-worldgen/"
pip install xmltodict jsonnet numpy-stl

echo "=== [2/4] mujoco-py 설치 ==="
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
pip install mujoco-py

echo "=== [3/4] Ray 2.3.0 / gym 호환 패키지 설치 ==="
pip install "pydantic<2"
pip install "gym==0.22.0"
pip install tensorboard

echo "=== [4/4] 환경변수 안내 ==="
echo ""
echo "  학습을 실행하기 전에 아래 환경변수를 설정하세요:"
echo ""
echo "  export LD_LIBRARY_PATH=\$HOME/.mujoco/mujoco210/bin:\$LD_LIBRARY_PATH"
echo "  export PYTHONPATH=$REPO_DIR/scripts/ray_compat:\$PYTHONPATH"
echo ""
echo "  그리고 아래 명령어로 학습을 시작하세요:"
echo ""
echo "  python $REPO_DIR/scripts/train_hns_mappo_lstm.py"
echo ""
echo "=== 세팅 완료 ==="
