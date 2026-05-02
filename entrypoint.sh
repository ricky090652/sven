#!/bin/bash
# ============================================================
# Container 啟動腳本
# ============================================================
# 每次 container 啟動時自動執行，確保 sven 套件被正確安裝。
# 因為 volume 掛載會覆蓋 image 裡的 /workspace，
# 所以需要在啟動時重新 pip install -e .
# ============================================================

# 安裝 sven 套件（editable 模式，很快，幾秒就好）
echo ">>> Installing sven package in editable mode..."
pip3 install -e . --quiet

# 執行傳入的指令（預設是 /bin/bash）
exec "$@"
