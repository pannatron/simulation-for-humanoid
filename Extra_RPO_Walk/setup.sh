#!/usr/bin/env bash
# ดึงตัวหุ่น RPO (rpo_description) กับ policy ที่เทรนแล้ว จาก roboto_origin ของ Roboparty
# แล้วก๊อปเข้าโปรเจกต์ rpo_lab   (repo เต็ม 3.4 GB ไม่ต้องโหลด ใช้แค่ ~80 MB)
set -e
cd "$(dirname "$0")"
if [ ! -d roboto_origin ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/Roboparty/roboto_origin.git
fi
git -C roboto_origin sparse-checkout set modules/rpo_description modules/roboparty_deploy/src/inference/robots/rpo
M=roboto_origin/modules
mkdir -p rpo_lab/assets/rpo rpo_lab/pretrained/roboparty
cp -r $M/rpo_description/urdf $M/rpo_description/meshes rpo_lab/assets/rpo/
cp $M/roboparty_deploy/src/inference/robots/rpo/models/policy.onnx rpo_lab/pretrained/roboparty/
echo "ได้แล้ว:"
echo "  rpo_lab/assets/rpo/urdf/rpo.urdf        ตัวหุ่น"
echo "  rpo_lab/pretrained/roboparty/policy.onnx policy เดิน"
echo "ต่อไป: cd rpo_lab && uv sync"
