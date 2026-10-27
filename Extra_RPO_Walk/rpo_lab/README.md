# rpo_lab: พาหุ่น RPO เดินใน Isaac Lab 3 + Newton

หุ่น RPO มาจาก Roboparty (โปรเจกต์ roboto_origin, https://github.com/Roboparty/roboto_origin)
เป็นหุ่นสองขามี 23 ข้อต่อ ขาข้างละ 6 ลำตัว 1 แขนข้างละ 5
Roboparty เทรน policy เดินมาแล้วใน Isaac Sim และแจกเป็นไฟล์ `policy.onnx`
งานนี้คือเอาหุ่นกับ policy ตัวนั้นมาเข้า Isaac Lab 3 + Newton ที่เราใช้ในคอร์ส แล้วบังคับด้วยคีย์บอร์ด

ต้องวางโฟลเดอร์ไว้ที่ `~/Downloads/Simulation_for_Humanoid/Extra_RPO_Walk/rpo_lab`
(`pyproject.toml` หา `~/IsaacLab` ด้วย path แบบ relative เหมือน Week 7)

## 1. ดึงไฟล์หุ่น (ครั้งเดียว)

```bash
cd ~/Downloads/Simulation_for_Humanoid/Extra_RPO_Walk
./setup.sh
```

ได้ `assets/rpo/urdf/rpo.urdf` พร้อม mesh กับ `pretrained/roboparty/policy.onnx`
ใช้พื้นที่ราว 80 MB ไม่ต้องโหลด repo เต็มที่ใหญ่ 3.4 GB

## 2. ติดตั้ง แล้วแปลง URDF เป็น USD (เหมือน Lab 6.2)

```bash
cd rpo_lab
uv sync
uv run python scripts/urdf_to_usd.py assets/rpo/urdf/rpo.urdf assets/usd
```

ได้ `assets/usd/rpo/rpo.usda` ที่ `src/rpo_lab/robot.py` ใช้

## 3. บังคับหุ่น

```bash
uv run python scripts/play_rpo.py --viz newton
```

คลิกที่หน้าต่าง Newton ก่อน แล้วกดค้างไว้

| ปุ่ม | ทำอะไร |
|---|---|
| I / K | เดินหน้า 0.5 m/s / ถอยหลัง 0.3 m/s |
| J / L | เดินข้างไปซ้าย / ขวา |
| U / O | เลี้ยวซ้าย / ขวา |
| C | เปิดปิดกล้องตามหุ่น |

ปุ่ม I J K L U O ใช้เหมือน Week 3 ส่วน WASD กับลูกศรเป็นปุ่มเลื่อนกล้องของ viewer

สั่งเป็นช่วง ๆ แทนคีย์บอร์ดก็ได้ แบบ `วินาที:vx,vy,yaw`

```bash
uv run python scripts/play_rpo.py --viz newton --plan "2:0,0,0 5:0.5,0,0 5:0.3,0,0.6"
```

## 4. ข้างในทำงานยังไง

| ไฟล์ | มีอะไร |
|---|---|
| `src/rpo_lab/robot.py` | `RPO_CFG` ใช้ kp kd ท่ายืน และแรงบิดสูงสุดชุดเดียวกับที่ Roboparty ใช้ตอนเทรน กับ `POLICY_JOINTS` ลำดับข้อต่อที่ policy ใช้ |
| `src/rpo_lab/tasks/rpo/env_cfg.py` | task `RPO-Velocity-Flat-Play` ค่าที่ policy เห็น 78 ค่า ซ้อน 10 รอบ เป็น 780 ค่า |
| `scripts/play_rpo.py` | รัน `policy.onnx` ด้วย onnxruntime แล้วอ่านคีย์บอร์ด |

ทุก 0.02 วินาที policy ได้ค่า 78 ตัว
ความเร็วหมุนตัว 3 ค่า, ทิศแรงโน้มถ่วงเทียบกับตัวหุ่น 3 ค่า, คำสั่ง vx vy yaw 3 ค่า,
มุมข้อต่อลบท่ายืน 23 ค่า, ความเร็วข้อต่อ 23 ค่า และ action รอบก่อน 23 ค่า
ซ้อน 10 รอบล่าสุด (เก่าสุดอยู่หน้า) ได้ 780 ค่า แล้ว policy ตอบกลับ 23 ค่า
มุมเป้าหมาย = ท่ายืน + 0.25 × action

**ลำดับข้อต่อสำคัญมาก** Newton เรียงข้อต่อตาม URDF (ขาซ้ายครบ 6 ก่อน แล้วค่อยขาขวา)
แต่ policy เทรนบน Isaac Sim ซึ่งเรียงสลับซ้ายขวา (ขาซ้าย yaw, ขาขวา yaw, ลำตัว, ...)
เราเลยบอกชื่อข้อต่อตามลำดับไว้ใน `POLICY_JOINTS` แล้วใช้ `preserve_order=True` ทั้งตอนอ่านค่าและตอนสั่ง

## 5. ลองแกล้งหุ่น

```bash
uv run python scripts/play_rpo.py --viz newton --plan "1:0,0,0 8:0.5,0,0" --no-remap      # ไม่จับคู่ลำดับข้อต่อ
uv run python scripts/play_rpo.py --viz newton --plan "1:0,0,0 8:0.5,0,0" --kp-scale 0.5  # มอเตอร์อ่อนลงครึ่งนึง
```

ผลที่ครูได้ (19 ต.ค. 2026 สั่งเดินหน้า 0.5 m/s)

| แบบ | ผล |
|---|---|
| ค่าเดิม | เดิน 0.50 m/s ตัวสูง 0.75 m ไม่ล้ม |
| `--kp-scale 0.5` | ไม่ล้ม แต่ย่อตัวเหลือ 0.69 m และเดินเร็วเกินคำสั่ง ราว 0.64 m/s |
| `--no-remap` | ล้มภายใน 1.4 ถึง 2.6 วินาที (สองรอบที่ลอง) |

## 6. ต่อยอด

- เทรน policy ของเราเองบนหุ่น RPO แบบ Week 4 (ต้องเพิ่ม reward กับ agent config และใช้ GPU)
- ลองเอาหุ่นไปเดินในวงกตของ Week 7

ลิขสิทธิ์: ตัวหุ่น (`rpo_description`) ใช้ CERN-OHL-W-2.0 ส่วน deploy ที่มี `policy.onnx` ใช้ GPL-3.0 ของ Roboparty
โฟลเดอร์นี้ไม่มีไฟล์ของเขา `setup.sh` ดึงจาก GitHub โดยตรง
