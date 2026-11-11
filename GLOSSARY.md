# IMU Data Analysis

This context describes the sensor recordings and derived signals used by IMU
experiments in this repository.

## Language

**原始角速度（Raw Angular Velocity）**:
陀螺仪直接记录的设备坐标系三轴角速度，单位为 rad/s。
_Avoid_: 原始角度、旋转角

**积分角度（Integrated Angle）**:
原始角速度按轴累计得到的相对角位移，单位为度；它不是设备姿态、朝向或
Euler 角。
_Avoid_: 姿态角、朝向、Euler 角

**原始加速度（Raw Acceleration）**:
加速度计直接记录的设备坐标系三轴读数，单位为 m/s²；当前不保证其中是否包含
重力分量。
_Avoid_: 线性加速度

**传感器时间戳（Sensor Timestamp）**:
传感器时钟产生的 ns 计数值，时钟起点未知，只有同一记录内的差值具有已确认的
时间含义。
_Avoid_: Unix 时间戳、墙上时钟时间

**采集会话（Capture Session）**:
同一目录内拥有相同文件名前缀、仅 `gyro` 与 `accel` 传感器字段不同的一组记录；
加速度记录是可选的。
_Avoid_: 仅按文件名跨目录配对
