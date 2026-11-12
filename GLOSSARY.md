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

**目标积分角度（Target Integrated Angle）**:
由积分角度派生的角度信号，是模型唯一的预测对象，也是角度信号变体的
对比基准；它不是实测的绝对姿态真值。
_Avoid_: ground truth、绝对参考角度、原始积分角度

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

**分析窗口（Analysis Window）**:
积分角度序列中按指定物理时长划分的连续、等样本数数据段；不足一个完整窗口的
尾部不属于任何分析窗口。
_Avoid_: 变长窗口、残余窗口

**窗口归零角度（Window-relative Angle）**:
分析窗口内各轴积分角度减去该轴窗口首值后得到的相对角位移。
_Avoid_: 重新积分角度、窗口姿态角

**窗口相对开始时间（Window Start Elapsed Time）**:
分析窗口首个样本在文件合成时间轴上的相对秒数；它不是传感器时间戳。
_Avoid_: 窗口开始时间戳、采集时间

**角度信号变体（Angle Signal Variant）**:
同一积分角度序列的原始版本或因果滤波版本；所有变体使用相同的窗口划分和窗口
归零语义。
_Avoid_: 独立重积分信号

**分析源组（Analysis Source Group）**:
由同一个原始角速度文件派生的积分角度、全部角度信号变体及其分析
产物组成的集合；每个源文件独立成组。
_Avoid_: 原始数据组、目录组

**活动场景（Activity Scenario）**:
用于分析源组统计聚合的互斥活动类别，限于走路、跑步和站立静止；持握方式、
屏幕方向、路面和鞋类是场景明细，不构成新的活动场景。
_Avoid_: 采集条件组合、持握场景

**违规窗口（Violating Window）**:
至少一个轴的窗口归零角度严格超出指定下界或上界的分析窗口；等于
边界不属于违规。
_Avoid_: 超阈值采样点、异常文件

**违规时间覆盖区间（Violation Time Coverage Interval）**:
同一分析源组、角度信号变体和轴上，违规窗口所覆盖的半开时间区间之并集；
重叠或首尾相接的区间属于同一段连续覆盖。
_Avoid_: 违规窗口时长之和、重复覆盖时长

**角度极值分析（Angle Extrema Analysis）**:
将分析窗口内各轴窗口归零角度的最小值和最大值与指定上下界比较的分析。
_Avoid_: 仅最大角度分析、姿态分析
