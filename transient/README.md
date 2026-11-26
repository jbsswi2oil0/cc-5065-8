# 瞬态光传输渲染器（Transient Light-Transport Renderer）

纯 Python 标准库实现的蒙特卡洛瞬态路径追踪器：为每条光路累计传播时间，把贡献
按可检查的核分配到接收端时间箱，生成反映光到达过程的图像序列，并提供带两个
可移动曝光门的图形实验窗口。

## 快速开始

```bash
# 图形入口（主要方式）
python3 -I run_server.py
# 浏览器打开 http://127.0.0.1:8000/

# 命令行（仅辅助）
python3 -I transient_renderer/cli.py render --scene two_gate --samples 16

# 全部测试
python3 -I tests/run_all.py
```

## 目录结构

```
transient_renderer/
  vector.py    向量/颜色与反射折射数学
  rng.py       可复现的计数器随机数
  scene.py     材质、球/平面/矩形/盒、点光源、针孔相机
  tracer.py    瞬态路径追踪（逐段几何长度、OPL、传播时间）
  kernel.py    时间沉积核与时间箱边界规则
  scenes.py    四个预设场景
  render.py    渲染编排、门积分、读出决策、历史粗箱
  server.py    图形窗口 HTTP 后端（按钮调用真实渲染）
  cli.py       命令行辅助
web/           实验窗口前端（拖动门、切片、像素响应、OPL 分布）
tests/         核测试、双门实验、物理测试、E2E
docs/          USAGE.md（图形入口）、PHYSICS.md（模型与规则）
```

## 四个预设场景

| 场景 | 检验内容 |
|---|---|
| `two_gate` | 真空短脉冲与穿慢速介质光程调至同刻到达；门交界脉冲按核分裂 |
| `path_lengths` | 1.5/3.0/4.5 m 不同光程的到达时刻与 OPL/c 独立复算 |
| `mirror_detour` | 镜面绕行路径比直射长，间接能量更晚到达 |
| `indirect_depth` | 遮挡直射光，验证有限深度反弹照明与最早到达下限 |

## 关键保证

- 时间箱积分与同模型稳态结果相容，核沉积能量守恒（测试强制断言）。
- 时间单位（ns）、长度单位（m）、光速与 IOR 在结果中显式声明。
- 历史整箱数据不能恢复亚箱顺序；门调整复用已追踪路径，不重新追踪。
- 所有解析到达时刻都可由几何长度与介质速度独立复算，不依赖视觉动画。
