# 智能图片元素提取工具

一个本地运行的 AI 图片元素提取项目，支持两种工作方式：

- 桌面 UI：打开图片、框选区域、生成透明图层、管理图层并导出。
- 命令行 CLI：使用自然语言或点击坐标调用 Grounding DINO + SAM2 自动提取图片元素。

项目目标是把一张普通 PNG/JPG/WEBP 图片中的人物、物体、文字块或局部区域提取成带透明通道的 PNG 图层，方便继续用于 Photoshop、Figma、ComfyUI、Unreal UMG 或其他素材工作流。

## 功能概览

- PySide6 桌面界面。
- 画布支持打开图片、缩放、平移、矩形框选。
- 框选区域可生成 mask 预览，并创建为可管理图层。
- 图层支持重命名、显示/隐藏、删除、单层导出、全部导出。
- UI 中可输入自然语言命令并调用现有 AI 管线生成图层。
- CLI 和 UI 都支持自然语言切割命令，例如 `cut out a man with sword and save to ./my_outputs`。
- CLI 支持文本提取、点选提取、图片信息查看。
- 输出透明 PNG、mask PNG、预览图和 `project.json` 元数据。
- 模型按需加载，普通 UI 框选分层不需要加载大模型。

## 当前状态

当前项目处于 Phase 1 MVP 阶段。

已完成：

- 桌面 UI 基础工作流。
- 手动矩形框选分层。
- OpenCV GrabCut / 矩形 fallback 分割后端。
- 文本驱动提取：Grounding DINO 检测 + SAM2 分割。
- 点选驱动提取：SAM2 点提示分割。
- mask 后处理和透明 PNG 导出。
- 快速验证脚本和单元测试。

未完成或计划中：

- UI 画笔级 mask 编辑。
- UI 项目保存/打开。
- CLI box 提取模式。
- 更完整的模型实机回归测试。
- PSD / UMG / 批量素材导出增强。

## 环境要求

推荐环境：

- Windows
- Python 3.13.1
- 虚拟环境：`.venv/`
- NVIDIA GPU：RTX 4090 或其他 CUDA 可用显卡
- CUDA：当前项目记录环境为 CUDA 12.8

主要依赖：

- `PySide6==6.10.3`
- `torch`
- `torchvision`
- `transformers`
- `sam2`
- `opencv-python`
- `Pillow`
- `numpy`
- `click`
- `pyyaml`

注意：`PySide6==6.7.x` 不支持 Python 3.13，本项目已使用并验证 `PySide6==6.10.3`。

## 安装

在项目根目录执行：

```powershell
cd C:\ML\Deep_Claude_project
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

如果你已经有 `.venv`，直接激活并安装依赖即可：

```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

首次使用 AI 文本提取或点选提取时，模型可能会从 HuggingFace 下载权重。手动 UI 框选分层不需要下载模型。

## 快速启动 UI

启动桌面界面：

```powershell
.venv\Scripts\python.exe main.py ui
```

启动时直接打开一张图片：

```powershell
.venv\Scripts\python.exe main.py ui -i GAS2.png
```

## UI 使用指南

1. 点击 `Open Image` 打开 PNG、JPG、JPEG 或 WEBP 图片。
2. 在画布中按住鼠标左键拖拽，画出矩形选区。
3. 松开鼠标后，程序会生成青色 mask 预览。
4. 点击 `Create Layer`，把当前预览创建为图层。
5. 在右侧 `Layers` 面板中管理图层：
   - 双击名称可重命名。
   - 勾选框控制显示/隐藏。
   - `Delete` 删除当前图层。
   - `Export Layer` 导出当前图层。
   - `Export All` 导出全部图层。
6. 鼠标滚轮缩放画布。
7. 鼠标右键或中键拖动画布进行平移。

## UI 中使用自然语言切割

打开图片后，可以在顶部工具栏输入自然语言命令，例如：

```text
cut out a man with sword
cut out a man with sword and save to ./my_outputs
抠出 a man with sword 导出到 ./my_outputs
```

然后点击 `Run NL Cut`。

该功能会调用当前项目已有的 AI 管线：

```text
自然语言命令 -> 解析目标和导出目录 -> Grounding DINO 检测框 -> SAM2 分割 -> mask -> UI 图层 -> 可选自动导出
```

注意：

- 建议使用英文描述，Grounding DINO 对英文目标更稳定。
- 如果命令中包含 `save/export/output/保存/导出/输出`，UI 会在创建图层后自动导出。
- 如果命令不包含导出意图，UI 只创建图层，你可以继续手动管理和导出。
- 首次运行可能下载模型。
- 该路径会加载大模型，速度和显存占用明显高于手动框选。
- 提取完成后模型会自动卸载，释放显存。

## 导出结果

UI 的 `Export All` 会生成如下目录结构：

```text
Export/
+-- layers/
+-- masks/
+-- project.json
+-- preview.png
```

说明：

- `layers/`：裁剪后的透明 PNG 图层。
- `masks/`：裁剪后的灰度 mask PNG。
- `project.json`：图层元数据，包含原图画布坐标。
- `preview.png`：所有可见图层合成预览。

`project.json` 中每个图层包含：

```json
{
  "id": "001",
  "name": "layer_001",
  "visible": true,
  "file": "layers/001_layer_001.png",
  "mask": "masks/001_layer_001_mask.png",
  "x": 120,
  "y": 350,
  "width": 280,
  "height": 420,
  "opacity": 1.0,
  "blend_mode": "normal"
}
```

图层 PNG 和 mask PNG 会裁剪到图层 bbox，但 `project.json` 会保留图层在原始画布上的坐标。

## 命令行使用指南

查看全部命令：

```powershell
.venv\Scripts\python.exe main.py --help
```

查看图片信息：

```powershell
.venv\Scripts\python.exe main.py info -i GAS2.png
```

文本驱动提取：

```powershell
.venv\Scripts\python.exe main.py extract-text -i photo.jpg -t "a red car"
```

自然语言命令切割：

```powershell
.venv\Scripts\python.exe main.py cut -i GAS2.png -n "cut out a man with sword and save to ./my_outputs"
```

中文命令也可以：

```powershell
.venv\Scripts\python.exe main.py cut -i GAS2.png -n "抠出 a man with sword 导出到 ./my_outputs"
```

如果命令里没有写导出目录，可以用 `-o` 指定 fallback 输出目录：

```powershell
.venv\Scripts\python.exe main.py cut -i GAS2.png -n "extract a man with sword" -o ./my_outputs
```

指定输出目录：

```powershell
.venv\Scripts\python.exe main.py extract-text -i photo.jpg -t "a cat" -o my_outputs
```

跳过 mask 后处理：

```powershell
.venv\Scripts\python.exe main.py extract-text -i photo.jpg -t "a cat" --no-postprocess
```

点选驱动提取：

```powershell
.venv\Scripts\python.exe main.py extract-point -i photo.jpg --x 500 --y 375
```

使用自定义配置：

```powershell
.venv\Scripts\python.exe main.py -c configs/config.yaml extract-text -i photo.jpg -t "a cat"
```

## 配置文件

默认配置文件：

```text
configs/config.yaml
```

主要配置项：

- `device.preferred`：首选设备，通常为 `cuda`。
- `device.fallback`：fallback 设备，通常为 `cpu`。
- `models.sam2.backend`：SAM2 后端，支持 `transformers` 和 `native`。
- `models.sam2.model_id`：SAM2 HuggingFace 模型 ID。
- `models.grounding_dino.model_id`：Grounding DINO 模型 ID。
- `models.grounding_dino.box_threshold`：检测框阈值。
- `models.grounding_dino.text_threshold`：文本匹配阈值。
- `postprocess.morph`：mask 后处理参数。
- `output.default_dir`：CLI 默认输出目录。

## 项目结构

```text
app/                         # PySide6 桌面 UI
core/                        # UI 图层、mask、项目状态、导出核心逻辑
segmenters/                  # UI 分割后端
src/                         # 原始 AI 提取管线
src/detection/               # Grounding DINO 封装
src/segmentation/            # SAM2 封装和 prompt builder
src/postprocess/             # mask 后处理
src/pipeline/                # AI 提取管线编排
src/utils/                   # 图片 IO 和可视化工具
configs/                     # 配置文件
tests/                       # 单元测试和 UI smoke test
scripts/verify.py            # 项目验证入口
docs/                        # 工程文档、ADR、项目记忆
main.py                      # CLI 和 UI 统一入口
```

## 架构说明

当前项目融合了两套能力：

1. AI 模型管线
   - 自然语言或点坐标作为输入。
   - Grounding DINO 负责文本目标检测。
   - SAM2 负责目标 mask 分割。
   - 后处理模块优化 mask。
   - 输出透明 PNG。

2. 桌面图层工作流
   - PySide6 提供 UI。
   - Canvas 负责显示、缩放、平移和选区。
   - `OpenCVSegmenter` 负责本地快速分割。
   - `ProjectData` 保存当前图片和图层状态。
   - `LayerExporter` 按稳定契约导出图层、mask、预览和元数据。

核心约定：

- 内部图片格式：RGB `numpy.ndarray`，形状 `(H, W, 3)`，类型 `uint8`。
- UI 图层 mask：全画布 `uint8` mask，取值 `0..255`。
- OpenCV 操作前 mask 必须是 2D。
- 模型类不会在 `__init__` 中加载权重，必须显式调用 `load()`。
- UI 手动分层不加载大模型。

## 开发与验证

快速验证：

```powershell
.venv\Scripts\python.exe scripts\verify.py --quick
```

完整验证：

```powershell
.venv\Scripts\python.exe scripts\verify.py
```

验证 UI 命令：

```powershell
.venv\Scripts\python.exe main.py ui --help
```

无窗口环境下 smoke test：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.venv\Scripts\python.exe -c "from PySide6.QtWidgets import QApplication; from app.main_window import MainWindow; app=QApplication([]); w=MainWindow(); w.open_image_path('GAS2.png'); assert w.project.has_image; print(w.project.canvas_size); w.close()"
```

当前基线：

- `scripts/verify.py` 通过。
- 单元测试和 UI offscreen smoke test 通过。
- `main.py --help` 和 `main.py ui --help` 通过。

## 常见问题

### 1. UI 能打开，但 AI Text Extract 很慢

这是正常的。`AI Text Extract` 会加载 Grounding DINO 和 SAM2 大模型，首次运行还可能下载权重。普通手动框选分层不需要加载模型。

### 2. 文本提取没有检测到目标

可以尝试：

- 使用英文目标描述。
- 简化描述，例如把 `the tiny red car on the left` 改成 `a red car`。
- 降低 `configs/config.yaml` 中的 `box_threshold` 和 `text_threshold`。

### 3. 显存不够

可以在 `configs/config.yaml` 中把设备改为 CPU：

```yaml
device:
  preferred: "cpu"
  fallback: "cpu"
```

CPU 可以运行但速度会更慢。

### 4. PySide6 安装失败

当前项目使用 Python 3.13，因此需要支持 Python 3.13 的 PySide6 版本。本项目已验证：

```text
PySide6==6.10.3
```

如果你使用 Python 3.12 或更低版本，可以自行调整 PySide6 版本，但建议优先保持当前项目配置。

### 5. 导出的图层为什么是裁剪后的？

为了方便素材使用，图层 PNG 会按 mask bbox 裁剪。原始画布坐标保存在 `project.json` 中，可用于在下游工具中还原位置。

## 相关文档

- `docs/project_memory.md`：项目长期记忆。
- `docs/agentic_engineering.md`：AI Agent 工程协作指南。
- `docs/ui_integration.md`：桌面 UI 集成说明。
- `docs/adr/0001-agentic-engineering-baseline.md`：Agentic 工程基线 ADR。
- `docs/adr/0002-desktop-ui-layer-workflow.md`：桌面 UI 图层工作流 ADR。

## 注意事项

- 不要提交 `.venv/`、`outputs/`、`my_outputs/`、`models/`、`.test_tmp/` 等本地生成内容。
- 不要提交 `.codex/config.toml`、`.codex/settings.local.json`、`.claude/settings.local.json` 或任何 API Key。
- 模型权重和输出图片通常较大，应保持在本地忽略目录中。
