# IdleBrain Frontend / 前端说明

## English
Run full UI + backend bridge:

```bash
cd frontend
python server.py
```

Open: `http://127.0.0.1:8787`

### Workflow Modes
- **One-Click Mode (recommended)**
  - Source TIFF -> Single-layer or Whole-brain -> Start
  - Auto-pick atlas + auto registration + enter manual review
- **Professional Mode**
  - Full control for atlas path, alignment mode, registration parameters, and overlay options

### Manual Review & Learning
- Liquify drag correction in preview canvas
- Manual landmark correction panel
- `Save Calibration + Learn` to append training pair and trigger auto-learning
- Sample library cap protection:
  - env: `IDLEBRAIN_MAX_CALIB_SAMPLES`
  - default: `180`

### Export
- Quick export current preview to: `png`, `tif`, `jpg`, `bmp`
- Figure export with annotations via canvas export button

### Desktop mode (double-click EXE)
```bash
cd frontend
build_desktop.bat
```
Then run `dist/IdleBrainUI.exe`.

## 中文
运行前端与后端桥接服务：

```bash
cd frontend
python server.py
```

浏览器打开：`http://127.0.0.1:8787`

### 流程模式
- **一键模式（推荐）**
  - 源 TIFF -> 单层配准或全脑配准 -> 开始
  - 自动选图谱、自动配准，然后进入人工复审
- **专业模式**
  - 可完整控制图谱路径、配准模式、参数与叠加选项

### 人工复审与学习
- 预览画布支持 Liquify 拖拽微调
- 支持手动地标校准
- 点击 `Save Calibration + Learn` 可写入训练样本并触发自动学习
- 样本库阈值保护：
  - 环境变量：`IDLEBRAIN_MAX_CALIB_SAMPLES`
  - 默认值：`180`

### 导出
- 可快速导出当前预览为：`png`、`tif`、`jpg`、`bmp`
- 也可通过画布导出带标注的图像

---
See `../README.md` and `../project/README.md` for full architecture and pipeline docs.
