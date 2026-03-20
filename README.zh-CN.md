[English →](README.md)

---

# Brainfast

**清脑样本荧光显微镜数据的脑图谱配准与细胞计数工具。**

Brainfast 是一款本地运行、数据不出本机的桌面工具，将荧光显微镜切片配准到 [Allen 小鼠脑图谱（CCFv3）](https://atlas.brain-map.org/)，并按解剖区域统计标记细胞数量，提供浏览器 UI、完整复现元数据，无需上传至云端。

> 专为处理清脑或全脑样本 lightsheet/共聚焦 TIFF Z-stack 的神经科学实验室设计。

---

## 功能一览

| 步骤 | 说明 |
|------|------|
| **AP 自动选层** | 按文件名 Z 坐标或模板互相关，将每张切片匹配到图谱冠状面 |
| **图像配准** | 仿射初定位 → 薄板样条（TPS）非线性形变，使组织边界与图谱轮廓吻合 |
| **细胞检测** | 多尺度 LoG（内置）或 Cellpose 实例分割，支持逐通道配置 |
| **去重** | 三维 KD-tree 聚类，消除跨切片重复检测 |
| **脑区映射** | 每个细胞质心在 CCFv3 注释体积中查找对应区域 |
| **分层统计** | 沿 Allen 本体树（叶节点 → 区域 → 分区 → …）逐级汇总细胞数 |
| **QC 与导出** | Z 连续性图表、每切片 SSIM、Excel/CSV 导出、自动生成方法段落 |

---

## 核心特性

- **浏览器 UI** — 四 Tab 单页应用（流程 / QC / 结果 / 项目），只需 Python，无需额外安装
- **中英双语** — 完整 EN/ZH 界面切换，所有标签和提示均已翻译
- **Garwood 95% CI** — 每个脑区计数附带 Wilson-Hilferty 泊松置信区间
- **图谱指纹** — `annotation_25.nii.gz` 的 SHA-256 写入 `detection_summary.json`，保障复现性
- **项目与批处理队列** — SQLite 样本管理，FIFO 后台批处理工作线程
- **跨样本对比** — 将多次运行的叶区 CSV 合并成透视表
- **多通道共表达** — 逐区域并排显示红/绿通道细胞数
- **三维体积流程** — 通过 ANTs 或 Elastix 进行全脑体积配准，生成 HTML 报告
- **亮色/暗色主题** — localStorage 持久化主题切换
- **Docker 支持** — `Dockerfile` + `docker-compose.yml`，可无头部署于 Linux 服务器
- **97 个单元测试**，GitHub Actions CI（Windows + Ubuntu）

---

## 快速开始

### 环境要求

- Python 3.10 或 3.11
- Windows 10/11（主要）· Linux（通过 Docker）
- `annotation_25.nii.gz` — Allen CCFv3 25 µm 注释文件（放置于 `project/` 目录）
- 建议 NVIDIA GPU（Cellpose 使用）；LoG 检测器可在 CPU 上运行

### 安装

```powershell
git clone https://github.com/Skiyoshika/Brainfast.git
cd Brainfast
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[advanced,dev]"
```

### 启动

```powershell
# Windows — 双击或命令行：
.\Start_Brainfast.bat

# 直接启动：
python project\frontend\server.py
```

浏览器打开 **http://127.0.0.1:8787**。

### Docker（Linux 服务器）

```bash
docker compose up -d
# 打开 http://localhost:8787
```

`BRAINFAST_HEADLESS=1` 已在 Docker 中默认启用，禁用 tkinter 文件浏览对话框。

---

## 完整流程

```
输入 TIFF 切片（Z-stack）
      │
      ▼
AP 自动选层 ──── atlas_autopick.py
      │
      ▼
配准：仿射初定位 → TPS 非线性形变
      │
      ▼
人工审核 / Liquify 矫正  ← 浏览器 UI
      │
      ▼
细胞检测（LoG · Cellpose · 逐通道）
      │
      ▼
三维去重（KD-tree，可配置半径）
      │
      ▼
脑区映射（CCFv3 注释查找）
      │
      ▼
分层聚合 + Garwood 95% CI
      │
      ▼
cell_counts_leaf.csv · cell_counts_hierarchy.csv
Excel 导出 · 方法段落 · Z 连续性图表
```

---

## 输出文件

| 文件 | 内容 |
|------|------|
| `cell_counts_leaf.csv` | 叶节点脑区计数，含 `ci_low`、`ci_high`、形态列 |
| `cell_counts_hierarchy.csv` | 沿 Allen 本体树逐级汇总的计数 |
| `cells_dedup.csv` | 去重后细胞质心（x, y, z_µm, score, region_id） |
| `detection_summary.json` | 检测器、采样模式、总数、`atlas_sha256` |
| `slice_registration_qc.csv` | 每张切片的边缘 SSIM 指标 |
| `z_smoothness_report.json` | AP 轴连续性分析（离群点标记） |
| `brainfast_results.xlsx` | 三 Sheet Excel：层次 / 叶区 / 运行参数 |

UI job 输出位于 `project/outputs/jobs/<job_id>/`。

---

## API（REST）

Flask 服务器提供完整文档化的 REST API。
完整 endpoint 和 `curl` 示例见 [docs/api_reference.md](docs/api_reference.md)。

关键接口：

```bash
POST /api/run               # 启动流程
GET  /api/poll?job=...      # 统一状态 + 日志尾 + 错误（替代3个独立轮询循环）
GET  /api/outputs/excel     # 下载 Excel
GET  /api/outputs/z-continuity   # Z 轴连续性 JSON
POST /api/compare/regions   # 跨样本透视表
```

所有错误响应均含机器可读的 `error_code` 常量（如 `PIPELINE_ALREADY_RUNNING`、`CONFIG_PATH_DENIED`）。

---

## 科学方法

配准各阶段、Garwood CI 推导、Z 连续性检测、图谱指纹等算法细节，见 [docs/science_methods.md](docs/science_methods.md)（英文）。

**方法段落模板**（UI 自动生成）：

> 脑图谱配准使用 Brainfast v0.5 完成（运行时间：…）。显微图像分辨率为 N µm/像素。图谱配准参照 Allen 小鼠脑图谱（CCFv3，annotation_25.nii.gz，体素间距 25 µm，sha256: …），采用非线性变换（薄板样条）方法对切片进行空间配准。配准质量通过边缘 SSIM 评估。细胞计数采用 LoG 检测器，基于原始单切片完成去重和图谱统计。置信区间采用 Garwood 方法（95% 泊松 CI）。荧光通道：红色。

---

## 软件架构

```
project/
├── frontend/
│   ├── server.py              Flask 入口（70 行编排层）
│   ├── server_context.py      共享运行时状态、job 隔离、GC
│   ├── blueprints/            11 个 API Blueprint
│   │   ├── api_pipeline.py    run / cancel / poll / preflight / methods-text
│   │   ├── api_outputs.py     CSV / Excel / Z-连续性 / AP-密度 / 共表达
│   │   ├── api_projects.py    项目 + 样本 CRUD（SQLite）
│   │   ├── api_batch.py       FIFO 批处理队列
│   │   ├── api_compare.py     跨样本脑区对比
│   │   └── …
│   ├── index.html             单页 UI（4 Tab，中英双语）
│   ├── app.js                 前端 JS（~3500 行，完整 i18n）
│   └── styles.css             暗色/亮色主题 CSS 变量
├── scripts/
│   ├── main.py                二维流程入口
│   ├── detect.py              LoG + Cellpose 检测
│   ├── map_and_aggregate.py   脑区映射 + 分层统计 + Garwood CI
│   ├── z_smoothness.py        AP 轴连续性分析
│   └── …
└── tests/
    ├── unit/                  97 个测试，无需图谱文件
    └── integration/           需要 annotation_25.nii.gz
```

**v0.5+ 安全约束：**
- 配置路径经 containment 检查，限制在 `PROJECT_ROOT/configs` 和 `OUTPUT_DIR` 内
- `running = True` 在 `_run_state_lock` 内、线程启动前设置，无竞态条件
- `_job_states` 上限 200 条，LRU 驱逐，无内存泄漏

---

## 测试

```powershell
# 单元测试（无需图谱文件）
python -m pytest project/tests/unit -v

# 含覆盖率（CI 要求 ≥60%）
python -m pytest project/tests/unit --cov=project/scripts --cov-report=term-missing

# Lint
ruff check project/scripts/ project/frontend/blueprints/
```

CI：[GitHub Actions](.github/workflows/test.yml) — 每次推送/PR 到 `main` 自动运行单元测试（Windows + Ubuntu）+ ruff 检查。

发布：[release workflow](.github/workflows/release.yml) — 打 `v*.*.*` tag → 自动构建 Windows EXE → 上传 GitHub Releases。

---

## 仓库结构

| 路径 | 用途 |
|------|------|
| `project/configs/` | 运行配置、Allen 元数据、样例预设 |
| `project/frontend/` | Flask 应用、UI 资源、桌面启动器 |
| `project/scripts/` | 配准、检测、映射、聚合、工具脚本 |
| `project/tests/` | 单元测试和集成测试 |
| `project/train_data_set/` | 人工校准样本（17 组） |
| `docs/` | [用户手册](docs/user_guide.md)（英文）· [API 参考](docs/api_reference.md)（英文）· [科学方法](docs/science_methods.md)（英文） |

---

## 许可证

见 [LICENSE](LICENSE)。
