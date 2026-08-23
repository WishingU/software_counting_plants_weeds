# Plant and Weed Detection and Counting

[English README](README.md) |
[English annotation guide](docs/ANNOTATION_GUIDE.en.md)

本仓库提供一条可复现的四类苗期植株检测与计数流程：

- wheat
- wild oat
- brome grass
- barley grass

当前主任务是整株目标检测。每个物理植株对应一个检测框和一次计数；根部点检测、
实例分割及局部分类不属于当前主流程。

## 项目与数据布局

```text
E:/cits3200_project/
  src/plant_counter/                # 数据、训练、评估代码
  tests/                            # 自动化测试
  configs/                          # 数据和标注规则
  docs/                             # 项目及标注文档
  data/raw/                         # 原始下载数据
  data/annotations/                 # COCO 标注
  data/processed/                   # YOLO 图片、标签和构建报告
  models/                           # 预训练权重
  outputs/runs/                     # 训练、评估和审计输出
  archives/                         # 压缩备份
  legacy/                           # 已停用的旧脚本
```

大图片、模型权重和 `runs` 不提交到 Git。

## 0. 环境

```powershell
cd E:\cits3200_project
conda activate plant-count
python -m pip install -e . --no-build-isolation
```

若暂时不安装项目，也可以设置：

```powershell
$env:PYTHONPATH = "$PWD\src"
```

## 1. 审计原始标注

先根据项目的类别和几何规则检查 COCO 标注：

```powershell
python -m plant_counter.audit `
  --annotations-dir .\data\annotations `
  --dataset-dir .\data\processed `
  --rules .\configs\annotation_rules.yaml `
  --output-dir .\outputs\runs\annotation_audit
```

输出文件：

- `annotation_audit.json`：数据量、类别数量、框尺寸分布和问题数量。
- `annotation_issues.csv`：逐条问题，`error` 必须修复，`warning` 需要人工抽查。

完整标注规范见
[docs/ANNOTATION_GUIDE.zh-CN.md](docs/ANNOTATION_GUIDE.zh-CN.md)。

## 2. 构建 YOLO 数据集

```powershell
python -m plant_counter.dataset `
  --annotations-dir .\data\annotations `
  --dataset-dir .\data\processed `
  --yaml .\configs\grass.yaml `
  --report .\data\processed\build_report.json
```

构建器会检查类别映射、未知图片、重复文件名、无效框和图片/标签数量，再生成
YOLO 标签。当前预期数据量：

- train：910 张图片，10,442 个实例
- val：239 张图片，2,848 个实例

## 3. 训练

```powershell
python -m plant_counter.train `
  --data .\configs\grass.yaml `
  --model .\models\yolo26n.pt `
  --epochs 100 `
  --imgsz 1280 `
  --batch -1 `
  --device 0 `
  --project .\outputs\runs\detect `
  --name whole_plant_v1
```

训练入口固定随机种子、确定性模式和 Windows 安全的 worker 数，并保存环境信息。
第一次清理流程时不要同时更换模型、分辨率和标注规则；一次只改变一个主要变量。

## 4. 验证与计数分析

```powershell
python -m plant_counter.evaluate `
  --model .\outputs\runs\detect\whole_plant_v1\weights\best.pt `
  --data .\configs\grass.yaml `
  --split val `
  --conf 0.25 `
  --iou 0.7 `
  --device 0 `
  --output-dir .\outputs\runs\evaluation\whole_plant_v1
```

输出包括 Precision、Recall、mAP50、mAP50-95，以及总体和各物种的计数 MAE、
RMSE、bias、exact-match rate 与逐图预测数量。模型选择看 mAP50-95；实际计数部署
还必须同时检查 count MAE 和 bias。

## 5. 推荐的迭代顺序

1. 修复标注审计中的 `error`，抽查 `warning`。
2. 固定数据版本训练一个基线。
3. 从逐图计数 CSV 中挑选绝对误差最大的照片。
4. 将问题归入漏检、重复框、错分类、遮挡或域差异。
5. 只针对主要错误补标/补数据，再训练下一版本。
6. 最终只在独立 test 集上报告一次结果，不能根据 test 结果反复调参。

## 测试

```powershell
conda activate plant-count
$env:PYTHONPATH = "$PWD\src"
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
```
