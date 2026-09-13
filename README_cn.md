# 植物与杂草计数

本项目提供基于 Ultralytics YOLO 的作物/杂草计数与四物种识别流程，包括
Streamlit 网页、命令行推理、数据处理工具、评估脚本、正式模型和自动化测试。

[English README](README.md) · [Windows 配置](README_WINDOWS.md) ·
[英文标注指南](docs/ANNOTATION_GUIDE.en.md)

## 目录结构

~~~text
app.py                 Streamlit 网页入口
count_plants.py        命令行推理入口
src/plant_counter/     正式的数据、审计、训练和评估代码
scripts/               数据转换、图片预处理和诊断工具
tests/                 自动化测试
configs/               配置示例和标注规则
data/                  数据集 YAML 和示例图片
models/counting/       作物/杂草计数模型
models/species/        四物种识别模型
runs/、outputs/        本地运行结果，不提交 Git
notebooks/             探索性切图 notebook
~~~

## 安装

~~~powershell
conda activate plant-count
python -m pip install -e ".[app,tools]" --no-build-isolation
~~~

只安装网页部署依赖：

~~~powershell
python -m pip install -r requirements.txt
~~~

## 启动网页

~~~powershell
python -m streamlit run app.py
~~~

网页会自动发现 models/ 和 runs/ 中的模型，默认使用
models/counting/yolov8n-50e.pt。

推理参数侧栏中的 **Enhance green vegetation** 可以开启绿色增强。它只适度增强
本来就偏绿的区域，预览图就是实际送入模型的图像。该功能默认关闭。

## 命令行推理

~~~powershell
python count_plants.py data/raw_images/sample.png --device 0
python count_plants.py data/raw_images/sample.png --device 0 --enhance-green
~~~

命令行添加 --enhance-green 即可开启与网页相同的预处理。

## 训练与评估

~~~powershell
python -m plant_counter.train --data data/yolo/data.yaml --model yolov8n.pt --project outputs/runs/detect --name crop_weed_v1

python -m plant_counter.evaluate --model models/counting/yolov8n-50e.pt --data data/yolo/data.yaml --split test --output-dir outputs/runs/evaluation/crop_weed
~~~

## 图片预处理

~~~powershell
# 3×3 切图
python scripts/split_images.py data/source data/split --grid 3

# 只处理文件名包含 wild 的图片，并进行 2×2 切图
python scripts/split_images.py data/source data/wild_split --grid 2 --name-contains wild

# HEIC/HEIF 转 JPEG
python scripts/heic_to_jpeg.py data/heic data/jpeg
~~~

## 测试

~~~powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
~~~
