# Brain MRI Segmentation Based on U-Net

本仓库用于《医学图像处理技术》期末项目：**基于 U-Net 的脑 MRI 低级别胶质瘤区域自动分割**。

项目使用 Kaggle Brain MRI segmentation / LGG MRI Segmentation 数据集，完成从数据读取、预处理、U-Net 训练、指标评价到可视化结果生成的完整流程。

## 1. 项目内容

- 数据集：Kaggle Brain MRI segmentation
- 任务：脑 MRI 中低级别胶质瘤 LGG 的 FLAIR 异常区域二分类分割
- 模型：
  - U-Net Baseline：BCE Loss
  - Improved：BCE + Dice Loss + 数据增强
- 指标：
  - Dice
  - IoU
  - Precision
  - Recall

## 2. 仓库结构

```text
Brain_MRI_Segmentation/
├── code/
│   ├── check_data.py          # 检查 Kaggle 数据路径、image-mask 数量与匹配情况
│   ├── train_mri.py           # 训练、验证、测试、保存曲线和预测图
│   ├── compare_metrics.py     # 合并 baseline 和 improved 指标表
│   └── package_results.py     # 打包最终结果
├── docs/
│   ├── kaggle_run_guide.md    # Kaggle Notebook 运行步骤
│   └── report_appendix.md     # 报告末尾可附加的仓库说明文字
├── results/
│   └── Brain_MRI_Segmentation_Final_Results.zip              # 结果文件
├── requirements.txt
├── .gitignore
└── README.md
```

## 3. Kaggle 运行环境

本项目主要在 Kaggle Notebook 线上运行。建议设置：

```text
Accelerator: GPU T4
Python: Kaggle Notebook 默认 Python 环境
Data path:
/kaggle/input/datasets/mateuszbuda/lgg-mri-segmentation/kaggle_3m
```

如果你的数据集挂载路径不同，请以 Kaggle Notebook 中 `/kaggle/input` 下实际路径为准。

## 4. Kaggle 一键运行命令

在 Kaggle Notebook 中，假设已经上传或克隆本仓库代码，并位于仓库根目录，可以运行：

```bash
DATA_DIR="/kaggle/input/datasets/mateuszbuda/lgg-mri-segmentation/kaggle_3m"

python code/check_data.py --data_dir "$DATA_DIR"

python code/train_mri.py \
  --mode baseline \
  --data_dir "$DATA_DIR" \
  --epochs 10 \
  --batch_size 8 \
  --num_workers 2 \
  --output_dir /kaggle/working/outputs_baseline

python code/train_mri.py \
  --mode improved \
  --data_dir "$DATA_DIR" \
  --epochs 10 \
  --batch_size 8 \
  --num_workers 2 \
  --output_dir /kaggle/working/outputs_improved

python code/compare_metrics.py \
  --baseline_csv /kaggle/working/outputs_baseline/metrics_summary_baseline.csv \
  --improved_csv /kaggle/working/outputs_improved/metrics_summary_improved.csv \
  --output_dir /kaggle/working/final_results

python code/package_results.py \
  --baseline_dir /kaggle/working/outputs_baseline \
  --improved_dir /kaggle/working/outputs_improved \
  --final_dir /kaggle/working/final_results \
  --package_dir /kaggle/working/Brain_MRI_Segmentation_Final_Results
```

运行结束后，最终结果压缩包一般位于：

```text
/kaggle/working/Brain_MRI_Segmentation_Final_Results.zip
```

## 5. 实验结果

本次实验中，数据划分为：

| 数据集 | 样本数 |
|---|---:|
| 训练集 | 2750 |
| 验证集 | 589 |
| 测试集 | 590 |

测试集结果如下：

| 模型 | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| U-Net Baseline | 0.8333 | 0.7143 | 0.8619 | 0.8066 |
| U-Net + BCE Dice + Aug | 0.8292 | 0.7083 | 0.8080 | 0.8516 |

结果分析：Baseline 在 Dice、IoU 和 Precision 上略高，说明整体重叠度和误分控制更好；Improved 模型 Recall 更高，说明其能够识别出更多真实病灶区域，但可能带来一定过分割。

## 6. 结果文件说明

实验结果可放在 `results/` 目录下，例如：

```text
results/
└── Brain_MRI_Segmentation_Final_Results.zip
```

压缩包中建议包含：

```text
curves/                       # loss 曲线与 Dice/IoU 曲线
metrics/                      # 指标表
prediction_visualizations/    # 测试集分割可视化图
training_logs/                # 训练日志
weights/                      # baseline / improved 模型权重
实验结果说明.txt
```

注意：模型权重 `.pth` 和结果压缩包可能比较大。如果普通 GitHub 上传失败，可以改用 Git LFS 或 GitHub Releases 存放大文件。

## 7. 复现实验

本仓库不包含 Kaggle 原始数据集。复现实验时需要先在 Kaggle Notebook 中挂载数据集：

- Dataset: Brain MRI segmentation / LGG MRI Segmentation
- Data directory example:
  `/kaggle/input/datasets/mateuszbuda/lgg-mri-segmentation/kaggle_3m`

然后按第 4 节命令运行即可。

## 8. 项目成员说明

个人报告对应成员：

- 姓名：张艺宝
- 学号：102301311
- 个人分工：方案设计 / 技术路线 / 项目背景与数据集介绍
