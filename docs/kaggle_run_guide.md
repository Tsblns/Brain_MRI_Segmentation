# Kaggle Notebook 运行指南

本项目主要在 Kaggle Notebook 线上运行，因此代码默认适配 Kaggle 路径。

## 1. 开启 GPU

在 Kaggle Notebook 右侧设置中选择：

```text
Settings → Accelerator → GPU T4
```

然后运行：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

正常应输出 `True` 和 `Tesla T4`。

## 2. 确认数据路径

本次实验使用的数据路径为：

```text
/kaggle/input/datasets/mateuszbuda/lgg-mri-segmentation/kaggle_3m
```

如果你的路径不同，运行：

```python
import os
for root, dirs, files in os.walk("/kaggle/input"):
    if "kaggle_3m" in root:
        print(root)
        break
```

## 3. 检查数据集

```bash
DATA_DIR="/kaggle/input/datasets/mateuszbuda/lgg-mri-segmentation/kaggle_3m"

python code/check_data.py --data_dir "$DATA_DIR"
```

应看到约 3929 张原图、3929 张 mask，缺失 mask 数量为 0。

## 4. 训练 Baseline

```bash
python code/train_mri.py \
  --mode baseline \
  --data_dir "$DATA_DIR" \
  --epochs 10 \
  --batch_size 8 \
  --num_workers 2 \
  --output_dir /kaggle/working/outputs_baseline
```

## 5. 训练改进模型

```bash
python code/train_mri.py \
  --mode improved \
  --data_dir "$DATA_DIR" \
  --epochs 10 \
  --batch_size 8 \
  --num_workers 2 \
  --output_dir /kaggle/working/outputs_improved
```

## 6. 生成最终指标对比表

```bash
python code/compare_metrics.py \
  --baseline_csv /kaggle/working/outputs_baseline/metrics_summary_baseline.csv \
  --improved_csv /kaggle/working/outputs_improved/metrics_summary_improved.csv \
  --output_dir /kaggle/working/final_results
```

## 7. 打包结果

```bash
python code/package_results.py \
  --baseline_dir /kaggle/working/outputs_baseline \
  --improved_dir /kaggle/working/outputs_improved \
  --final_dir /kaggle/working/final_results \
  --package_dir /kaggle/working/Brain_MRI_Segmentation_Final_Results
```

最终下载：

```text
/kaggle/working/Brain_MRI_Segmentation_Final_Results.zip
```
