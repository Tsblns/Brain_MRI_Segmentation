# 报告末尾可添加的仓库说明

可将下面文字添加到报告“附录：实验输出文件说明”之后。

---

附录：项目代码仓库说明

本项目代码和实验结果已整理并上传至 GitHub 仓库，仓库中包含数据检查脚本、U-Net 训练脚本、指标合并脚本、结果打包脚本以及 Kaggle Notebook 运行说明。由于本项目主要在 Kaggle Notebook 线上环境中完成训练，代码中的默认运行路径以 Kaggle 数据集挂载目录和 `/kaggle/working/` 输出目录为主。

GitHub 仓库地址：

`https://github.com/你的用户名/Brain_MRI_Segmentation`

仓库主要内容包括：

1. `code/check_data.py`：检查数据集路径、image-mask 数量及匹配情况；
2. `code/train_mri.py`：完成 U-Net baseline 和改进模型的训练、验证、测试、曲线绘制和预测可视化；
3. `code/compare_metrics.py`：合并 baseline 与 improved 的测试集指标；
4. `code/package_results.py`：整理并打包训练日志、指标表、曲线图、预测可视化图和模型权重；
5. `docs/kaggle_run_guide.md`：Kaggle Notebook 运行步骤说明；
6. `results/`：实验结果压缩包或结果文件说明。

本仓库不直接包含 Kaggle 原始数据集。复现实验时，需要先在 Kaggle Notebook 中挂载 Brain MRI segmentation 数据集，再按照仓库 README 中的命令运行。
