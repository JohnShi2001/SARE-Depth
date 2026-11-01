# SARE-Depth

**Scale Adaptation and Range Experts for monocular metric-depth estimation**

## Release status / 发布状态

**Research materials are being prepared for upload. All materials will be reviewed and approved by the author before they are made public.**

This initial repository contains project information and this release notice only. The study-specific code, configuration files, evaluation scripts, and trained weights have not yet been uploaded. The author will determine the scope and timing of subsequent uploads after review.

**相关资料正在整理并准备上传，须经作者审核后方可公开。**

当前仓库仅提供项目介绍和发布状态说明。研究代码、配置文件、评价脚本及训练权重尚未上传。后续公开内容与上传时间由作者审核后确定。

## Associated manuscript / 对应论文

**Monocular Metric-Depth Adaptation for Agricultural Perception: A CitrusFarm Case Study with Embedded Inference**

**Authors:** Zhuoyu Shi, Sen Yang, and Xiangzhou Li  
**Affiliation:** Gansu Agricultural University, Lanzhou, China

## Research overview / 研究概述

SARE-Depth investigates parameter-efficient, supervised output-side adaptation of a compact teacher-guided monocular metric-depth predictor. The framework combines global-scale and local-residual correction with subsequent depth-conditioned residual refinement.

The study includes quantitative depth evaluation using public datasets, including CitrusFarm, and embedded-inference measurements on Jetson Orin NX. Separately collected agricultural-field images are used for exploratory observations. Headland detection is an intended downstream application, not a complete detection or vehicle-control system evaluated in this study.

SARE-Depth 研究紧凑单目度量深度模型的参数高效监督适配，通过全局尺度与局部残差校正，以及后续距离条件残差细化调整模型输出。研究包含公共数据集上的深度评价和 Jetson Orin NX 嵌入式推理测量。作者自采农田图像用于探索性观察；地头检测为下游应用目标，不代表本文已完成完整检测或车辆控制系统验证。

## Materials planned for review / 拟审核资料

Subject to author review, subsequent updates may include:

- Study-specific implementation and experiment configurations.
- Inference, preprocessing, and evaluation scripts.
- Checkpoint information and approved trained weights.
- Dataset access instructions and experiment-to-result mappings.

拟审核内容包括本研究实现与实验配置、推理及评价脚本、模型权重，以及数据获取和实验结果对应说明。上述为拟整理范围，不表示这些文件目前已经可以获取。

Third-party datasets and software remain subject to their original licenses and access conditions. Any future release will identify the applicable dependencies and permissions.

第三方数据与软件遵循原始许可及获取条件。后续发布将说明相关依赖与许可要求。
