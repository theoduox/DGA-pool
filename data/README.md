# 数据与权重使用说明

代码包中的examples/包含三张ROI、对应原图与预期结果，以及19.56秒的正常状态拼接视频，配合weights/enhancement_realtime.pth运行。无需下载完整数据即可尝试首页样例。

## 下载与放置

| 下载包 | ZIP顶层目录 | 解压到项目后的路径 |
|---|---|---|
| dual_stage_melt_pool_images.zip | dataset-images/ | data/dataset-images/ |
| dual_stage_melt_pool_videos.zip | dataset-videos/ | data/dataset-videos/ |

两个包均解压到项目data/下，分别进入自己的目录，无需合并或覆盖文件。代码包中的dataset-images/、dataset-videos/只含.gitkeep占位；下载后才有完整数据、说明、许可、引用与索引。只做图像实验可不下载视频包。

图像包包含13,157张原图、对应ROI、metadata.csv和四套历史划分；视频包包含60段AVI及videos.csv。图像包也保留一份来源视频索引，其path以../dataset-videos/videos/指向旁边的视频目录；视频包内videos.csv的path以videos/指向自身视频。CSV路径均相对于各自CSV所在目录。


## 分类输入

分类训练和评估入口接收已有目录，每个目录内包含以下类别文件夹：

| 文件夹 | 索引 | 状态 |
|---|---:|---|
| excessive（兼容Excessive） | 0 | 过熔 |
| insufficient（兼容Insufficient） | 1 | 能量不足/未熔合 |
| normal（兼容Normal） | 2 | 正常 |

训练、验证、独立评估目录由运行时分别传入，也支持--split-file配合--train-subset/--val-subset或评估的--subset从同一图像目录选择集合，不自动生成划分。历史DividedBySubjects的test用于训练期选权重、val用于独立评估，复现时按用途传入，保留原目录名称。

图像标签按类别目录读取。文件名中的状态前缀是历史信息。原始工况命名中的两个数值对应激光功率W和扫描速度mm/min；末尾数字保留从1开始计数的视频帧号；用户确认抽帧后手动删除首尾纯黑图像。

## 图像处理与模型输入

- 增强训练：递归图像目录，转RGB、resize后归一化至[0,1]，不需要分类标签。
- 图像增强/ROI：接收单图或递归目录，输出保留相对目录，原图名后追加.png。
- 分类：RGB，224×224，使用对应配置的resize或center-crop及ImageNet归一化。
- 权重：使用与模型结构对应的裸state_dict，通过命令行显式传入。

