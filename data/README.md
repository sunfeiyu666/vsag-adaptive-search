# 数据集说明

本目录存放 ANN 自适应检索实验所用数据集。由于文件总大小约 4GB，且 GitHub 单文件上限为 100MB，**实际数据文件未上传到本仓库**，仅在此记录数据集名称与信息。

## 数据集列表

| 文件名 | 大小 | 说明 |
|---|---|---|
| `cohere_train.f32` | 2.9 GB | Cohere 文本嵌入基底向量（训练集），float32 |
| `cohere_test.f32` | 2.9 MB | Cohere 查询向量，float32 |
| `cohere_groundtruth.i32` | 3.8 MB | Cohere 真值近邻索引，int32 |
| `sift-128-euclidean.hdf5` | 500.8 MB | SIFT1M，128 维，欧氏距离 |
| `nytimes-256-angular.hdf5` | 300.6 MB | NYTimes，256 维，余弦距离（angular） |
| `fashion-mnist-784-euclidean.hdf5` | 217.0 MB | Fashion-MNIST，784 维，欧氏距离 |
| `coco-t2i-512-angular.hdf5` | 135.7 MB | COCO text-to-image，512 维，余弦距离（angular） |

## 数据来源

这些均为 [ann-benchmarks](https://ann-benchmarks.com/) 标准评测数据集，可从其官网或官方数据仓库下载：

- SIFT1M / Fashion-MNIST / NYTimes：<http://corpus-texmex.irisa.fr/> 及 ann-benchmarks 数据源
- Cohere 数据集：<https://huggingface.co/datasets/Cohere/wikipedia-22-12>
- COCO text-to-image：ann-benchmarks 提供的 `coco-t2i-512-angular.hdf5`

## 使用方式

将下载后的数据文件放回本目录（`data/`），保持文件名不变即可，`scripts/` 下的脚本默认从 `../data/` 读取。

涉及文件：`cohere_train.f32`、`cohere_test.f32`、`cohere_groundtruth.i32`、`sift-128-euclidean.hdf5`、`nytimes-256-angular.hdf5`、`fashion-mnist-784-euclidean.hdf5`、`coco-t2i-512-angular.hdf5`
