# LIBERO 数据集下载、查看、训练与测试教程

本教程以 `libero_object` 为例，覆盖官方数据下载、`hdf5` 内容查看、本地生成数据，以及训练和测试。

## 0. 环境准备

```bash
pip install -r requirements.txt
pip install -e .
export DATA_ROOT=libero_data
mkdir -p $DATA_ROOT
```

## 1. 下载官方子集

```bash
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_object \
  --download-dir $DATA_ROOT/official \
  --use-huggingface
```

## 2. 查看 `hdf5` 里有什么

官方 `*_demo.hdf5` 已经是训练可用格式，不需要额外解码。若要查看内容，可先打印结构：

```bash
python scripts/get_dataset_info.py \
  --dataset $DATA_ROOT/official/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

如果你想把一个目录里的所有 `hdf5` 都展开成普通文件目录：

```bash
python scripts/unpack_hdf5_dataset.py \
  --input-dir $DATA_ROOT/official/libero_object
```

输出内容通常包括：
- `attrs.json`：任务和环境元数据
- `*.npy`：动作、状态、奖励、图像张量
- `*_frames/`：完整图像帧和对应 `video.mp4`

## 3. 本地生成数据

有图形界面时可人工采集：

```bash
python scripts/collect_demonstration.py \
  --device keyboard \
  --num-demonstration 50 \
  --directory $DATA_ROOT/raw_demos \
  --bddl-file libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl
```

无图形界面时可用已训练策略无头采集：

```bash
python scripts/headless_collect_from_policy.py \
  --benchmark libero_object \
  --task-id 0 \
  --algo base \
  --policy bc_rnn_policy \
  --seed 0 \
  --load_task 9 \
  --device-id 0 \
  --num-demos 50 \
  --output $DATA_ROOT/raw_policy/libero_object_task0_demo.hdf5
```

原始 `demo.hdf5` 需要转换成训练格式：

```bash
python scripts/create_dataset.py \
  --demo-file <path/to/demo.hdf5> \
  --use-camera-obs
```

## 4. 训练与测试

用官方下载子集训练：

```bash
python libero/lifelong/main.py \
  benchmark_name=LIBERO_OBJECT \
  policy=bc_rnn_policy \
  lifelong=base \
  seed=0 \
  folder=$DATA_ROOT/official
```

测试：

```bash
python libero/lifelong/evaluate.py \
  --benchmark libero_object \
  --task_id 0 \
  --algo base \
  --policy bc_rnn_policy \
  --seed 0 \
  --load_task 9 \
  --device_id 0
```
