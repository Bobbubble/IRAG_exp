# 在处理scene_graph之前，请从google drive链接下载gqa文件夹，拷贝到/dataset目录下
# 1.1 prepare grag dataset (理论上三个可并行)
python -m src.dataset.preprocess.expla_graphs
# python -m src.dataset.preprocess.webqsp
# python -m src.dataset.preprocess.scene_graphs

# 1.2 preprocess grag dataset (理论上三个可并行)
# python -m src.dataset.expla_graphs
# python -m src.dataset.webqsp
# python -m src.dataset.scene_graphs


# 1.3 train grag model  (暂时不进行)
# python train.py --dataset scene_graphs

# ---------------------9.2 update----------------------------
# 1.4 train plugin_lora on scenegraph (需要约24h左右。)
# step1: 删除原来/dataset/scene_graphs/split下的三个txt文件，把根目录下的1.4_test_indices.txt，1.4_train_indices.txt，1.4_val_indices.txt拷贝进去。
# step2: python train_v3.py --dataset scene_graphs_pair --llm_frozen False



