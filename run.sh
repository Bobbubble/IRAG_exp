# 在处理scene_graph之前，请从google drive链接下载gqa文件夹，拷贝到/dataset目录下
# 1.1 prepare grag dataset (理论上两个可并行)
# python -m src.dataset.preprocess.webqsp_para
# python -m src.dataset.preprocess.scene_graphs

# 1.2 preprocess grag dataset (理论上两个可并行)
# python -m src.dataset.webqsp_para
# python -m src.dataset.scene_graphs

# 1.3 train grag model on scenegraph (约20h)
# step1: 删除原来/dataset/scene_graphs/split下的三个txt文件，把根目录下的1.3_test_indices.txt，1.3_train_indices.txt，1.3_val_indices.txt拷贝进去,
#        去掉前缀，重命名为test_indices.txt，train_indices.txt，val_indices.txt
# python train.py --dataset scene_graphs

# 1.4 train plugin_lora on scenegraph (一个需要约24h左右，两个可并行)
# step1: 删除原来/dataset/scene_graphs/split下的三个txt文件，把根目录下的1.4_test_indices.txt，1.4_train_indices.txt，1.4_val_indices.txt拷贝进去。
#        去掉前缀，重命名为test_indices.txt，train_indices.txt，val_indices.txt
# step2.1 train without lora:
# python train_v3.py --dataset scene_graphs_pair --llm_frozen True
#
# step2.2 train with lora:
# python train_v3.py --dataset scene_graphs_pair --llm_frozen False



