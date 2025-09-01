# 在处理scene_graph之前，请从google drive链接下载gqa文件夹，拷贝到/dataset目录下
# 1.1 prepare grag dataset
python -m src.dataset.preprocess.expla_graphs
# python -m src.dataset.preprocess.webqsp
# python -m src.dataset.preprocess.scene_graphs

# 1.2 preprocess grag dataset
# python -m src.expla_graphs
# python -m src.webqsp
# python -m src.scene_graphs
# 注意，当处理完scene_graphs的时候，需要删除原来/dataset/scene_graphs/split下的三个txt文件，把根目录下三个txt文件拷贝进去。

# 1.3 train grag model
# python train.py --dataset scene_graphs

