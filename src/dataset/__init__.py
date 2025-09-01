from src.dataset.expla_graphs import ExplaGraphsDataset
from src.dataset.expla_graphs_v3 import EdgePairDataset as EdgePairDataset_expla_graph
from src.dataset.webqsp import WebQSPDataset
from src.dataset.webqsp_v3 import EdgePairDataset as EdgePairDataset_webqsp
from src.dataset.scene_graphs import SceneGraphsDataset
from src.dataset.scene_graphs_v3 import EdgePairDataset as EdgePairDataset_scene_graph

load_dataset = {
    'expla_graphs': ExplaGraphsDataset,
    'expla_graphs_pair': EdgePairDataset_expla_graph,
    'webqsp': WebQSPDataset,
    'webqsp_pair': EdgePairDataset_webqsp,
    'scene_graphs': SceneGraphsDataset,
    'scene_graphs_pair': EdgePairDataset_scene_graph
}
