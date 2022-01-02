from pedestrians_video_2_carla.skeletons.nodes.smpl import SMPL_SKELETON
import torch


def test_mappings_are_reversible():
    random_nodes = torch.randn((4, 21*3))

    mapped = SMPL_SKELETON.map_from_original(random_nodes)

    assert mapped.shape == (4, 21, 3)

    reversed = SMPL_SKELETON.map_to_original(mapped)

    assert reversed.shape == (4, 21*3)
    assert torch.allclose(random_nodes, reversed)
