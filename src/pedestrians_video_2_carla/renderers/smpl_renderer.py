from functools import lru_cache
from typing import List, Tuple, Any, Dict
from human_body_prior.body_model.body_model import BodyModel
from body_visualizer.mesh.mesh_viewer import MeshViewer
import os
import numpy as np
import trimesh
import torch

from pedestrians_video_2_carla.renderers.renderer import Renderer


BODY_MODEL_DIR = os.path.join(os.getcwd(), 'models', 'smpl-x', 'smplx_locked_head')
MODELS = {
    'male': os.path.join('male', 'model.npz'),
    'female': os.path.join('female', 'model.npz'),
    'neutral': os.path.join('neutral', 'model.npz')
}


class SMPLRenderer(Renderer):
    """
    This renderer basically does the same job as `body_visualizer.tools.vis_tools.render_smpl_params`.

    TODO: Rewrite to use pytorch3d instead of pyrender?.
    """

    def __init__(self, body_model_dir: str = BODY_MODEL_DIR, color=(0.7, 0.7, 0.7), **kwargs) -> None:
        super().__init__(**kwargs)

        self.body_model_dir = body_model_dir
        self.color = list(color)
        self.mesh_viewer = MeshViewer(
            width=self._image_size[0], height=self._image_size[1], use_offscreen=True)

    @lru_cache(maxsize=3)
    def __get_body_model(self, gender):
        model_path = os.path.join(self.body_model_dir, MODELS[gender])
        return BodyModel(bm_fname=model_path)

    def render(self, meta: List[Dict[str, Any]], **kwargs) -> List[np.ndarray]:
        rendered_videos = len(meta['video_id'])

        for clip_idx in range(rendered_videos):
            video = self.render_clip(
                body_model=self.__get_body_model(meta['gender'][clip_idx]),
                body_pose_clip=meta['amass_body_pose'][clip_idx]
            )
            yield video

    def render_clip(self, body_model: BodyModel, body_pose_clip: torch.Tensor) -> np.ndarray:
        video = []

        faces = body_model.f
        vertices = body_model(pose_body=body_pose_clip).v
        _, num_verts = vertices.shape[:-1]

        for vert in vertices:
            frame = self.render_frame(vert, faces, num_verts)
            video.append(frame)

        return np.stack(video)

    def render_frame(self, vertices, faces, num_verts) -> np.ndarray:
        mesh = trimesh.base.Trimesh(
            vertices, faces, vertex_colors=num_verts*self.color)
        self.mesh_viewer.set_meshes([mesh], 'static')

        return self.mesh_viewer.render()
