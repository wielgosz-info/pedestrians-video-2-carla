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


class SMPLRenderer(Renderer):
    """
    This renderer basically does the same job as `body_visualizer.tools.vis_tools.render_smpl_params`.

    TODO: Rewrite to use pytorch3d instead of pyrender?.
    """

    def __init__(self, body_model_dir: str = BODY_MODEL_DIR, color=(0.7, 0.7, 0.7), **kwargs) -> None:
        super().__init__(**kwargs)

        self.body_model_dir = body_model_dir
        self.color = list(color)

    @lru_cache(maxsize=3)
    def __get_body_model(self, gender):
        model_path = os.path.join(self.body_model_dir, gender, 'model.npz')
        return BodyModel(bm_fname=model_path)

    def render(self, meta: List[Dict[str, Any]], image_size: Tuple[int, int] = (800, 600), **kwargs) -> List[np.ndarray]:
        rendered_videos = len(meta['video_id'])
        mesh_viewer = MeshViewer(
            width=image_size[0], height=image_size[1], use_offscreen=True)

        for clip_idx in range(rendered_videos):
            video = self.render_clip(
                mesh_viewer,
                body_model=self.__get_body_model(meta['gender'][clip_idx]),
                body_pose_clip=meta['amass_body_pose'][clip_idx],
                global_orientation_clip=meta['amass_global_orientation'][
                    clip_idx] if 'amass_global_orientation' in meta else None,
                image_size=image_size
            )
            yield video

    def render_clip(self, mesh_viewer: MeshViewer, body_model: BodyModel, body_pose_clip: torch.Tensor, global_orientation_clip: torch.Tensor, image_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        video = []

        faces = body_model.f
        vertices = body_model(pose_body=body_pose_clip,
                              root_orient=global_orientation_clip).v
        _, num_verts = vertices.shape[:-1]

        for vert in vertices:
            frame = self.render_frame(mesh_viewer, vert, faces, num_verts)
            video.append(frame)

        return np.stack(video)

    def render_frame(self, mv: MeshViewer, vertices, faces, num_verts) -> np.ndarray:
        mesh = trimesh.base.Trimesh(
            vertices, faces, vertex_colors=num_verts*self.color)
        mv.set_meshes([mesh], 'static')

        return mv.render()
