import glob
import logging
import os
from typing import List

import numpy as np
import pims
from pandas.core.frame import DataFrame
import torch

from tqdm.auto import tqdm
from PIL import Image
from torchvision.transforms.functional import equalize, normalize, resize

from pedestrians_video_2_carla.data.openpose.skeleton import COCO_SKELETON
from pedestrians_video_2_carla.data.openpose.jaad_openpose_datamodule import JAADOpenPoseDataModule
from pedestrians_video_2_carla.data.openpose.constants import JAAD_DIR
from pedestrians_video_2_carla.data.unipose.constants import UNIPOSE_MODEL_DIR

from pedestrians_scenarios.karma.renderers.points_renderer import PointsRenderer


class JAADUniPoseDataModule(JAADOpenPoseDataModule):
    """
    An attempt to see how changing the pose extractor affects the results.
    This in theory works, but the results are so poor that it makes no sense to use it (or optimize its inner workings).
    Replacing UniPoseCOCO with UniPoseLSTM was even worse.
    Code is kept here for future reference (e.g. when better pose extractor will be found).

    Potential optimizations should include:
    - going through all the videos only once & caching the results (as OpenPose-compatible keypoints.json files)
    - then leveraging the rest of OpenPoseDataModule's functionality to extract the keypoints for specific clips
    - histogram equalization of the images (to improve the results)
    - better normalization of the images (to improve the results)
    """

    def __init__(self,
                 **kwargs
                 ):
        super().__init__(
            **{
                **kwargs,
                'data_nodes': kwargs.get('data_nodes', COCO_SKELETON) or COCO_SKELETON,
                'fast_dev_run': True
            }
        )

        # override openpose dir with unipose dumped data
        self._unipose_model_path = os.path.join(UNIPOSE_MODEL_DIR, 'UniPose_COCO.pth')

    def _extract_additional_data(self, clips: List[DataFrame]):
        """
        Extract skeleton data from keypoint files. This potentially modifies data in place!

        :param clips: List of DataFrames
        :type clips: List[DataFrame]
        """

        # only load it when we really need it
        unipose_model = self._load_unipose_model(self._unipose_model_path)

        # TODO: make this configurable, but for now we only use CPU since it doesn't fit into memory
        device = torch.device('cpu')
        unipose_model = unipose_model.to(device)

        updated_clips = []
        with torch.no_grad():
            for clip in tqdm(clips, desc='Extracting skeleton data', leave=False):
                info = self._extract_unipose_keypoints(unipose_model, device, clip)
                if info is not None:
                    updated_clips.append(info)

        return updated_clips

    def _extract_unipose_keypoints(self, unipose_model, device, clip):
        pedestrian_info = clip.reset_index().sort_values('frame')

        video_id = pedestrian_info.iloc[0]['video']
        pedestrian_id = pedestrian_info.iloc[0]['id']
        clip_id = pedestrian_info.iloc[0]['clip']
        start_frame = int(pedestrian_info.iloc[0]['frame'])
        end_frame = int(pedestrian_info.iloc[-1]['frame'] + 1)

        paths = glob.glob(os.path.join(
            self.datasets_dir, JAAD_DIR, 'videos', '{}.*'.format(os.path.splitext(video_id)[0])))

        if len(paths) != 1:
            # no video or multiple candidates - skip
            logging.getLogger(__name__).warn(
                "Clip extraction failed for {}, {}, {}".format(
                    video_id,
                    pedestrian_id,
                    clip_id))
            return None

        bboxes = pedestrian_info[['x1', 'y1', 'x2', 'y2']
                                 ].to_numpy().reshape((-1, 2, 2))

        target_size = 368
        canvas_size = (
            bboxes[:, 1] - bboxes[:, 0]).max().astype(int)
        canvas_size = max(canvas_size, target_size)
        half_size = canvas_size // 2

        canvas = np.zeros((end_frame - start_frame, canvas_size,
                           canvas_size, 3), dtype=np.uint8)

        with pims.PyAVReaderIndexed(paths[0]) as video:
            clip = video[start_frame:end_frame]
            clip_length = len(clip)
            (clip_height, clip_width, _) = clip.frame_shape

            centers = (bboxes.mean(axis=-2) + 0.5).round().astype(int)
            shifts = np.zeros((clip_length, 2), dtype=int)

            for idx in range(clip_length):
                shifts[idx] = self._extract_bbox_from_frame(canvas[idx], clip[idx],
                                                            (half_size, half_size),
                                                            centers[idx],
                                                            (clip_width, clip_height),
                                                            )

        # right now we have a canvas with all the frames of the clip
        # that hopefully contains the pedestrian in the center
        # it is now time to run the unipose model

        # convert to tensor
        tensor_canvas = torch.from_numpy(
            canvas.transpose((0, 3, 1, 2)).copy()
        )

        # equalize histogram of the images
        tensor_canvas = equalize(tensor_canvas)

        # resize if needed
        if canvas_size > target_size:
            scaled_canvas = torch.zeros(
                (clip_length, 3, target_size, target_size), dtype=torch.uint8)
            for idx in range(clip_length):
                scaled_canvas[idx] = resize(
                    tensor_canvas[idx],
                    (target_size, target_size),
                    antialias=True
                )
            tensor_canvas = scaled_canvas

        # normalize
        tensor_canvas = tensor_canvas.div(255.0)
        tensor_canvas = normalize(
            tensor_canvas, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        heatmaps = unipose_model(tensor_canvas.to(device))

        clip_keypoints = self._keypoints_from_heatmaps(
            heatmaps.cpu().numpy(),
            (canvas_size, canvas_size),
            shifts
        )

        for idx, keypoints in enumerate(clip_keypoints):
            pedestrian_info.at[pedestrian_info.index[idx],
                               'keypoints'] = keypoints

            # debug
            # pth = os.path.join(
            #     '/outputs/', '{}_{}_{}'.format(video_id, pedestrian_id, clip_id))
            # if not os.path.exists(pth):
            #     os.makedirs(pth)
            # kp = np.array(keypoints)[..., :2]
            # bbox_img = PointsRenderer.draw_projection_points(
            #     (tensor_canvas[idx]*128 + 127).numpy().transpose((1, 2, 0)
            #                                                      ).round().clip(0,255).astype(np.uint8),
            #     kp - shifts[idx], COCO_SKELETON, lines=True)
            # Image.fromarray(canvas[idx]).save(
            #     os.path.join(pth, 'bbox_{}.png'.format(idx)))
            # Image.fromarray(bbox_img).save(
            #     os.path.join(pth, 'bbox_{}_sk.png'.format(idx)))

            # frame_img = PointsRenderer.draw_projection_points(
            #     clip[idx], kp, COCO_SKELETON, lines=True)
            # Image.fromarray(frame_img).save(
            #     os.path.join(pth, 'frame_{}_sk.png'.format(idx)))

        # everything went well, append the clip to the list
        return pedestrian_info

    def _extract_bbox_from_frame(self, canvas, clip, half_size, bbox_center, clip_size):
        (half_width, half_height) = half_size
        (x_center, y_center) = bbox_center
        (clip_width, clip_height) = clip_size

        frame_x_min = int(max(0, x_center-half_width))
        frame_x_max = int(min(clip_width, x_center+half_width))
        frame_y_min = int(max(0, y_center-half_height))
        frame_y_max = int(min(clip_height, y_center+half_height))
        frame_width = frame_x_max - frame_x_min
        frame_height = frame_y_max - frame_y_min
        canvas_x_shift = max(0, half_width-x_center)
        canvas_y_shift = max(0, half_height-y_center)
        canvas[canvas_y_shift:canvas_y_shift+frame_height, canvas_x_shift:canvas_x_shift +
               frame_width] = clip[frame_y_min:frame_y_max, frame_x_min:frame_x_max]

        return frame_x_min - canvas_x_shift, frame_y_min - canvas_y_shift

    def _load_unipose_model(self, unipose_model_path):
        """
        Load the Unipose model.
        """
        from pedestrians_video_2_carla.data.unipose.unipose_coco import UniPoseCOCO

        return UniPoseCOCO(unipose_model_path)

    def _keypoints_from_heatmaps(self, heatmaps, bbox_size, bbox_shifts):
        """
        Convert heatmaps to keypoints.
        """
        keypoints = np.zeros((len(heatmaps), len(self.data_nodes), 3), dtype=np.float32)
        (bbox_width, bbox_height) = bbox_size

        for i, heatmap in enumerate(heatmaps):
            (x_shift, y_shift) = bbox_shifts[i]
            kpts = []
            for m in heatmap[1:]:
                h, w = np.unravel_index(m.argmax(), m.shape)
                x = int(w * bbox_width / m.shape[1]) + x_shift
                y = int(h * bbox_height / m.shape[0]) + y_shift
                c = m.max()
                if c > 0:
                    kpts.append((x, y, c))
                else:
                    kpts.append((0, 0, 0))

            # points are detected in the order other than standard COCO order
            # this is a bit of a guesswork, because I couldn't find a reference
            # no neck or nose
            unipose_order = tuple([n.value for n in (
                COCO_SKELETON.LEye,  # 0
                COCO_SKELETON.REye,  # 1
                COCO_SKELETON.LEar,  # 2
                COCO_SKELETON.REar,  # 3
                COCO_SKELETON.LShoulder,  # 4
                COCO_SKELETON.RShoulder,  # 5
                COCO_SKELETON.LElbow,  # 6
                COCO_SKELETON.RElbow,  # 7
                COCO_SKELETON.LWrist,  # 8
                COCO_SKELETON.RWrist,  # 9
                COCO_SKELETON.LHip,  # 10
                COCO_SKELETON.RHip,  # 11
                COCO_SKELETON.LKnee,  # 12
                COCO_SKELETON.RKnee,  # 13
                COCO_SKELETON.LAnkle,  # 14
                COCO_SKELETON.RAnkle,  # 15
            )])
            keypoints[i, unipose_order] = kpts

        return keypoints.tolist()


if __name__ == "__main__":
    dm = JAADUniPoseDataModule(input_nodes=COCO_SKELETON,
                               fast_dev_run=True, clip_length=180, clip_offset=1800)
    dm.prepare_data()