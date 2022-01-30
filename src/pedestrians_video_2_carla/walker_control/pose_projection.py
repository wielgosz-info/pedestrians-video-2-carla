import os
import warnings
from collections import OrderedDict
from typing import Tuple

import cameratransform as ct
import numpy as np
from pedestrians_video_2_carla.carla_utils.setup import get_camera_transform


try:
    import carla
except ImportError:
    import pedestrians_video_2_carla.carla_utils.mock_carla as carla
    warnings.warn("Using mock carla.", ImportWarning)


class RGBCameraMock(object):
    """
    Mocks up the default CARLA camera.
    """

    def __init__(self, pedestrian: 'ControlledPedestrian' = None, x: int = 800, y: int = 600, **kwargs):
        super().__init__()

        self.attributes = {
            'image_size_x': str(x),
            'image_size_y': str(y),
            'fov': '90.0',
            'lens_x_size': '0.08',
            'lens_y_size': '0.08'
        }
        if pedestrian is not None:
            self._transform = get_camera_transform(pedestrian, **kwargs)
        else:
            self._transform = carla.Transform()

    def get_transform(self):
        return self._transform


class PoseProjection(object):
    def __init__(self, pedestrian: 'ControlledPedestrian', camera_rgb: 'carla.Sensor' = None, *args, **kwargs) -> None:
        super().__init__()

        self._pedestrian = pedestrian

        if camera_rgb is None:
            camera_rgb = RGBCameraMock(pedestrian)

        self._image_size = (
            int(camera_rgb.attributes['image_size_x']),
            int(camera_rgb.attributes['image_size_y'])
        )
        self.camera = self._setup_camera(camera_rgb)

    @property
    def image_size(self) -> Tuple:
        """
        Returns projection image size.

        :return: (width, height)
        :rtype: Tuple
        """
        return self._image_size

    def _calculate_distance_and_elevation(self, camera_rgb: 'carla.Sensor') -> Tuple[float, float]:
        distance = camera_rgb.get_transform().location.x - \
            self._pedestrian.world_transform.location.x + \
            self._pedestrian.spawn_shift.x
        elevation = camera_rgb.get_transform().location.z - \
            self._pedestrian.world_transform.location.z + \
            self._pedestrian.spawn_shift.z

        return distance, elevation

    def _setup_camera(self, camera_rgb: 'carla.Sensor'):
        # basic transform is in UE world coords, axes of which are different
        # additionally, we need to correct spawn shift error
        distance, elevation = self._calculate_distance_and_elevation(camera_rgb)

        camera_ct = ct.Camera(
            ct.RectilinearProjection(
                image_width_px=self._image_size[0],
                image_height_px=self._image_size[1],
                view_x_deg=float(camera_rgb.attributes['fov']),
                sensor_width_mm=float(camera_rgb.attributes['lens_x_size'])*1000,
                sensor_height_mm=float(camera_rgb.attributes['lens_y_size'])*1000
            ),
            ct.SpatialOrientation(
                pos_y_m=distance,
                elevation_m=elevation,
                heading_deg=180,
                tilt_deg=90
            )
        )

        return camera_ct

    def update_camera(self, camera_position: Tuple[float, float, float]):
        """
        Updates camera position.

        :param camera_position: new camera position (x, y, z)
        :type camera_position: List[Tuple[float, float, float]]
        """

        (x, y, z) = camera_position

        self.camera.orientation = ct.SpatialOrientation(
            pos_x_m=y,
            pos_y_m=x,
            elevation_m=z
        )

    def current_pose_to_points(self):
        # switch from UE world coords, axes of which are different
        root_transform = carla.Transform(location=carla.Location(
            x=self._pedestrian.transform.location.y,
            y=self._pedestrian.transform.location.x,
            z=self._pedestrian.transform.location.z
        ), rotation=carla.Rotation(
            yaw=-self._pedestrian.transform.rotation.yaw
        ))

        relativeBones = [
            root_transform.transform(carla.Location(
                x=-bone.location.x,
                y=bone.location.y,
                z=bone.location.z
            ))
            for bone in self._pedestrian.current_pose.absolute.values()
        ]
        return self.camera.imageFromSpace([
            (bone.x, bone.y, bone.z)
            for bone in relativeBones
        ], hide_backpoints=False)


if __name__ == "__main__":
    from collections import OrderedDict
    from queue import Empty, Queue

    from pedestrians_video_2_carla.carla_utils.destroy import \
        destroy_client_and_world
    from pedestrians_video_2_carla.carla_utils.setup import *
    from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
        ControlledPedestrian

    client, world = setup_client_and_world()
    pedestrian = ControlledPedestrian(world, 'adult', 'female')

    sensor_dict = OrderedDict()
    camera_queue = Queue()

    sensor_dict['camera_rgb'] = setup_camera(
        world, camera_queue, pedestrian
    )

    projection = PoseProjection(
        pedestrian,
        sensor_dict['camera_rgb']
    )

    ticks = 0
    while ticks < 10:
        w_frame = world.tick()

        try:
            sensor_data = camera_queue.get(True, 1.0)
            sensor_data.save_to_disk(
                '/outputs/carla/{:06d}.png'.format(sensor_data.frame))
            ticks += 1
        except Empty:
            print("Some sensor information is missed in frame {:06d}".format(w_frame))

        # rotate & apply slight movement to pedestrian to see if projections are working correctly
        pedestrian.teleport_by(carla.Transform(
            location=carla.Location(0.1, 0, 0),
            rotation=carla.Rotation(yaw=15)
        ))
        pedestrian.update_pose({
            'crl_arm__L': carla.Rotation(yaw=-6),
            'crl_foreArm__L': carla.Rotation(pitch=-6)
        })

    destroy_client_and_world(client, world, sensor_dict)
