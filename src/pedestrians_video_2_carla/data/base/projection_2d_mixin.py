from typing import Callable, Literal, Optional

import torch


class Projection2DMixin:
    def __init__(self,
                 transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 missing_point_probability: Optional[float] = 0.0,
                 noise: Optional[Literal['zero', 'gaussian', 'uniform']] = 'zero',
                 noise_param: Optional[float] = 1.0,
                 **kwargs):
        """
        Mixing to handle common operations on 2D input data.
        Optional seed can be provided for torch.Generator handling
        noise and missing points, HOWEVER to ensure reproducibility
        important DataModule parameters like batch_size and num_workers
        need to also be consistent across calls.
        """
        super().__init__(**kwargs)

        self.transform = transform
        self.missing_point_probability = missing_point_probability
        self.noise = noise
        self.noise_param = noise_param

        if kwargs.get('overfit_batches', 0):
            self.generator = None
        else:
            self.generator = torch.Generator()

    @property
    def needs_missing_points(self) -> bool:
        return self.missing_point_probability > 0.0

    @property
    def needs_noise(self) -> bool:
        return self.noise is not None and self.noise != 'zero'

    @property
    def needs_deform(self) -> bool:
        return self.needs_missing_points or self.needs_noise

    @staticmethod
    def add_cli_args(parser):
        parser.add_argument(
            "--missing_point_probability",
            type=float,
            default=0.0,
            metavar='PROB',
            help="""
                Probability that a joint/node will be missing ("not detected") in a skeleton in a frame.
                Missing nodes are selected separately for each frame.
            """
        )
        parser.add_argument(
            "--noise",
            type=str,
            default='zero',
            choices=['zero', 'gaussian', 'uniform'],
            help="""
                Noise to add to the skeleton. Noise is added to each joint/node separately
                and before the missing_point_probability and any transformation
                is applied (before normalization, too!).
                'zero' (default) means no noise.
            """
        )
        parser.add_argument(
            "--noise_param",
            type=float,
            default=1.0,
            metavar='STD or SCALE',
            help="""
                Standard deviation of the Gaussian noise to add to the skeleton OR
                Scale of the uniform noise to add to the skeleton.
            """
        )
        return parser

    @staticmethod
    def extract_hparams(kwargs) -> dict:
        return {
            'missing_point_probability': kwargs.get('missing_point_probability', 0.0),
            'noise': kwargs.get('noise', 'zero'),
            'noise_param': kwargs.get('noise_param', 1.0),
        }

    def apply_deform(self, projection_2d: torch.Tensor) -> torch.Tensor:
        """
        Deforms the data by adding noise and missing points.
        Returns a clone of the original data.
        """
        assert projection_2d.shape[-1] == 2, 'projection_2d must be 2D'

        deformed_projection_2d = projection_2d.clone()

        if self.needs_noise:
            if self.noise == 'gaussian':
                noise = torch.normal(mean=0.0, std=self.noise_param,
                                     size=deformed_projection_2d.shape,
                                     generator=self.generator,
                                     dtype=deformed_projection_2d.dtype,
                                     device=deformed_projection_2d.device)
            elif self.noise == 'uniform':
                # TODO: this is not fully symmetric, since rand generates numbers from [0,1) interval
                noise = torch.rand_like(deformed_projection_2d, generator=self.generator) * \
                    self.noise_param - self.noise_param / 2.0
            else:
                raise ValueError('Unknown noise type: {}'.format(self.noise))

            deformed_projection_2d += noise

        if self.needs_missing_points:
            missing_indices = torch.rand(
                deformed_projection_2d.shape[:-1], generator=self.generator) < self.missing_point_probability
            deformed_projection_2d[missing_indices] = torch.zeros(
                deformed_projection_2d.shape[-1:], device=deformed_projection_2d.device)

        return deformed_projection_2d

    @property
    def needs_transform(self) -> bool:
        return self.transform is not None

    def apply_transform(self, projection_2d: torch.Tensor) -> torch.Tensor:
        """
        Transforms the data by applying a transformation.
        Returns a clone of the original data.
        """
        # clone data before user transform
        transformed_projection_2d = projection_2d.clone()

        if self.needs_transform:
            transformed_projection_2d = self.transform(transformed_projection_2d)

        return transformed_projection_2d

    def process_projection_2d(self, projection_2d: torch.Tensor) -> torch.Tensor:
        """
        Deforms the data by adding noise and missing points and then applies a transformation.
        Returns a clone of the original data suitable for training and a dict of the relevant targets.

        """
        deformed_projection_2d = self.apply_deform(projection_2d)
        transformed_deformed_projection_2d = self.apply_transform(
            deformed_projection_2d)
        transformed_projection_2d = self.apply_transform(projection_2d)

        targets = {
            'projection_2d': projection_2d
        }

        if self.needs_deform:
            targets['projection_2d_deformed'] = deformed_projection_2d

        if self.needs_transform:
            targets['projection_2d_transformed'] = transformed_projection_2d
            targets['projection_2d_shift'] = self.transform.shift
            targets['projection_2d_scale'] = self.transform.scale

        return transformed_deformed_projection_2d, targets
