# pedestrians-video-2-carla

This is a part of the bigger project to bring the more realistic pedestrian movements to CARLA.
It isn't intended for fully standalone use. Please see the [main project README.md](https://github.com/wielgosz-info/carla-pedestrians/blob/main/README.md) for details.

## Setup

Copy the `.env.template` file to `.env` and edit the values as required.

Run the CARLA server (optional) & the container with our code (`carla-pedestrians-client-1`).

If you don't have the Nvidia GPU, there is CPU-only version available `docker-compose.cpu.yml`.
Please note that currently running CARLA server requires GPU, so without it the `source_carla`
`carla` renderers shouldn't be used, since it would result in errors.

```sh
COMMIT=$(git rev-parse --short HEAD) docker-compose -f "docker-compose.yml" --env-file .env up -d --build
```

## Running

If needed, run the
```sh
python src/pedestrians_video_2_carla/data/utils/jaad_annotations_xml_2_csv.py
```
script to convert the JAAD annotations from XML to CSV. The output will be in `/outputs/JAAD/annotations.csv` and this is where `JAADOpenPoseDataModule` will look for it by default.

Inside the container, you can run the following command to start the training:

```sh
python -m pedestrians_video_2_carla \
  --data_module_name=Carla2D3D \
  --model_movements_name=LinearAE \
  --batch_size=256 \
  --num_workers=32 \
  --clip_length=180 \
  --input_nodes=CARLA_SKELETON \
  --output_nodes=CARLA_SKELETON \
  --max_epochs=500 \
  --loss_modes=loc_2d_3d \
  --renderers none \
  --check_val_every_n_epoch=10 \
  --gpus=0,1 \
  --accelerator=ddp \
  --limit_train_batches=32 \
  --log_every_n_steps=16 \
  --flush_logs_every_n_steps=64
```

Full list of options is available by running:

```sh
python -m pedestrians_video_2_carla --help
```

## Example run with rendering

```sh
python -m pedestrians_video_2_carla \
  --data_module_name=Carla2D3D \
  --batch_size=256 \
  --num_workers=32 \
  --input_nodes=CARLA_SKELETON \
  --output_nodes=CARLA_SKELETON \
  --loss_modes=loc_2d_3d \
  --check_val_every_n_epoch=10 \
  --gpus=0,1 \
  --accelerator=ddp \
  --limit_train_batches=1 \
  --limit_val_batches=1 \
  --log_every_n_steps=16 \
  --renderers source_carla input_points projection_points carla \
  \
  --movements_model_name=Seq2SeqEmbeddings \
  --max_epochs=200 \
  --clip_length=10 \
  --val_batches=32 \
  --test_batches=32 \
  --resume_from_checkpoint=/runs/Seq2SeqEmbeddings/version_8/checkpoints/epoch=199-step=6399.ckpt \

```

## Conda

The preferred way of running is via Docker. If conda is used, in addition to creating the env from the provided `environment.yml` file, following steps need to be done:

1. Create and activate conda environment with the following command:
   
    ```sh
    conda create -f environment.yml
    conda activate pedestrians
    ```
    
2. Install the `pedestrians_video_2_carla` package with:

    ```sh
    COMMIT=$(git rev-parse --short HEAD) SETUPTOOLS_SCM_PRETEND_VERSION="0.0.post0.dev38+${COMMIT}.dirty" pip install -e .
    ```

3. Run `pytest tests` to see if everything is working.

**Please note that conda env is not actively maintained.**

## Reference skeletons
Reference skeleton data in `src/pedestrians_video_2_carla/skeletons/reference` are extracted form [CARLA project Walkers *.uasset files](https://bitbucket.org/carla-simulator/carla-content).

## License
[MIT License](https://github.com/wielgosz-info/pedestrians-video-2-carla/blob/main/LICENSE)

This project uses videos and annotations from [JAAD dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/), created by Amir Rasouli, Iuliia Kotseruba, and John K. Tsotsos, to extract pedestrians movements and attributes. The videos and annotations are released under [MIT License](https://github.com/ykotseruba/JAAD/blob/JAAD_2.0/LICENSE).

This project uses [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), created by Ginés Hidalgo, Zhe Cao, Tomas Simon, Shih-En Wei, Yaadhav Raaj, Hanbyul Joo, and Yaser Sheikh, to extract pedestrians skeletons from videos. OpenPose has its [own licensing](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE) (basically, academic or non-profit organization noncommercial research use only).

## Funding

|                                                                                                                              |                                                                                                                      |                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <img src="docs/_static/images/logos/Logo Tecniospring INDUSTRY_white.JPG" alt="Tecniospring INDUSTRY" style="height: 24px;"> | <img src="docs/_static/images/logos/ACCIO_horizontal.PNG" alt="ACCIÓ Government of Catalonia" style="height: 35px;"> | <img src="docs/_static/images/logos/EU_emblem_and_funding_declaration_EN.PNG" alt="This project has received funding from the European Union's Horizon 2020 research and innovation programme under Marie Skłodowska-Curie grant agreement No. 801342 (Tecniospring INDUSTRY) and the Government of Catalonia's Agency for Business Competitiveness (ACCIÓ)." style="height: 70px;"> |

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
