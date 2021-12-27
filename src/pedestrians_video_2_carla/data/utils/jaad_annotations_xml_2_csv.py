import os
import xmltodict
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, OrderedDict


class AnnotationsXml2Csv(object):
    # This is to be used for JAAD xml annotations to csv mapping
    def __init__(self, path_annotations_folder: str):
        super(AnnotationsXml2Csv).__init__()
        self.path_annotations_folder = path_annotations_folder

    @staticmethod
    def _check_and_add_to_dict(dictionary: Dict, key, value) -> Dict:
        if not key in dictionary:
            dictionary[key] = [value]
        else:
            dictionary[key].append(value)

        return dictionary

    def _process_box(self, box, video):
        self._check_and_add_to_dict(
            dictionary=self.dict_aggregated_annotations, key="video", value=video
        )
        self._check_and_add_to_dict(
            dictionary=self.dict_aggregated_annotations,
            key="frame",
            value=box["@frame"],
        )
        self._check_and_add_to_dict(
            dictionary=self.dict_aggregated_annotations,
            key="occluded",
            value=box["@occluded"],
        )
        self._check_and_add_to_dict(
            dictionary=self.dict_aggregated_annotations,
            key="outside",
            value=box["@outside"],
        )
        self._check_and_add_to_dict(
            dictionary=self.dict_aggregated_annotations,
            key="x2",
            value=float(box["@xbr"]),
        )
        self._check_and_add_to_dict(
            dictionary=self.dict_aggregated_annotations,
            key="y2",
            value=float(box["@ybr"]),
        )
        self._check_and_add_to_dict(
            dictionary=self.dict_aggregated_annotations,
            key="x1",
            value=float(box["@xtl"]),
        )
        self._check_and_add_to_dict(
            dictionary=self.dict_aggregated_annotations,
            key="y1",
            value=float(box["@ytl"]),
        )
        for attribute in box["attribute"]:
            self._check_and_add_to_dict(
                dictionary=self.dict_aggregated_annotations,
                key=attribute["@name"],
                value=attribute["#text"],
            )

            # add attributes of the pedestrian
            if attribute["@name"] == "id":
                if attribute["#text"] in self.dict_aggregated_attributes.keys():
                    for key, value in self.dict_aggregated_attributes[
                        attribute["#text"]
                    ].items():
                        self._check_and_add_to_dict(
                            dictionary=self.dict_aggregated_annotations,
                            key=key,
                            value=value,
                        )

    def get_annotations(self):
        # main annotation procedure

        folder = "annotations"
        print(f"Processing: {folder} ")

        self.dict_aggregated_annotations = OrderedDict()

        # iterate over files in the folder
        for file in tqdm(
            sorted(os.listdir(os.path.join(self.path_annotations_folder, folder)))
        ):
            # get file path
            path_file = os.path.join(
                os.path.join(os.path.abspath(self.path_annotations_folder), folder),
                file,
            )

            with open(path_file) as f:
                root = xmltodict.parse(f.read())

            # if there are no pedestians in the video -> skip the video
            # (there are some videos where there are no pedestians)
            if not "track" in root["annotations"].keys():
                continue

            # iterate over tracks in the file
            if isinstance(
                root["annotations"]["track"], List
            ):  # there are multiple predestians (tracks)
                for track in root["annotations"]["track"]:
                    if (
                        track["@label"] == "pedestrian"
                    ):  # only for pedestian not bypassers (denoted as peds)
                        # iterate over boxes within each track
                        for box in track["box"]:
                            self._process_box(
                                box=box,
                                video=root["annotations"]["meta"]["task"]["name"],
                            )

            else:  # there is only one track
                if (
                    root["annotations"]["track"]["@label"] == "pedestrian"
                ):  # only for pedestian not bypassers (denoted as peds)
                    for box in root["annotations"]["track"]["box"]:
                        self._process_box(
                            box=box, video=root["annotations"]["meta"]["task"]["name"]
                        )

        return self.dict_aggregated_annotations

    def get_annotations_appearance(self):
        folder = "annotations_appearance"
        print(f"Processing: {folder} ")

        self.dict_aggregated_appearance = OrderedDict()

        # iterate over files in the folder
        for file in tqdm(
            sorted(os.listdir(os.path.join(self.path_annotations_folder, folder)))
        ):
            # get file path
            path_file = os.path.join(
                os.path.join(os.path.abspath(self.path_annotations_folder), folder),
                file,
            )

            with open(path_file) as f:
                root = xmltodict.parse(f.read())

            # there are no pedestians
            if root["pedestrian_appearance"] is None:
                continue

            # iterate over tracks in the file
            if isinstance(
                root["pedestrian_appearance"]["track"], List
            ):  # there are multiple predestians (tracks)
                for track in root["pedestrian_appearance"]["track"]:
                    if (
                        track["@label"] == "pedestrian"
                    ):  # only for pedestian not bypassers (denoted as peds)

                        # iterate over boxes within each track
                        for box in track["box"]:
                            self._check_and_add_to_dict(
                                dictionary=self.dict_aggregated_appearance,
                                key="id",
                                value=track["@id"],
                            )
                            for key, value in box.items():
                                self._check_and_add_to_dict(
                                    dictionary=self.dict_aggregated_appearance,
                                    key=key[1:],
                                    value=value,
                                )

            else:  # there is only one track
                if (
                    root["pedestrian_appearance"]["track"]["@label"] == "pedestrian"
                ):  # only for pedestian not bypassers (denoted as peds)
                    if (
                        track["@label"] == "pedestrian"
                    ):  # only for pedestian not bypassers (denoted as peds)

                        for box in root["pedestrian_appearance"]["track"]["box"]:
                            self._check_and_add_to_dict(
                                dictionary=self.dict_aggregated_appearance,
                                key="id",
                                value=root["pedestrian_appearance"]["track"]["@id"],
                            )
                            for key, value in box.items():
                                self._check_and_add_to_dict(
                                    dictionary=self.dict_aggregated_appearance,
                                    key=key[1:],
                                    value=value,
                                )

        return self.dict_aggregated_appearance

    def get_attributes(self):
        folder = "annotations_attributes"
        print(f"Processing: {folder} ")

        self.dict_aggregated_attributes = OrderedDict()

        # iterate over files in the folder
        for file in tqdm(
            sorted(os.listdir(os.path.join(self.path_annotations_folder, folder)))
        ):
            # get file path
            path_file = os.path.join(
                os.path.join(os.path.abspath(self.path_annotations_folder), folder),
                file,
            )

            with open(path_file) as f:
                root = xmltodict.parse(f.read())

            # there are no pedestians
            if root["ped_attributes"] is None:
                continue

            if isinstance(
                root["ped_attributes"]["pedestrian"], List
            ):  # there are multiple predestians (tracks)
                for pedestian in root["ped_attributes"]["pedestrian"]:
                    self.dict_aggregated_attributes[pedestian["@id"]] = {
                        "age": pedestian["@age"],
                        "crossing": pedestian["@crossing"],
                        "decision_point": int(pedestian["@decision_point"]),
                        "gender": pedestian["@gender"],
                        "group_size": int(pedestian["@group_size"]),
                        "intersection": pedestian["@intersection"],
                        "num_lanes": int(pedestian["@num_lanes"]),
                    }
            else:  # single pedestian
                self.dict_aggregated_attributes[
                    root["ped_attributes"]["pedestrian"]["@id"]
                ] = {
                    "age": root["ped_attributes"]["pedestrian"]["@age"],
                    "crossing": root["ped_attributes"]["pedestrian"]["@crossing"],
                    "decision_point": int(
                        root["ped_attributes"]["pedestrian"]["@decision_point"]
                    ),
                    "gender": root["ped_attributes"]["pedestrian"]["@gender"],
                    "group_size": int(
                        root["ped_attributes"]["pedestrian"]["@group_size"]
                    ),
                    "intersection": root["ped_attributes"]["pedestrian"][
                        "@intersection"
                    ],
                    "num_lanes": int(
                        root["ped_attributes"]["pedestrian"]["@num_lanes"]
                    ),
                }

        return self.dict_aggregated_attributes

    def get_annotations_vehicle(self):
        folder = "annotations_vehicle"
        print(f"Processing: {folder} ")

        self.dict_aggregated_annotations_vehicle = OrderedDict()

        # iterate over files in the folder
        for file in tqdm(
            sorted(os.listdir(os.path.join(self.path_annotations_folder, folder)))
        ):
            # get file path
            path_file = os.path.join(
                os.path.join(os.path.abspath(self.path_annotations_folder), folder),
                file,
            )
            # print(f'doing file :{path_file}')

            with open(path_file) as f:
                root = xmltodict.parse(f.read())

            # there are no pedestians
            if root["vehicle_info"] is None:
                continue

            for frame in root["vehicle_info"]["frame"]:
                self._check_and_add_to_dict(
                    dictionary=self.dict_aggregated_annotations_vehicle,
                    key="video",
                    value=file[:10],
                )
                self._check_and_add_to_dict(
                    dictionary=self.dict_aggregated_annotations_vehicle,
                    key="frame",
                    value=frame["@id"],
                )
                self._check_and_add_to_dict(
                    dictionary=self.dict_aggregated_annotations_vehicle,
                    key="speed",
                    value=frame["@action"],
                )
                path_to_videos = os.path.join(self.path_annotations_folder, "videos")
                video_file_name = file[:10] + ".mp4"
                path_to_video_file = os.path.join(path_to_videos, video_file_name)
                self._check_and_add_to_dict(
                    dictionary=self.dict_aggregated_annotations_vehicle,
                    key="video_path",
                    value=path_to_video_file,
                )

        return self.dict_aggregated_annotations_vehicle

    def get_complete_annotations(self):
        self.get_attributes()  # this is needed for the line below (a == ...)
        a = self.get_annotations()
        b = self.get_annotations_appearance()
        c = self.get_annotations_vehicle()
        df_a = pd.DataFrame.from_dict(a)
        df_b = pd.DataFrame.from_dict(b)
        df_c = pd.DataFrame.from_dict(c)

        self.df_full = pd.merge(df_a, df_b, how="left", on=["id", "frame"])
        self.df_full = pd.merge(self.df_full, df_c, how="left", on=["video", "frame"])

        return self.df_full

    def generate_df(self, file_path="/outputs/JAAD/annotations.csv"):
        dirname = os.path.dirname(file_path)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self.get_complete_annotations()
        self.df_full.to_csv(file_path)

        print("Annotations generated.")


if __name__ == "__main__":
    annotations_xml_2_csv = AnnotationsXml2Csv(path_annotations_folder="/datasets/JAAD")
    annotations_xml_2_csv.generate_df()
