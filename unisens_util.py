import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import unisens
from alive_progress import alive_bar

logger = logging.getLogger("unisens")

# Set the logging level to ignore specific types of messages (e.g., INFO, WARNING, ERROR)
logger.setLevel(logging.WARNING)

NUM_YAMNET_CLASSES = 521


def import_values_entry(value_name: str, file_path: str, np_value_datatype: np.dtype) -> tuple[np.array, np.array]:
    """imports unisens value entries

    Parameters
    ----------
    file_path : str
        The path to the directory where the unisens files are in.
    value_name : str
        The name of the value entry (name of the file, without the .bin), should be the same as described in the unisens
        xml file.
    np_value_datatype : np.dtype
        The numpy.dtpye used for the values.

    Returns
    -------
    tuple
        numpy_array
            The values in a numpy array.
        numpy_array
            The timeStamps in an numpy array.
    """

    # unisens logs a warning and I dont know why: "WARNING:root:id "None" does not end in .csv"

    u = unisens.Unisens(file_path)
    num_channels = len(u[value_name + ".bin"].channel)

    record_dtype = np.dtype([("timeStamps", np.int64), ("values", np_value_datatype, num_channels)])

    data = np.fromfile(file_path + "/" + value_name + ".bin", dtype=record_dtype)

    value_result = data["values"]
    long_data = data["timeStamps"]

    return value_result, long_data


def import_signal_entry(value_name: str, file_path: str, np_value_datatype: np.dtype) -> np.array:
    """imports unisens signal entries

    Parameters
    ----------
    file_path : str
        The path to the directory where the unisens files are in
    value_name : str
        The name of the value entry (name of the file, without the .bin), should be the same as described in the unisens
        xml file.
    np_value_datatype : np.dtype
        The numpy.dtpye used for the values.

    Returns
    -------
    numpy_array
        The values in a numpy array.
    """

    return np.fromfile(file_path + "/" + value_name + ".bin", dtype=np_value_datatype)


def extract_timestamp_from_values_entry(value_name: str, file_path: str, np_value_datatype: np.dtype) -> np.array:
    """imports only the timeStanps from the unisens value entries

    Parameters
    ----------
    file_path : str
        The path to the directory where the unisens files are in
    value_name : str
        The name of the value entry (name of the file, without the .bin), should be the same as described in the unisens
        xml file.
    np_value_datatype : np.dtype
        The numpy.dtpye used for the values.

    Returns
    -------
    numpy_array
        The timeStamps in an numpy array.
    """

    u = unisens.Unisens(file_path)
    num_channels = len(u[value_name + ".bin"].channel)

    record_dtype = np.dtype([("timeStamps", np.int64), ("values", np_value_datatype, num_channels)])

    data = np.fromfile(file_path + "/" + value_name + ".bin", dtype=record_dtype)

    return data["timeStamps"]


def csv_from_values_entry(
    value_name: str,
    file_path: str,
    np_value_datatype: np.dtype,
    float_format: str = None,
    zoneInfo: str = "Europe/Berlin",
    abs_time: bool = True,
) -> pd.DataFrame:
    """
    Extracts the data from the unisens file and transforms it to a human readable csv file.

    Parameters
    ----------
    file_path : str
        The path to the directory where the unisens files are in
    value_name : str
        The name of the value entry (name of the file, without the .bin), should be the same as described in the unisens
        xml file.
    np_value_datatype : np.dtype
        The numpy.dtpye used for the values.
    float_format : str
        To specify the float number format when saving to csv, example float_format="%.3f". This can either de- or increase the csv file size.
        While working with R there were some issues when loading in the csv files as the floats were interpreted as strings which led to
        problems with the scientific float representation. Giving the amount of decimal places this gets fixed rather crudely.

    Returns
    -------
    dataframe
        The pandas dataframe with the timestamps and corresponding unisens data.
    """

    u = unisens.Unisens(file_path)
    value_result, timestamp_result = import_values_entry(value_name, file_path, np_value_datatype)

    df = pd.DataFrame(
        data=value_result, columns=[i.name for i in u[value_name + ".bin"].channel], dtype=np_value_datatype
    )

    if abs_time:
        date_string = u.timestampStart
        date_format = "%Y-%m-%dT%H:%M:%S.%f"

        # Parse the date string into a datetime object
        dt_object = datetime.strptime(date_string, date_format)

        # Convert the datetime object to milliseconds
        milliseconds_start = int(dt_object.timestamp() * 1000)
        timestamp_result_seconds = (milliseconds_start + timestamp_result) / 1000.0

        date_array = [datetime.fromtimestamp(second, tz=ZoneInfo(zoneInfo)) for second in timestamp_result_seconds]

        df.insert(0, "timeStamp", date_array)

    else:
        df.insert(0, "timeStamp", timestamp_result)

    df.to_csv(file_path + "/" + value_name + ".csv", index=False, sep=";", float_format=float_format)

    return df


def find_sample_rate_resolution(data_path: Path) -> float:
    """
    Searches in the unisens.xml file for the sample rate resolution, this currently only works for the audio_classifier.

    Parameters
    ----------
    data_path: Path
        Path to the unisens folder.

    Returns
    -------
    sample_rate_resolution: float
        The used sample_rate_resolution currently it is set to centiseconds
    """

    unisens_as_string = unisens.Unisens(str(data_path)).to_xml()
    filtered_lines = [line for line in unisens_as_string.splitlines() if "audio_classifier_scores.csv" in line]

    matches = []
    for line in filtered_lines:
        matches.extend(re.findall(r'sampleRate="(\d+)"', line))

    sample_rates = list(map(int, matches))

    return float(sample_rates[0])


class AudioClassifierDataReader:
    def __init__(
        self, data_path: Path, float_format: str = "%.4f", reorganize: bool = True, sep: str = ";", decimal: str = ","
    ):
        self.float_format = float_format
        self.sep = sep
        self.decimal = decimal
        self.data_path = data_path
        self.result = {}
        self.result["abs_timestamp"] = []
        self.result["rel_timestamp_millis"] = []

        self.sample_rate_resolution = find_sample_rate_resolution(data_path)
        self.merged_df = self._read(reorganize)

    def add_start_time(self, rel_timestamp, start_datetime):
        return start_datetime + timedelta(seconds=rel_timestamp / self.sample_rate_resolution)

    def create_column_name(self, pre_name: str, length: int):
        return [f"{pre_name}_{i}" for i in range(1, length + 1)]

    def get_classes_scores(self, values: np.ndarray):
        top_indices = np.argpartition(values, -self.num)[-self.num :]
        sorted_indices = top_indices[np.argsort(values[top_indices])[::-1]]

        for i, cls in enumerate(sorted_indices):
            self.result[self.class_columns[i]].append(cls)
            self.result[self.score_columns[i]].append(values[cls])

    def _read(self, reorganize: bool = True):
        date_string = unisens.Unisens(str(self.data_path)).timestampStart
        date_format = "%Y-%m-%dT%H:%M:%S.%f"

        dt_object = datetime.strptime(date_string, date_format)

        class_df = pd.read_csv(self.data_path / "audio_classifier_class.csv", header=None)
        num_classes = class_df.shape[1] - 1
        self.class_columns = self.create_column_name("class", num_classes)
        class_df.columns = ["rel_timestamp"] + self.class_columns

        scores_df = pd.read_csv(self.data_path / "audio_classifier_scores.csv", header=None)
        num_scores = scores_df.shape[1] - 1
        self.score_columns = self.create_column_name("score", num_scores)
        scores_df.columns = ["rel_timestamp"] + self.score_columns

        self.num = min(num_scores, num_classes)

        merged_df = pd.merge(class_df, scores_df, on="rel_timestamp")
        merged_df["abs_timestamp"] = merged_df["rel_timestamp"].apply(self.add_start_time, args=(dt_object,))

        merged_df["rel_timestamp_millis"] = (merged_df["rel_timestamp"] * 1000 / self.sample_rate_resolution).astype(
            int
        )

        if reorganize:
            new_column_order = ["abs_timestamp", "rel_timestamp_millis"]

            for i in range(1, self.num + 1):
                new_column_order += [f"class_{i}", f"score_{i}"]

            merged_df = merged_df[new_column_order]

        merged_df.to_csv(
            self.data_path / Path("result.csv"),
            index=False,
            float_format=self.float_format,
            sep=self.sep,
            decimal=self.decimal,
        )

        for i in range(self.num):
            self.result[self.class_columns[i]] = []
            self.result[self.score_columns[i]] = []

        return merged_df

    def compute_mean(self, accepted_time_delta: timedelta) -> pd.DataFrame:
        counter = 0
        class_scores = np.zeros(NUM_YAMNET_CLASSES)

        abs_timestamp_list = []
        rel_timestamp_list = []

        with alive_bar(self.merged_df.shape[0], force_tty=True) as bar:
            for _, row in self.merged_df.iterrows():
                counter += 1

                abs_timestamp_list.append(row["abs_timestamp"])
                rel_timestamp_list.append(row["rel_timestamp_millis"])

                for i, ind in enumerate(row[self.class_columns].values):
                    class_scores[ind] += row[self.score_columns].values[i]

                abs_delta = abs_timestamp_list[-1] - abs_timestamp_list[0]
                rel_delta = rel_timestamp_list[-1] - rel_timestamp_list[0]

                if abs_delta >= accepted_time_delta:
                    class_scores /= counter
                    self.get_classes_scores(class_scores)

                    self.result["abs_timestamp"].append(abs_timestamp_list[0] + abs_delta / 2)
                    self.result["rel_timestamp_millis"].append(rel_timestamp_list[0] + rel_delta / 2)

                    # Reset values
                    abs_timestamp_list.clear()
                    rel_timestamp_list.clear()

                    class_scores.fill(0)
                    counter = 0

                bar()

        if counter != 0:
            class_scores /= counter
            self.get_classes_scores(class_scores)
            self.result["abs_timestamp"].append(abs_timestamp_list[0] + abs_delta / 2)
            self.result["rel_timestamp_millis"].append(rel_timestamp_list[0] + rel_delta / 2)

        mean_df = pd.DataFrame(self.result)
        mean_df.to_csv(
            self.data_path / Path("mean_result.csv"),
            index=False,
            float_format=self.float_format,
            sep=self.sep,
            decimal=self.decimal,
        )

        return mean_df
