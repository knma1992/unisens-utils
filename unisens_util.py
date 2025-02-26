import logging
import os
import re
import shutil
import time
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


def create_column_name(pre_name: str, length: int):
    return [f"{pre_name}_{i}" for i in range(1, length + 1)]


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
        self.class_columns = create_column_name("class", num_classes)
        class_df.columns = ["rel_timestamp"] + self.class_columns

        scores_df = pd.read_csv(self.data_path / "audio_classifier_scores.csv", header=None)
        num_scores = scores_df.shape[1] - 1
        self.score_columns = create_column_name("score", num_scores)
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

        else:
            new_column_order = ["abs_timestamp", "rel_timestamp_millis"]

            for i in range(1, self.num + 1):
                new_column_order += [f"class_{i}"]
            
            for i in range(1, self.num + 1):
                new_column_order += [f"score_{i}"]

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
        np_timedelta = np.timedelta64(accepted_time_delta)

        abs_timestamps = self.merged_df["abs_timestamp"].to_numpy()
        rel_timestamps = self.merged_df["rel_timestamp_millis"].to_numpy()
        class_values = self.merged_df[self.class_columns].to_numpy()
        score_values = self.merged_df[self.score_columns].to_numpy()

        counter = 0
        class_scores = np.zeros(NUM_YAMNET_CLASSES)

        abs_start_time = abs_timestamps[0]
        rel_start_time = rel_timestamps[0]

        num_rows = self.merged_df.shape[0]

        with alive_bar(num_rows, force_tty=True) as bar:
            for index in range(num_rows - 1):
                counter += 1

                for i, ind in enumerate(class_values[index]):
                    # print(i, ind)
                    class_scores[ind] += score_values[index][i]

                abs_next_start_time = abs_timestamps[index + 1]
                rel_next_start_time = rel_timestamps[index + 1]

                abs_delta = abs_next_start_time - abs_start_time
                rel_delta = rel_next_start_time - rel_start_time

                if abs_delta >= np_timedelta:
                    class_scores /= counter
                    self.get_classes_scores(class_scores)

                    self.result["abs_timestamp"].append(abs_start_time)  #  + abs_delta / 2
                    self.result["rel_timestamp_millis"].append(int(rel_start_time))  #  + rel_delta / 2

                    # Reset values
                    abs_start_time = abs_next_start_time
                    rel_start_time = rel_next_start_time

                    class_scores.fill(0)
                    counter = 0

                bar()

        if counter != 0:
            class_scores /= counter
            self.get_classes_scores(class_scores)
            self.result["abs_timestamp"].append(abs_start_time)  #  + abs_delta / 2
            self.result["rel_timestamp_millis"].append(int(rel_start_time))  #  + rel_delta / 2

        mean_df = pd.DataFrame(self.result)
        mean_df["rel_timestamp_millis"] = mean_df["rel_timestamp_millis"].astype(int)

        mean_df.to_csv(
            self.data_path / Path("mean_result.csv"),
            index=False,
            float_format=self.float_format,
            sep=self.sep,
            decimal=self.decimal,
        )

        # mean df is great
        return mean_df


def create_huge_dataset_for_testing(data_path: Path, N: int = 2):
    """
    TLDR: Do not use this function! 
    
    It will copy an existing audio classifier study and duplicate it N times. 
    Then it will save it in to a new folder and copy the unisens.xml in to it. After that it will run the 
    combine/averaging on it to see how it behaves for large data.  

    Parameters
    ----------
    data_path : Path
        The path to the directory where the unisens files are in.
    N : int
        The amount of times the original data is duplicated.
    """

    class_df = pd.read_csv(data_path / "audio_classifier_class.csv", header=None)
    num_classes = class_df.shape[1] - 1
    class_columns = create_column_name("class", num_classes)
    class_df.columns = ["rel_timestamp"] + class_columns

    scores_df = pd.read_csv(data_path / "audio_classifier_scores.csv", header=None)
    num_scores = scores_df.shape[1] - 1
    score_columns = create_column_name("score", num_scores)
    scores_df.columns = ["rel_timestamp"] + score_columns

    float_format = "%.4f"

    sum_class_df = class_df.copy()
    sum_score_df = scores_df.copy()

    with alive_bar(N, force_tty=True) as bar:
        for i in range(N):
            sum_class_df = pd.concat([sum_class_df, class_df.copy()])
            sum_score_df = pd.concat([sum_score_df, scores_df.copy()])

            bar()

    new_path = data_path.parent / Path("long_test_study")

    try:
        os.makedirs(new_path, exist_ok=True)
    except ():
        print("folder already exists")

    print(f"created new folder: {new_path}")
    shutil.copyfile(data_path / Path("unisens.xml"), new_path / Path("unisens.xml"))
    print("copied xml from source")

    num_samples = sum_score_df.shape[0]
    sum_class_df["rel_timestamp"] = np.arange(num_samples) * 200
    sum_score_df["rel_timestamp"] = np.arange(num_samples) * 200

    study_time_hours = sum_class_df.iloc[-1]["rel_timestamp"].item() / (100 * 3600)

    print(f"fake study_time_hours: {timedelta(hours=study_time_hours)}")

    sum_class_df.to_csv(
        new_path / Path("audio_classifier_class.csv"), header=False, index=False, float_format=float_format
    )
    sum_score_df.to_csv(
        new_path / Path("audio_classifier_scores.csv"), header=False, index=False, float_format=float_format
    )

    t1 = time.perf_counter()
    reader = AudioClassifierDataReader(new_path)
    _ = reader.compute_mean(timedelta(minutes=1))
    elapsed_time = round(time.perf_counter() - t1, 3)

    print(
        f"For a fake study with {num_samples} samples it took {elapsed_time} seconds to complete averaging the results."
    )
