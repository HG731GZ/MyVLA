import json
import os
from pathlib import Path

import h5py
import numpy as np

from libero_dataset_utils import scan_libero_dataset, validate_official_counts
from prepare_libero_data import prepare_libero_data


def _write_demo_file(path: Path, num_demos: int = 2, timesteps: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        data = h5_file.create_group("data")
        for demo_index in range(num_demos):
            demo = data.create_group(f"demo_{demo_index}")
            demo.create_dataset(
                "actions",
                data=np.full((timesteps, 7), demo_index + 1, dtype=np.float32),
            )
            obs = demo.create_group("obs")
            obs.create_dataset(
                "agentview_rgb",
                data=np.zeros((timesteps, 2, 2, 3), dtype=np.uint8),
            )
            obs.create_dataset(
                "eye_in_hand_rgb",
                data=np.zeros((timesteps, 2, 2, 3), dtype=np.uint8),
            )
            obs.create_dataset(
                "ee_pos",
                data=np.full((timesteps, 3), demo_index, dtype=np.float32),
            )
            obs.create_dataset(
                "ee_ori",
                data=np.zeros((timesteps, 3), dtype=np.float32),
            )
            obs.create_dataset(
                "gripper_states",
                data=np.zeros((timesteps, 2), dtype=np.float32),
            )


def test_strict_validation_can_be_relaxed_for_leave_out_data(tmp_path):
    data_dir = tmp_path / "datasets"
    _write_demo_file(data_dir / "libero_object" / "task_one_demo.hdf5")
    snapshot = scan_libero_dataset(str(data_dir), ["libero_object"])

    try:
        validate_official_counts(snapshot, allow_incomplete=False)
    except RuntimeError as error:
        assert "LIBERO_STRICT_VALIDATION=0" in str(error)
    else:
        raise AssertionError("Strict validation should reject an incomplete subset")

    validate_official_counts(snapshot, allow_incomplete=True)


def test_prepare_rebuilds_stale_metadata_and_norm_stats(tmp_path):
    data_dir = tmp_path / "datasets"
    h5_path = data_dir / "libero_object" / "task_one_demo.hdf5"
    metadata_path = tmp_path / "libero_object_train.json"
    norm_path = tmp_path / "libero_object_norm.json"
    _write_demo_file(h5_path)

    first = prepare_libero_data(
        data_dir=str(data_dir),
        subsets=["libero_object"],
        metadata_path=str(metadata_path),
        norm_stats_path=str(norm_path),
        allow_incomplete=True,
    )
    assert first["metadata_rebuilt"]
    assert first["norm_stats_rebuilt"]

    with metadata_path.open() as input_file:
        first_metadata = json.load(input_file)
    with norm_path.open() as input_file:
        first_norm = json.load(input_file)
    first_fingerprint = first["snapshot"]["dataset_fingerprint"]
    assert first_metadata["dataset_fingerprint"] == first_fingerprint
    assert first_norm["metadata"]["dataset_fingerprint"] == first_fingerprint

    second = prepare_libero_data(
        data_dir=str(data_dir),
        subsets=["libero_object"],
        metadata_path=str(metadata_path),
        norm_stats_path=str(norm_path),
        allow_incomplete=True,
    )
    assert not second["metadata_rebuilt"]
    assert not second["norm_stats_rebuilt"]

    previous_mtime = h5_path.stat().st_mtime_ns
    with h5py.File(h5_path, "a") as h5_file:
        del h5_file["data/demo_1"]
    os.utime(h5_path, ns=(previous_mtime + 1_000_000_000,) * 2)

    third = prepare_libero_data(
        data_dir=str(data_dir),
        subsets=["libero_object"],
        metadata_path=str(metadata_path),
        norm_stats_path=str(norm_path),
        allow_incomplete=True,
    )
    assert third["metadata_rebuilt"]
    assert third["norm_stats_rebuilt"]
    assert third["snapshot"]["dataset_fingerprint"] != first_fingerprint

    with metadata_path.open() as input_file:
        refreshed_metadata = json.load(input_file)
    with norm_path.open() as input_file:
        refreshed_norm = json.load(input_file)
    refreshed_fingerprint = third["snapshot"]["dataset_fingerprint"]
    assert refreshed_metadata["num_episodes"] == 1
    assert refreshed_metadata["dataset_fingerprint"] == refreshed_fingerprint
    assert refreshed_norm["metadata"]["num_demos"] == 1
    assert refreshed_norm["metadata"]["dataset_fingerprint"] == refreshed_fingerprint


def test_prepare_can_exclude_a_task_without_moving_hdf5(tmp_path):
    data_dir = tmp_path / "datasets"
    subset_dir = data_dir / "libero_object"
    _write_demo_file(subset_dir / "task_one_demo.hdf5")
    _write_demo_file(subset_dir / "task_two_demo.hdf5")
    metadata_path = tmp_path / "libero_object_train.json"
    norm_path = tmp_path / "libero_object_norm.json"

    result = prepare_libero_data(
        data_dir=str(data_dir),
        subsets=["libero_object"],
        metadata_path=str(metadata_path),
        norm_stats_path=str(norm_path),
        allow_incomplete=True,
        exclude_patterns=["task_two_demo.hdf5"],
    )

    assert result["snapshot"]["num_files"] == 1
    assert result["snapshot"]["excluded_files"][0]["relative_path"].endswith(
        "task_two_demo.hdf5"
    )
    with metadata_path.open() as input_file:
        metadata = json.load(input_file)
    with norm_path.open() as input_file:
        norm = json.load(input_file)
    assert metadata["exclude_patterns"] == ["task_two_demo.hdf5"]
    assert len(metadata["datalist"]) == 1
    assert norm["metadata"]["exclude_patterns"] == ["task_two_demo.hdf5"]


def test_prepare_repairs_derived_file_configuration_drift(tmp_path):
    data_dir = tmp_path / "datasets"
    _write_demo_file(data_dir / "libero_object" / "task_one_demo.hdf5")
    metadata_path = tmp_path / "libero_object_train.json"
    norm_path = tmp_path / "libero_object_norm.json"

    prepare_libero_data(
        data_dir=str(data_dir),
        subsets=["libero_object"],
        metadata_path=str(metadata_path),
        norm_stats_path=str(norm_path),
        allow_incomplete=True,
    )
    with metadata_path.open() as input_file:
        metadata = json.load(input_file)
    metadata["datalist"] = []
    with metadata_path.open("w") as output_file:
        json.dump(metadata, output_file)

    with norm_path.open() as input_file:
        norm = json.load(input_file)
    norm["metadata"]["state_orientation_format"] = "euler"
    with norm_path.open("w") as output_file:
        json.dump(norm, output_file)

    repaired = prepare_libero_data(
        data_dir=str(data_dir),
        subsets=["libero_object"],
        metadata_path=str(metadata_path),
        norm_stats_path=str(norm_path),
        allow_incomplete=True,
    )
    assert repaired["metadata_rebuilt"]
    assert repaired["norm_stats_rebuilt"]
    with metadata_path.open() as input_file:
        repaired_metadata = json.load(input_file)
    with norm_path.open() as input_file:
        repaired_norm = json.load(input_file)
    assert len(repaired_metadata["datalist"]) == 1
    assert repaired_norm["metadata"]["state_orientation_format"] == "axis_angle"


def test_unmatched_exclude_pattern_fails_loudly(tmp_path):
    data_dir = tmp_path / "datasets"
    _write_demo_file(data_dir / "libero_object" / "task_one_demo.hdf5")

    try:
        scan_libero_dataset(
            str(data_dir),
            ["libero_object"],
            exclude_patterns=["misspelled_task_demo.hdf5"],
        )
    except RuntimeError as error:
        assert "did not match any HDF5 file" in str(error)
    else:
        raise AssertionError("An unmatched exclusion pattern should fail")
