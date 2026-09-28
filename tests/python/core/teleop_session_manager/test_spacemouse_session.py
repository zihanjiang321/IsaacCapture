# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A SpaceMouse-only TeleopSession runs for real, with no plugin, no OpenXR runtime and no mocks.

The in-process SpaceMouseTracker reads a 3Dconnexion HID device. A FIFO stands in for
/dev/hidrawN: the test writes 7-byte HID reports into it and closes it to unplug the device.
"""

import os
import struct

import numpy as np
import pytest

from isaaccapture import deviceio
from isaaccapture.deviceio_trackers import HeadTracker, SpaceMouseTracker
from isaaccapture.retargeting_engine.deviceio_source_nodes import SpaceMouseSource
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)

FULL_SCALE = 350  # raw counts for a normalized axis value of 1.0
BUTTON_LEFT, BUTTON_RIGHT = 0, 1


class FakeSpaceMouse:
    """A FIFO the test holds open for writing, standing in for /dev/hidrawN."""

    def __init__(self, path):
        self.path = str(path)
        os.mkfifo(self.path)
        # O_RDWR keeps a writer attached, so the tracker never sees EOF until unplug().
        self._fd = os.open(self.path, os.O_RDWR | os.O_NONBLOCK)

    def translation(self, x, y, z):
        os.write(self._fd, struct.pack("<Bhhh", 1, x, y, z))

    def buttons(self, *held):
        mask = sum(1 << b for b in held)
        os.write(self._fd, struct.pack("<BB5x", 3, mask))

    def unplug(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


@pytest.fixture
def device(tmp_path):
    fake = FakeSpaceMouse(tmp_path / "hidraw0")
    yield fake
    fake.unplug()


def _buttons(result):
    return np.flatnonzero(np.asarray(result["spacemouse_buttons"][0])).tolist()


def _translation(result):
    return np.asarray(result["spacemouse_translation"][0]).tolist()


def test_requires_openxr_reflects_trackers():
    assert not deviceio.DeviceIOSession.requires_openxr([SpaceMouseTracker()])
    assert deviceio.DeviceIOSession.requires_openxr(
        [SpaceMouseTracker(), HeadTracker()]
    )


def test_spacemouse_session_runs_without_openxr(device):
    spacemouse = SpaceMouseSource(name="spacemouse", device_path=device.path)
    config = TeleopSessionConfig(app_name="SpaceMouseNoOpenXR", pipeline=spacemouse)

    with TeleopSession(config) as session:
        assert session.oxr_session is None

        device.translation(FULL_SCALE, -FULL_SCALE, 0)
        device.buttons(BUTTON_RIGHT)
        result = session.step()
        assert _translation(result) == [1.0, -1.0, 0.0]
        assert _buttons(result) == [BUTTON_RIGHT]

        device.translation(0, 0, 0)
        device.buttons()
        result = session.step()
        assert _translation(result) == [0.0, 0.0, 0.0]
        assert _buttons(result) == []

        device.unplug()
        result = session.step()
        assert result["spacemouse_translation"].is_none
        assert result["spacemouse_buttons"].is_none


def test_missing_spacemouse_yields_none(tmp_path):
    spacemouse = SpaceMouseSource(
        name="spacemouse", device_path=str(tmp_path / "absent")
    )
    config = TeleopSessionConfig(app_name="SpaceMouseMissing", pipeline=spacemouse)

    with TeleopSession(config) as session:
        result = session.step()
        assert result["spacemouse_buttons"].is_none


def test_spacemouse_session_records_and_replays_without_openxr(device, tmp_path):
    from isaaccapture.deviceio_session import McapRecordingConfig, McapReplayConfig

    mcap_path = str(tmp_path / "spacemouse.mcap")
    config = TeleopSessionConfig(
        app_name="SpaceMouseRecord",
        pipeline=SpaceMouseSource(name="spacemouse", device_path=device.path),
        mcap_config=McapRecordingConfig(mcap_path),
    )
    live = []
    with TeleopSession(config) as session:
        for held in ((BUTTON_LEFT,), ()):
            device.buttons(*held)
            live.append(_buttons(session.step()))

    replay_config = TeleopSessionConfig(
        app_name="SpaceMouseReplay",
        pipeline=SpaceMouseSource(name="spacemouse"),
        mode=SessionMode.REPLAY,
        mcap_config=McapReplayConfig(mcap_path),
    )
    with TeleopSession(replay_config) as session:
        replayed = [_buttons(session.step()) for _ in live]

    assert live == [[BUTTON_LEFT], []]
    assert replayed == live
