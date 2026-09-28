# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Keyboard Source Node - in-process keyboard device for the retargeting engine.

Keys reach Isaac Teleop from whatever surface the user has focused (a native window, a
browser tab, ...) through :class:`KeyboardProvider` objects on the source's
``KeyboardTracker``. The tracker merges every provider, records to MCAP like any DeviceIO
device, and this node converts each frame to two 256-entry bitmaps indexed by evdev key code
(:class:`EvdevKeyCode`). Carries no semantic mapping: which keys mean what is up to the
consuming retargeter.

Hosts either drive a provider directly (``create_provider``) or hand the source any object
implementing :class:`KeyEventSource` (``attach``).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, TYPE_CHECKING, runtime_checkable
from .interface import IDeviceIOSource
from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..tensor_types import NDArrayType, DLDataType
from ..interface.tensor_group_type import OptionalType, TensorGroupType
from .deviceio_tensor_types import DeviceIOKeyboardOutputTracked

if TYPE_CHECKING:
    from isaaccapture.deviceio_trackers import ITracker, KeyboardProvider
    from isaaccapture.schema import KeyboardOutput


@runtime_checkable
class KeyEventSource(Protocol):
    """A focused input surface that can feed a :class:`KeyboardSource`.

    Implemented by the host (a window, a browser viewer, ...) without importing Isaac Teleop
    types. Requirements:

    - Call ``on_key(code, pressed)`` for press and release only while the surface has focus;
      ``code`` is a W3C ``KeyboardEvent.code`` string ("KeyW", "ArrowUp", ...) or an evdev int.
      Autorepeat may be forwarded; repeated presses of a held key are ignored.
    - Call ``on_focus_lost()`` on blur, close or disconnect so no key can stay stuck.
    - The host's own UI text input always wins: never forward keys typed into its text fields,
      and call ``on_focus_lost()`` the moment its UI takes the keyboard (a text field gains
      focus), exactly as on blur -- otherwise a key held at that moment never sees its release.
    - ``set_keyboard_captured`` yields only host shortcuts that conflict with teleop (camera
      keys, hotkeys); it never blocks the host's text input.
    - A surface that cannot report releases (press-only hotkeys, a plain terminal) reports each
      keystroke as ``on_key(code, True)`` immediately followed by ``on_key(code, False)``: a
      tap that reaches ``keyboard_pressed`` but is never held.
    - Callbacks may arrive on any thread.
    """

    supports_keyboard: bool

    def add_key_listener(
        self,
        on_key: Callable[[str | int, bool], None],
        on_focus_lost: Callable[[], None],
    ) -> Callable[[], None]:
        """Subscribe; returns a callable that unsubscribes."""
        ...

    def set_keyboard_captured(self, captured: bool) -> None:
        """While captured, the host disables its own conflicting key bindings (e.g. camera keys)."""
        ...


# Every standard PC keyboard key (letters, digits, function keys, navigation,
# modifiers, numpad, punctuation) fits under evdev code 255; anything above that
# is an exotic/vendor key not tracked here.
ALL_KEYS_BITMAP_SIZE = 256


def _key_bitmap_type(name: str) -> TensorGroupType:
    return TensorGroupType(
        name,
        [
            NDArrayType(
                "bitmap",
                shape=(ALL_KEYS_BITMAP_SIZE,),
                dtype=DLDataType.UINT,
                dtype_bits=8,
            )
        ],
    )


def KeyboardAllKeysType() -> TensorGroupType:
    """Type for "keyboard_all_keys": keys held at the end of the frame, indexed by evdev code."""
    return _key_bitmap_type("keyboard_all_keys")


def KeyboardPressedType() -> TensorGroupType:
    """Type for "keyboard_pressed": keys with at least one press event during the frame.

    Catches taps shorter than a frame and gives toggles an edge without keeping state.
    """
    return _key_bitmap_type("keyboard_pressed")


class KeyboardSource(IDeviceIOSource):
    """
    In-process keyboard device: KeyboardTracker providers -> key bitmaps.

    Inputs:
        - "deviceio_keyboard": KeyboardOutput from the source's KeyboardTracker

    Outputs (Optional -- absent while no provider is attached):
        - "keyboard_all_keys": 256-entry uint8 bitmap, 1 = held at the end of the frame
        - "keyboard_pressed": 256-entry uint8 bitmap, 1 = pressed at least once this frame

    Usage:
        keyboard = KeyboardSource("keyboard")
        detach = keyboard.attach(window)            # any KeyEventSource
        # or, driving a provider directly:
        provider = keyboard.create_provider("my_window")
        provider.key_down("KeyW"); provider.key_up("KeyW")
    """

    def __init__(self, name: str) -> None:
        """Initialize the keyboard source and its in-process KeyboardTracker.

        Args:
            name: Unique name for this source node (also its MCAP channel base name).
        """
        from isaaccapture.deviceio_trackers import KeyboardTracker

        self._keyboard_tracker = KeyboardTracker()
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        """Get the KeyboardTracker instance for TeleopSession to register."""
        return self._keyboard_tracker

    def create_provider(self, name: str) -> "KeyboardProvider":
        """Create a provider for one input surface. Close it (or use ``with``) to detach."""
        return self._keyboard_tracker.create_provider(name)

    def attach(
        self,
        surface: KeyEventSource,
        *,
        name: str | None = None,
        capture: bool = True,
    ) -> Callable[[], None]:
        """Feed this source from ``surface``; returns a callable that detaches it.

        Creates a provider, subscribes it to the surface's key and focus-loss callbacks and,
        when ``capture`` is set, asks the surface to yield its own key bindings. Detaching
        unsubscribes, releases the provider's keys and restores the surface's bindings.
        """
        if not getattr(surface, "supports_keyboard", False):
            raise ValueError(
                f"{type(surface).__name__} does not support keyboard input"
            )
        provider = self.create_provider(name or type(surface).__name__)

        def on_key(code: str | int, pressed: bool) -> None:
            if pressed:
                provider.key_down(code)
            else:
                provider.key_up(code)

        unsubscribe = surface.add_key_listener(on_key, provider.release_all)
        if capture:
            surface.set_keyboard_captured(True)

        detached = False

        def detach() -> None:
            nonlocal detached
            if detached:
                return
            detached = True
            unsubscribe()
            if capture:
                surface.set_keyboard_captured(False)
            provider.close()

        return detach

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll the keyboard tracker and return input data.

        Returns:
            Dict with "deviceio_keyboard" TensorGroup containing KeyboardOutput | None.
        """
        keys = self._keyboard_tracker.get_keyboard_data(deviceio_session)
        tg = TensorGroup(DeviceIOKeyboardOutputTracked())
        tg[0] = keys
        return {"deviceio_keyboard": tg}

    def input_spec(self) -> RetargeterIOType:
        """Declare DeviceIO keyboard input."""
        return {
            "deviceio_keyboard": DeviceIOKeyboardOutputTracked(),
        }

    def output_spec(self) -> RetargeterIOType:
        """Declare the held and pressed bitmaps (Optional -- absent without a provider)."""
        return {
            "keyboard_all_keys": OptionalType(KeyboardAllKeysType()),
            "keyboard_pressed": OptionalType(KeyboardPressedType()),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Convert KeyboardOutput to the held and pressed bitmaps.

        Calls ``set_none()`` on both outputs when no provider is attached.
        """
        import numpy as np

        from isaaccapture.schema import KeyAction

        keys: KeyboardOutput | None = inputs["deviceio_keyboard"][0]

        all_keys_out = outputs["keyboard_all_keys"]
        pressed_out = outputs["keyboard_pressed"]
        if keys is None:
            all_keys_out.set_none()
            pressed_out.set_none()
            return

        held = np.zeros(ALL_KEYS_BITMAP_SIZE, dtype=np.uint8)
        for code in keys.pressed_keys:
            if code < ALL_KEYS_BITMAP_SIZE:
                held[code] = 1

        pressed = np.zeros(ALL_KEYS_BITMAP_SIZE, dtype=np.uint8)
        for event in keys.events:
            if event.action == KeyAction.PRESS and event.code < ALL_KEYS_BITMAP_SIZE:
                pressed[event.code] = 1

        all_keys_out[0] = held
        pressed_out[0] = pressed
