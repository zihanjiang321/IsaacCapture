# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TerminalKeySource restores the terminal on every exit path (against a pseudo-terminal)."""

import os
import pty
import termios

import pytest

from keyboard_terminal_example import TerminalKeySource


@pytest.fixture
def terminal():
    """A TerminalKeySource over a pty whose probe finds a kitty-capable terminal."""
    master, slave = pty.openpty()
    source = TerminalKeySource(fd=slave)
    source._write = lambda data: os.write(slave, data)  # keep escapes off the real tty
    source._read_until = lambda terminator, timeout_s: b"\x1b[?1u\x1b[?62c"
    before = termios.tcgetattr(slave)
    yield source, before, slave
    os.close(master)
    os.close(slave)


def _fail(message, exc=RuntimeError):
    def fail(*args, **kwargs):
        raise exc(message)

    return fail


def test_normal_exit_restores_the_terminal(terminal):
    source, before, fd = terminal
    with source:
        assert source.kitty
        assert termios.tcgetattr(fd) != before
    assert termios.tcgetattr(fd) == before


def test_failing_listener_still_restores_the_terminal(terminal):
    source, before, fd = terminal
    source.add_key_listener(lambda code, pressed: None, _fail("listener"))

    with pytest.raises(RuntimeError, match="listener"):
        with source:
            pass
    assert termios.tcgetattr(fd) == before


def test_closed_output_still_restores_the_terminal(terminal):
    source, before, fd = terminal
    with source:
        source._write = _fail("stdout closed", BrokenPipeError)
    assert termios.tcgetattr(fd) == before


def test_body_exception_survives_a_failing_reset(terminal):
    source, before, fd = terminal
    with pytest.raises(ValueError, match="body"):
        with source:
            source._write = _fail("stdout closed", BrokenPipeError)
            raise ValueError("body")
    assert termios.tcgetattr(fd) == before


def test_failed_probe_restores_the_terminal(terminal):
    source, before, fd = terminal
    source._read_until = _fail("probe")

    with pytest.raises(RuntimeError, match="probe"):
        source.__enter__()
    assert termios.tcgetattr(fd) == before
