// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_trackers/gamepad_tracker.hpp"

#include <algorithm>
#include <filesystem>
#include <system_error>
#include <utility>
#include <vector>

namespace core
{

std::optional<std::string> discover_gamepad_device()
{
    const std::filesystem::path by_path_dir = "/dev/input/by-path";
    std::error_code ec;
    if (!std::filesystem::exists(by_path_dir, ec))
        return std::nullopt;

    std::vector<std::string> candidates;
    for (const auto& entry : std::filesystem::directory_iterator(by_path_dir, ec))
    {
        const std::string name = entry.path().filename().string();
        // "*-event-joystick" is the evdev node (/dev/input/eventN) of the same device; the
        // joystick-API reader needs the "*-joystick" one (/dev/input/jsN).
        if (name.ends_with("-joystick") && !name.ends_with("-event-joystick"))
            candidates.push_back(entry.path().string());
    }
    if (candidates.empty())
        return std::nullopt;

    std::sort(candidates.begin(), candidates.end());
    return candidates.front();
}

GamepadTracker::GamepadTracker(std::string device_path) : device_path_(std::move(device_path))
{
}

const Serialized<GamepadOutput>& GamepadTracker::get_data(const ITrackerSession& session) const
{
    return static_cast<const IGamepadTrackerImpl&>(session.get_tracker_impl(*this)).get_data();
}

} // namespace core
