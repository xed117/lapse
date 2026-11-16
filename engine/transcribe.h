// LAPSE - Language-Agnostic subtitle synchronization engine
// Copyright (C) 2026 Rasmus Stisen Jensen (rs-jensen)
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#pragma once
#include <optional>
#include <string>
#include <vector>

// lang: an ISO language code the caller wants whisper-cli to force with -l.
// When it is empty we fall back to "en" for an English-only model and "auto" otherwise.
//
// dtw: the whisper-cli DTW preset name (tiny, base.en, large.v3.turbo, ...) that
// enables dynamic time warping for tighter word timestamps. It is passed through
// to whisper-cli verbatim as -dtw <value>. When it is nullopt or empty, DTW is
// off and whisper-cli is run without -dtw at all.
std::vector<int> spoken_words(const std::string& wav, const std::string& model,
                              std::optional<std::string> lang = std::nullopt,
                              std::optional<std::string> dtw = std::nullopt);
