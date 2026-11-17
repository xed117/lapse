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
#include <string>

// Silero wants 512 samples at a time when it is fed 16kHz audio
const int SILERO_WINDOW = 512;
const int SILERO_RATE = 16000;

bool silero_open();
void silero_close();

// the model does not care how many streams you hand it at once and the per call
// overhead is most of the cost, so a film gets cut into lanes and they all walk
// forward together. silero_begin returns how many lanes it actually got
#include <vector>
struct Lanes {
    int count = 0;
    std::vector<float> state, context, window;
};

int silero_begin(Lanes& run, int lanes);
bool silero_step(Lanes& run, const float* lanes_pcm, float* out);

// True once silero_open() has the session running on a GPU execution provider
// (CUDA or DirectML). False means the CPU provider. Only meaningful after a
// successful silero_open(); callers use it for logging and cache keys only.
bool silero_on_gpu();
