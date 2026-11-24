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

struct Lanes;

// Frees whatever silero_begin bolted onto the run. The destructor does it, so
// callers never have to; it is public because Lanes lives in headers that must
// not see what the device fast path actually is.
void silero_end(Lanes& run);

struct Lanes {
    int count = 0;
    std::vector<float> state, context, window;
    // Opaque handle on the plumbing silero_begin bolts on once a run is proven
    // against the live model: pinned staging, and on the stock contract an
    // IoBinding with a recurrent state that never leaves the GPU. Owned by the
    // run, freed by the destructor, null on the plain path. Which structure it
    // points at follows fast_kind, because the two model families the engine
    // speaks to carry their state differently and free it differently.
    void* fast = nullptr;
    int fast_kind = 0;   // FAST_NONE / FAST_STOCK / FAST_SEQUENCE

    Lanes() = default;
    ~Lanes() { silero_end(*this); }
    // The fast path cannot be copied, only owned. Nothing in the engine copies
    // a run today; this keeps anyone who tries honest.
    Lanes(const Lanes&) = delete;
    Lanes& operator=(const Lanes&) = delete;
};

// How the two model families this engine speaks to differ, as read off the
// session rather than assumed. Both answer "how likely is it that someone is
// talking in this 32ms window"; they disagree about everything else.
//
//   stock     inputs  input [lanes, 576], state [2, lanes, 128], sr
//             outputs output [lanes], stateN [2, lanes, 128]
//             One Run advances every lane by exactly one window. The lanes are
//             independent time slices, and the batch axis is what carries them.
//
//   sequence  inputs  input [K, 576], h [1, 1, 128], c [1, 1, 128]
//             outputs speech_probs [K], hn [1, 1, 128], cn [1, 1, 128]
//             One Run advances one time slice by K windows, and hands back all
//             K scores. The state is one LSTM pair with its batch axis nailed to
//             one, so lanes here are separate Runs rather than one batch, and
//             what makes it fast is K: K model invocations collapse into one.
//             There is no sr either, the rate is compiled into the graph.
enum SileroContract {
    SILERO_CONTRACT_UNKNOWN = 0,
    SILERO_CONTRACT_STOCK,
    SILERO_CONTRACT_SEQUENCE
};

// What the plumbing behind Lanes::fast actually is. Follows the contract, and
// is recorded per run so silero_end can free it without consulting the session.
enum SileroFastKind {
    FAST_NONE = 0,
    FAST_STOCK,
    FAST_SEQUENCE
};

int silero_begin(Lanes& run, int lanes);

// Windows per lane per call: 1 on the stock contract, K on the sequence one.
// Callers lay out both the audio they hand silero_step and the scores they get
// back at this stride, which is the whole difference between the two contracts
// as far as a caller is concerned. Meaningful after a successful silero_open().
int silero_step_width();

// Which contract the live session was read as, so callers that care about what
// the model actually is can say so in their own logging. Only meaningful after
// a successful silero_open(); it never changes underneath a live session.
SileroContract silero_contract();

// lanes_pcm holds count * silero_step_width() windows of SILERO_WINDOW samples
// each, lane-major, and within a lane in time order. out holds the same number
// of scores in the same order. On the sequence contract the windows inside one
// lane are consecutive windows of one time slice, which is what the model
// expects to be shown; the context between them is glued on in here.
bool silero_step(Lanes& run, const float* lanes_pcm, float* out);

// True once silero_open() has the session running on a GPU execution provider
// (CUDA or DirectML). False means the CPU provider. Only meaningful after a
// successful silero_open(); callers use it for logging and cache keys only.
bool silero_on_gpu();

// The batch width a caller should ask silero_begin for. The per-call overhead of
// this model dominates a step, so the width of a step is most of its throughput,
// and a GPU pays far more overhead per call than a CPU - it wants a wider batch.
// Callers thin this down when the audio they have will not fill it (every lane
// needs enough windows to amortize the contextless first one); the value is a
// request that silero_begin proves against the model before granting it. Only
// meaningful after a successful silero_open().
int silero_preferred_lanes();
