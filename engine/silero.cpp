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

#include "silero.h"
#include "onnxruntime_c_api.h"
#include "log.h"

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    typedef HMODULE library_handle;
#elif defined(__APPLE__)
    #include <dlfcn.h>
    #include <mach-o/dyld.h>
    typedef void* library_handle;
#else
    #include <dlfcn.h>
    #include <sys/mman.h>
    #include <unistd.h>
    typedef void* library_handle;
#endif

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <chrono>
#include <string>
#include <new>

static library_handle library = nullptr;
static const OrtApi* ort = nullptr;
static OrtEnv* env = nullptr;
static OrtSessionOptions* options = nullptr;
static OrtSession* session = nullptr;
static OrtMemoryInfo* memory = nullptr;
static bool tried = false;

// The engine runs analyse() on a crew of threads that all share this one session.
// onnxruntime guarantees Run() is safe to call from several threads at once and every
// caller carries its own Lanes, so only opening and closing needs a latch here.
static std::mutex open_mutex;
static bool gpu = false;
static std::string chosen_provider;
// Which of the two contracts the live session turned out to answer to, and how
// many windows of one time slice a single call advances it by. Both are decided
// once, by read_contract, on the session that won the provider loop, and stay
// put for the life of it.
static SileroContract contract = SILERO_CONTRACT_UNKNOWN;
static int seq_windows = 0;
// One half of the recurrent state, as the session declared it. The stock
// contract's is 2 * lanes * 128 for the whole pair, so it is not a per-lane
// figure; the sequence contract's is per lane and identical for every lane,
// because its state tensors carry a batch axis the export nailed to one.
static size_t seq_state_elems = 0;
static int cuda_device = 0;
static long long cuda_mem_bytes = 512LL * 1024 * 1024;

// The model wants the tail of the previous chunk glued to the front of this one. Without it the numbers come back near zero for speech that is plainly there
static const int SILERO_CONTEXT = 64;

// Every silero export this engine has met keeps its recurrent state as
// [2, lanes, 128]. try_fast_path sizes and labels its buffers by what the
// model declares instead; these are the fallback for a run whose model will
// not declare itself, and what the plain path in silero_step feeds the
// runtime, which accepts the silero convention it was exported for.
static const int SILERO_STATE_DIRS = 2;
static const int SILERO_STATE_HIDDEN = 128;

// How deep the shape queries go before the model stops looking like anything
// this engine speaks to. Silero's tensors are rank 1 to 3.
static const size_t MAX_SHAPE_RANK = 4;

// The concrete shapes read_contract settled on, with every dynamic axis
// resolved to the engine's chosen K. These are what the sequence path builds
// its tensors from; the probe's copy is transient and lives only as long as
// the detection call that filled it.
static int64_t seq_window_dims[2] = {0, 0};
static int64_t seq_score_dims[MAX_SHAPE_RANK]; static size_t seq_score_rank = 0;
static int64_t seq_state_dims[MAX_SHAPE_RANK]; static size_t seq_state_rank = 0;

static const char* input_names[]  = {"input", "state", "sr"};
static const char* output_names[] = {"output", "stateN"};

// The sequence export - silero's own onnx_sequence flavour - answers the same
// question as the stock one and is otherwise a different program. It has no
// sample-rate input, because the rate is baked into the graph; it hands the
// LSTM's two halves over as two separate rank-3 tensors instead of one stacked
// [2, lanes, 128]; and its leading axis is not a set of independent time slices
// stepping forward in lockstep, it is how many consecutive windows of a single
// time slice the model chews through in one call. Those three differences are
// why the stock names arriving at a sequence model come back as "Invalid input
// name: state", and why mapping the names across is not enough on its own.
static const char* seq_input_names[]  = {"input", "h", "c"};
static const char* seq_output_names[] = {"speech_probs", "hn", "cn"};

// Windows of one time slice per call on the sequence contract. That axis is
// dynamic, so this number is the engine's choice rather than the model's, and
// the probe step in silero_begin proves it against the live session before
// anyone commits to it. 512 is sixteen seconds of audio per call, which is
// where silero's own benchmarking stops gaining; LAPSE_VAD_CHUNK overrides it.
static const int SEQ_WINDOWS = 512;
static const int SEQ_WINDOWS_MAX = 4096;

static bool ok(OrtStatus* status) {
    if (!status) return true;
    std::cerr << "onnxruntime: " << ort->GetErrorMessage(status) << '\n';
    ort->ReleaseStatus(status);
    return false;
}

// Page-locked host memory without linking the CUDA runtime. The driver treats
// locked pages as pinned, so the copy paths DMA them straight instead of
// staging through a bounce buffer of their own, which is what turns the per
// step host-to-device transfer from a synchronous copy into a real one.
// Windows locks a committed region, POSIX mlocks the mapping; both round the
// request up to the page size the machine actually uses, asked of the kernel
// rather than assumed, so unlocking covers exactly what was locked.
static size_t page_bytes(void) {
    static size_t page = 0;
    if (page == 0) {
#ifdef _WIN32
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        page = (size_t)info.dwPageSize;
#else
        long size = sysconf(_SC_PAGESIZE);
        page = size > 0 ? (size_t)size : 4096;
#endif
    }
    if (page < 4096) page = 4096;   // the pinned staging is DMA'd; small pages stay a bad idea
    return page;
}

static void* pin_alloc(size_t bytes) {
    size_t page = page_bytes();
    size_t pages = (bytes + page - 1) & ~(page - 1);
#ifdef _WIN32
    void* p = VirtualAlloc(nullptr, pages, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) return nullptr;
    if (!VirtualLock(p, pages)) {
        VirtualFree(p, 0, MEM_RELEASE);
        return nullptr;
    }
    return p;
#else
    void* p = nullptr;
    if (posix_memalign(&p, page, pages) != 0) return nullptr;
    if (mlock(p, pages) != 0) {
        free(p);
        return nullptr;
    }
    return p;
#endif
}

static void pin_free(void* p, size_t bytes) {
    if (!p) return;
    size_t page = page_bytes();
    size_t pages = (bytes + page - 1) & ~(page - 1);
#ifdef _WIN32
    VirtualUnlock(p, pages);
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munlock(p, pages);
    free(p);
#endif
}

// Everything a lane touches every step, parked once instead of rebuilt per
// 32ms of audio. The window and the scores cross through page-locked staging;
// the tensors that wrap them live as long as the run; and on CUDA the
// recurrent state never comes back to the host at all - it ping-pongs between
// two device buffers behind one IoBinding so no step ever reads and writes the
// same buffer, and the only transfers left per step are the window in and the
// scores out. Shapes are nailed down by silero_begin, so nothing needs
// rebinding between steps except those two state slots.
struct FastPath {
    bool device_state = false;         // CUDA only: the state stays on the device
    int active = 0;                    // which device buffer the last step wrote
    bool pending_zero = true;          // the next step reads the zero state

    size_t window_bytes = 0, score_bytes = 0, rate_bytes = 0, state_bytes = 0;

    float* pin_window = nullptr;       // page-locked input staging
    float* pin_score = nullptr;        // page-locked score output
    int64_t* pin_rate = nullptr;       // page-locked sample rate, written once
    float* pin_state_zero = nullptr;   // page-locked zeros, first step after a reset

    OrtValue* v_window = nullptr;
    OrtValue* v_sr = nullptr;
    OrtValue* v_score = nullptr;
    OrtValue* v_state_zero = nullptr;
    OrtValue* v_state_host = nullptr;  // host mode: wraps run.state in place
    OrtValue* v_dev_state[2] = {nullptr, nullptr};

    OrtIoBinding* binding = nullptr;
    OrtMemoryInfo* cuda_mem = nullptr;
    OrtMemoryInfo* host_mem = nullptr;
    OrtAllocator* gpu_alloc = nullptr;
    float* dev_state[2] = {nullptr, nullptr};
};

static void fast_teardown(FastPath* fast) {
    if (!fast) return;
    if (ort) {
        if (fast->binding) ort->ReleaseIoBinding(fast->binding);
        if (fast->v_window) ort->ReleaseValue(fast->v_window);
        if (fast->v_sr) ort->ReleaseValue(fast->v_sr);
        if (fast->v_score) ort->ReleaseValue(fast->v_score);
        if (fast->v_state_zero) ort->ReleaseValue(fast->v_state_zero);
        if (fast->v_state_host) ort->ReleaseValue(fast->v_state_host);
        for (OrtValue* value : fast->v_dev_state) if (value) ort->ReleaseValue(value);
        if (fast->gpu_alloc) {
            for (float* block : fast->dev_state)
                // AllocatorFree answers with an OrtStatus; on a teardown there is
                // nothing left to do about a refusal but say so, which ok() does.
                if (block) ok(ort->AllocatorFree(fast->gpu_alloc, block));
            ort->ReleaseAllocator(fast->gpu_alloc);
        }
        if (fast->host_mem) ort->ReleaseMemoryInfo(fast->host_mem);
        if (fast->cuda_mem) ort->ReleaseMemoryInfo(fast->cuda_mem);
    }
    pin_free(fast->pin_window, fast->window_bytes);
    pin_free(fast->pin_score, fast->score_bytes);
    pin_free(fast->pin_rate, fast->rate_bytes);
    pin_free(fast->pin_state_zero, fast->state_bytes);
    delete fast;
}

// Everything the fast path needs to know about the model before it lays a
// single buffer out, read from the session rather than assumed. The score has
// been [lanes] for as long as the old exports but the v6 exports hand back
// [lanes, 1]; nothing in this file should be the authority on what the model
// looks like. try_fast_path only runs on a model that declared every tensor
// this engine touches, in a shape and dtype the run can be sized by, because
// a wrong guess is rejected at bind time at best and reads past the staging at
// worst. Anything else returns false and the run keeps working through the
// path where the runtime allocates its own everything.
struct ModelProbe {
    // dims are as declared until sized_dims runs, and carry the run's lane
    // count substituted in after it.
    int64_t window_dims[MAX_SHAPE_RANK]; size_t window_rank = 0; size_t window_elems = 0;
    int64_t state_dims[MAX_SHAPE_RANK];  size_t state_rank = 0;  size_t state_elems = 0;
    int64_t rate_dims[MAX_SHAPE_RANK];   size_t rate_rank = 0;
    int64_t score_dims[MAX_SHAPE_RANK];  size_t score_rank = 0;  size_t score_elems = 0;
};

// Finds one named I/O of the live session and reports its element type and
// dimensions. The names come from input_names/output_names, never an index.
static bool shape_by_name(bool input, const char* name, ONNXTensorElementDataType& type,
                          int64_t* dims, size_t& rank) {
    OrtAllocator* allocator = nullptr;
    if (!ok(ort->GetAllocatorWithDefaultOptions(&allocator)) || !allocator) return false;

    size_t total = 0;
    bool listed = input ? ok(ort->SessionGetInputCount(session, &total))
                        : ok(ort->SessionGetOutputCount(session, &total));
    if (!listed) return false;

    for (size_t i = 0; i < total; i++) {
        char* found = nullptr;
        bool named = input ? ok(ort->SessionGetInputName(session, i, allocator, &found))
                           : ok(ort->SessionGetOutputName(session, i, allocator, &found));
        if (!named) break;
        bool wanted = found && std::strcmp(found, name) == 0;
        if (found) allocator->Free(allocator, found);
        if (!wanted) continue;

        OrtTypeInfo* type_info = nullptr;
        bool typed = input ? ok(ort->SessionGetInputTypeInfo(session, i, &type_info))
                           : ok(ort->SessionGetOutputTypeInfo(session, i, &type_info));
        if (!typed) return false;

        const OrtTensorTypeAndShapeInfo* info = nullptr;
        bool got = ok(ort->CastTypeInfoToTensorInfo(type_info, &info)) && info != nullptr;
        bool read = false;
        if (got && ok(ort->GetTensorElementType(info, &type)) &&
            ok(ort->GetDimensionsCount(info, &rank)) &&
            rank >= 1 && rank <= MAX_SHAPE_RANK &&
            ok(ort->GetDimensions(info, dims, rank)))
            read = true;
        ort->ReleaseTypeInfo(type_info);
        return read;   // a name appears at most once in the I/O list
    }
    return false;
}

// Replaces the batch with the lane count this run carries. The batch is the
// one axis the run sizes for itself - wherever the model left a dynamic axis
// (-1, or the symbolic name that comes back as -2); a model with two of those
// has an axis nobody can size buffers for. Concrete axes are kept as declared,
// and the element count that comes back is what the caller sizes buffers by,
// so a model whose batch was nailed to some other number fails the caller's
// count checks instead of being force-fit into them.
static bool sized_dims(int64_t* dims, size_t rank, int lanes, size_t& elems) {
    elems = 1;
    bool batch = false;
    for (size_t d = 0; d < rank; d++) {
        if (dims[d] <= 0) {
            if (batch) return false;
            dims[d] = lanes;
            batch = true;
        }
        elems *= (size_t)dims[d];
    }
    return true;
}

// Fills the probe from the session: element types as declared, shapes with the
// run's lanes substituted in, and element counts the buffers are sized from.
// Then checks those counts against what this engine will actually feed and
// read: the assembled window of context-plus-samples per lane, and exactly one
// score per lane, which is all the out[] array silero_step hands the caller
// holds. The state is checked by the caller against the buffer silero_begin
// made for it. A model that declares anything else is not a model this engine
// knows how to feed.
static bool model_probe(int lanes, ModelProbe& probe) {
    ONNXTensorElementDataType window_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ONNXTensorElementDataType state_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ONNXTensorElementDataType rate_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ONNXTensorElementDataType score_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

    bool known =
        shape_by_name(true, input_names[0], window_type, probe.window_dims, probe.window_rank) &&
        shape_by_name(true, input_names[1], state_type, probe.state_dims, probe.state_rank) &&
        shape_by_name(true, input_names[2], rate_type, probe.rate_dims, probe.rate_rank) &&
        shape_by_name(false, output_names[0], score_type, probe.score_dims, probe.score_rank);
    if (!known) return false;

    // Everything this engine feeds and reads is float, and the sample rate has
    // always been the one-element int64 the exports declare. A model that
    // wants anything else is not a model this engine knows how to feed.
    if (window_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        state_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        score_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        rate_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) return false;
    if (probe.rate_rank != 1 || probe.rate_dims[0] != 1) return false;

    if (!sized_dims(probe.window_dims, probe.window_rank, lanes, probe.window_elems)) return false;
    if (!sized_dims(probe.state_dims, probe.state_rank, lanes, probe.state_elems)) return false;
    if (!sized_dims(probe.score_dims, probe.score_rank, lanes, probe.score_elems)) return false;

    // The window is what silero_step assembles: context glued in front of the
    // fresh samples, once per lane. A model that declares a different width,
    // or a batch this run is not feeding, lands here instead of on a bind
    // error at the first step.
    if (probe.window_elems != (size_t)lanes * (SILERO_CONTEXT + SILERO_WINDOW)) return false;
    if (probe.score_elems != (size_t)lanes) return false;
    return true;
}

// Does this session have an I/O of that name at all. Contract detection asks
// that question before it asks anything about shapes, because the two families
// this engine speaks to are told apart by their names first and their shapes
// second - and a name is the one thing that cannot be guessed at or coerced.
static bool has_io(bool input, const char* name) {
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    int64_t dims[MAX_SHAPE_RANK];
    size_t rank = 0;
    return shape_by_name(input, name, type, dims, rank);
}

// Everything the sequence contract needs proved before a byte is laid out.
// Unlike the stock probe there is no lane count handed in: this export's state
// tensors carry a batch axis the exporter nailed to one, so parallelism comes
// from running lanes one after another and the only axis this engine gets to
// size is the sequence length, which is how many consecutive windows of a
// single time slice one call chews through.
struct SeqProbe {
    int64_t window_dims[MAX_SHAPE_RANK]; size_t window_rank = 0;
    int64_t state_dims[MAX_SHAPE_RANK];  size_t state_rank = 0;
    int64_t score_dims[MAX_SHAPE_RANK];  size_t score_rank = 0;
    size_t state_elems = 0;
    // The sequence axis as the model declared it: > 0 when it is nailed down to
    // a number, in which case this engine has no say in K at all, and -1 when it
    // is dynamic and K is the engine's to choose.
    int64_t declared_seq = -1;
};

static bool same_dims(const int64_t* a, const int64_t* b, size_t rank) {
    for (size_t d = 0; d < rank; d++) if (a[d] != b[d]) return false;
    return true;
}

// Pins the single dynamic axis a state tensor may leave open to one and counts
// the elements. The sequence export declares h and c as [1, dyn, 128], where
// that middle axis is the LSTM's own and not a batch this engine owns, so it
// resolves to one; a state tensor with two axes left open has more freedom than
// anybody can size a buffer for.
static bool pin_state_dims(int64_t* dims, size_t rank, size_t& elems) {
    if (rank < 1 || rank > MAX_SHAPE_RANK) return false;
    if (dims[0] != 1) return false;
    elems = 1;
    bool open = false;
    for (size_t d = 1; d < rank; d++) {
        if (dims[d] <= 0) {
            if (open) return false;
            dims[d] = 1;
            open = true;
        }
        elems *= (size_t)dims[d];
    }
    return elems > 0;
}

// Reads one of the four state tensors and checks it is the same tensor as a
// reference: same element type, same rank, same dimensions. hn and cn go back
// in as h and c on the next call, so a model that spells any of the four
// differently is not the model this path feeds, and finding that out here beats
// discovering it as a bind error at the first step.
static bool state_matches(const char* name, bool input, ONNXTensorElementDataType wanted,
                          const int64_t* reference, size_t reference_rank) {
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    int64_t dims[MAX_SHAPE_RANK];
    size_t rank = 0;
    if (!shape_by_name(input, name, type, dims, rank)) return false;
    return type == wanted && rank == reference_rank && same_dims(dims, reference, rank);
}

static bool seq_model_probe(SeqProbe& probe) {
    ONNXTensorElementDataType window_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ONNXTensorElementDataType score_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ONNXTensorElementDataType h_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

    // h is read first and kept as the reference the other three are compared
    // against. It lands in the probe's own slots because after the comparison
    // the four are known to agree, and this is the one the buffers are built to.
    if (!shape_by_name(true, seq_input_names[1], h_type, probe.state_dims, probe.state_rank)) return false;

    bool known =
        shape_by_name(true, seq_input_names[0], window_type, probe.window_dims, probe.window_rank) &&
        shape_by_name(false, seq_output_names[0], score_type, probe.score_dims, probe.score_rank) &&
        state_matches(seq_input_names[2], true, h_type, probe.state_dims, probe.state_rank) &&
        state_matches(seq_output_names[1], false, h_type, probe.state_dims, probe.state_rank) &&
        state_matches(seq_output_names[2], false, h_type, probe.state_dims, probe.state_rank);
    if (!known) return false;

    // Everything fed and read is float. The sequence export has no sample-rate
    // input at all, which is what its name is short for; one that does is being
    // read as the stock contract instead.
    if (window_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        h_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        score_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;

    // The rows are context glued in front of fresh samples, exactly as the stock
    // contract assembles them, so the width check is the same one.
    if (probe.window_rank != 2) return false;
    if (probe.window_dims[1] != SILERO_CONTEXT + SILERO_WINDOW) return false;

    // One score per window, so the scores and the rows have to agree about how
    // many windows a call carries. Either may declare it; where both do, they
    // must agree, and a concrete number on either side is the declared one.
    if (probe.score_rank != 1) return false;
    if (probe.window_dims[0] > 0 && probe.score_dims[0] > 0 &&
        probe.window_dims[0] != probe.score_dims[0]) return false;
    probe.declared_seq = probe.window_dims[0] > 0 ? probe.window_dims[0]
                     : probe.score_dims[0] > 0 ? probe.score_dims[0] : -1;
    if (probe.declared_seq > SEQ_WINDOWS_MAX) return false;

    // The state is one lane's worth and no more: its leading axis is the batch
    // the export nailed to one, which is why lanes here are separate calls
    // rather than one batch. Rank 3 because this is the [1, dir, hidden] the
    // exporter writes, and the element count is what every lane's buffer and
    // every ping-pong slot is sized by.
    if (probe.state_rank != 3) return false;
    if (!pin_state_dims(probe.state_dims, probe.state_rank, probe.state_elems)) return false;
    return probe.state_elems > 0;
}

// Decides which of the two contracts the live session answers to, and with the
// sequence one, how wide a call it wants. Runs on a session that has just been
// built and before anything has been run through it, because it is the answer
// that picks which code runs the first throwaway window. Stock is checked first
// so an export this engine has always spoken to keeps being read the way it
// always was; a model that answers to neither is rejected here, loudly, rather
// than being fed stock names and coming back as "Invalid input name: state"
// from every provider in turn.
static bool read_contract() {
    contract = SILERO_CONTRACT_UNKNOWN;
    seq_windows = 0;
    seq_state_elems = 0;
    seq_score_rank = 0;
    seq_state_rank = 0;
    seq_window_dims[0] = seq_window_dims[1] = 0;

    bool stock = has_io(true, input_names[0]) && has_io(true, input_names[1]) &&
                 has_io(true, input_names[2]) && has_io(false, output_names[0]) &&
                 has_io(false, output_names[1]);
    if (stock) {
        contract = SILERO_CONTRACT_STOCK;
        return true;
    }

    bool named = has_io(true, seq_input_names[0]) && has_io(true, seq_input_names[1]) &&
                 has_io(true, seq_input_names[2]) && has_io(false, seq_output_names[0]) &&
                 has_io(false, seq_output_names[1]) && has_io(false, seq_output_names[2]);
    if (!named) return false;

    SeqProbe probe;
    if (!seq_model_probe(probe)) return false;

    int want = SEQ_WINDOWS;
    const char* forced = std::getenv("LAPSE_VAD_CHUNK");
    if (forced) {
        int asked = atoi(forced);
        if (asked > 0) want = asked > SEQ_WINDOWS_MAX ? SEQ_WINDOWS_MAX : asked;
    }
    // A model with the axis nailed down is not open to suggestions.
    if (probe.declared_seq > 0) want = (int)probe.declared_seq;
    if (want < 1) want = 1;

    seq_windows = want;
    seq_state_elems = probe.state_elems;
    // The shapes the sequence path will build its tensors from: the window
    // block and the score both carry the chosen K on their leading axis, and
    // the state dims are the pinned ones the probe already resolved to one.
    seq_window_dims[0] = want;
    seq_window_dims[1] = SILERO_CONTEXT + SILERO_WINDOW;
    seq_score_rank = probe.score_rank;
    seq_score_dims[0] = want;
    seq_state_rank = probe.state_rank;
    for (size_t d = 0; d < seq_state_rank; d++) seq_state_dims[d] = probe.state_dims[d];
    contract = SILERO_CONTRACT_SEQUENCE;
    return true;
}

static std::filesystem::path binary_dir() {
#ifdef _WIN32
    wchar_t buffer[MAX_PATH];
    DWORD length = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    if (length > 0 && length < MAX_PATH) {
        std::filesystem::path dir = std::filesystem::path(buffer).parent_path();
        if (!dir.empty()) return dir;
    }
#else
    // The real file, not argv[0] as typed, so PATH and symlinks find what sits beside it
    std::error_code ec;
    std::filesystem::path exe;
#ifdef __APPLE__
    char buffer[4096];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) exe = std::filesystem::canonical(buffer, ec);
#else
    exe = std::filesystem::canonical("/proc/self/exe", ec);
#endif
    if (!ec && !exe.parent_path().empty()) return exe.parent_path();
#endif
    return std::filesystem::path(".");
}

static library_handle load_from(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::path full = std::filesystem::absolute(path, ec);
    if (ec) full = path;
#ifdef _WIN32
    library_handle handle = LoadLibraryExW(full.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!handle) say() << "Silero VAD: LoadLibrary(" << full.string() << ") failed, error " << GetLastError() << '\n';
    return handle;
#else
    return dlopen(full.string().c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
}

static library_handle load_named(const char* name) {
#ifdef _WIN32
    return LoadLibraryA(name);
#else
    return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
#endif
}

static void* library_symbol(library_handle handle, const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress(handle, name);
#else
    return dlsym(handle, name);
#endif
}

static library_handle load_runtime() {
#ifdef _WIN32
    const char* names[] = {"onnxruntime.dll"};
#elif defined(__APPLE__)
    const char* names[] = {"libonnxruntime.dylib", "libonnxruntime.1.dylib"};
#else
    const char* names[] = {"libonnxruntime.so", "libonnxruntime.so.1"};
#endif

    const char* forced = std::getenv("LAPSE_ONNXRUNTIME");
    if (forced) return load_from(forced);

    std::filesystem::path dir = binary_dir();
    for (const char* name : names) {
        library_handle handle = load_from(dir / name);
        if (handle) return handle;
    }
    for (const char* name : names) {
        library_handle handle = load_named(name);
        if (handle) return handle;
    }
    return nullptr;
}

static std::filesystem::path find_model() {
    std::vector<std::filesystem::path> paths;

    const char* forced = std::getenv("LAPSE_VAD_MODEL");
    if (forced) paths.push_back(forced);

    paths.push_back(binary_dir() / "silero_vad.onnx");
#ifndef _WIN32
    paths.push_back("/usr/local/share/lapse/silero_vad.onnx");
    paths.push_back("/usr/share/lapse/silero_vad.onnx");
#endif
    paths.push_back("silero_vad.onnx");

    for (const std::filesystem::path& path : paths) {
        std::ifstream file(path, std::ios::binary);
        if (file.good()) return path;
    }
    return {};
}

// The names onnxruntime reports, e.g. "CUDAExecutionProvider", "CPUExecutionProvider".
// Empty when the call fails. A provider whose shared library failed to load is simply
// absent from this list, with no error of its own.
static std::vector<std::string> list_providers() {
    std::vector<std::string> names;
    char** raw = nullptr;
    int count = 0;
    if (ok(ort->GetAvailableProviders(&raw, &count))) {
        for (int i = 0; i < count; i++) names.push_back(raw[i]);
        ok(ort->ReleaseAvailableProviders(raw, count));
    }
    return names;
}

static std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return text;
}

// Compares a name onnxruntime reported against the short stem this file works with.
// The runtime spells its providers "CUDAExecutionProvider", "DmlExecutionProvider"
// and "TensorrtExecutionProvider", so comparing stems with the shared suffix and the
// case set aside is what stops a spelling difference from silently leaving only the
// CPU in the running. DirectML is spelled Dml, TensorRT is spelled Tensorrt.
static bool provider_is(const std::string& name, const char* stem) {
    std::string have = lower(name);
    const std::string tail = "executionprovider";
    if (have.size() > tail.size() &&
        have.compare(have.size() - tail.size(), tail.size(), tail) == 0) {
        have.erase(have.size() - tail.size());
    }
    return have == lower(stem);
}

// The name exactly as this onnxruntime spells it, "CUDAExecutionProvider" for "CUDA",
// empty when it does not have one. The by-name append call has to be handed back the
// spelling the runtime uses, so it is looked up rather than guessed at.
static std::string provider_name(const std::vector<std::string>& names, const char* stem) {
    for (const std::string& name : names) {
        if (provider_is(name, stem)) return name;
    }
    return {};
}

// Where a compiled TensorRT engine is kept. Next to the model normally, or wherever
// LAPSE_VAD_TRT_CACHE points. Failing to make the directory is not fatal, the
// provider then just rebuilds the engine at every run instead of reusing one.
static std::string trt_cache_dir(const std::filesystem::path& model) {
    const char* forced = std::getenv("LAPSE_VAD_TRT_CACHE");
    std::filesystem::path dir = forced ? std::filesystem::path(forced)
                                       : model.parent_path() / "trt_cache";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return ec ? std::string(".") : dir.string();
}

// One CUDA option at a time. When this onnxruntime does not recognise a key, or wants
// a value in a form we did not guess, that one option is reported and skipped and the
// rest still apply. Setting them all in a single call lets one option that changed
// between releases fail the whole lot, and the GPU then goes back to the CPU.
static void try_cuda_option(OrtCUDAProviderOptionsV2* cuda, const char* key, const std::string& value) {
    const char* keys[] = {key};
    const char* values[] = {value.c_str()};
    OrtStatus* status = ort->UpdateCUDAProviderOptions(cuda, keys, values, 1);
    if (!status) return;
    say() << "Silero VAD: ignoring CUDA option " << key << "=" << value << ": "
          << ort->GetErrorMessage(status) << "\n";
    ort->ReleaseStatus(status);
}

// The TensorRT equivalent of the above.
static void try_trt_option(OrtTensorRTProviderOptionsV2* trt, const char* key, const std::string& value) {
    const char* keys[] = {key};
    const char* values[] = {value.c_str()};
    OrtStatus* status = ort->UpdateTensorRTProviderOptions(trt, keys, values, 1);
    if (!status) return;
    say() << "Silero VAD: ignoring TensorRT option " << key << "=" << value << ": "
          << ort->GetErrorMessage(status) << "\n";
    ort->ReleaseStatus(status);
}

// Bolts one execution provider onto a fresh set of session options. An empty name
// means plain CPU. CUDA and TensorRT go through their V2 option objects so CUDA can
// cap its memory footprint and TensorRT can be told where to cache its compiled
// engines; everything else (DirectML) uses the generic by-name call. The name handed
// in is whatever this onnxruntime reported, matched by stem rather than spelled out
// here. Returns false when the provider cannot be attached.
static bool append_provider(OrtSessionOptions* opts, const std::string& provider,
                            const std::filesystem::path& model) {
    if (provider.empty()) return true;

    if (provider_is(provider, "CUDA")) {
        OrtCUDAProviderOptionsV2* cuda = nullptr;
        if (!ok(ort->CreateCUDAProviderOptions(&cuda)) || !cuda) return false;

        // arena kSameAsRequested = grow by what was asked for instead of doubling, so
        // we do not squat on the whole card while whisper-cli and the decoder want
        // room too. conv search = the heuristic, which skips the exhaustive autotune
        // that would stall the first window. Both of these are enums, and recent
        // runtimes insist on the name and turn the old number down, so they are
        // spelled out. copy in default stream = transfers share the compute stream,
        // which is what keeps the many threads calling Run() from racing on them.
        std::string device = std::to_string(cuda_device);
        std::string limit = std::to_string(cuda_mem_bytes);
        try_cuda_option(cuda, "device_id", device);
        try_cuda_option(cuda, "gpu_mem_limit", limit);
        try_cuda_option(cuda, "arena_extend_strategy", "kSameAsRequested");
        try_cuda_option(cuda, "cudnn_conv_algo_search", "HEURISTIC");
        try_cuda_option(cuda, "do_copy_in_default_stream", "1");
        // No enable_cuda_graph here on purpose. The device fast path ping-pongs the
        // recurrent state between two buffers and binds the host zeros on the first
        // step after a reset (see device_run). CUDA Graph capture freezes the buffer
        // addresses at capture time, and the ORT C API exposes no way to repoint a
        // captured graph's I/O between steps, so enabling it would replay stale state
        // or fault. For a launch-bound model this small the real lever is batch width,
        // which silero_preferred_lanes raises on the GPU instead.
        //
        // No cudnn_conv_use_max_workspace either. Silero's feature front end is an
        // STFT and its body is cuBLAS LSTMs, not convolutions, so the cuDNN conv
        // workspace knob buys nothing here while eating into gpu_mem_limit; leaving
        // cuDNN's default in place is both faster and safer on a shared card.

        bool good = ok(ort->SessionOptionsAppendExecutionProvider_CUDA_V2(opts, cuda));
        ort->ReleaseCUDAProviderOptions(cuda);
        return good;
    }

    if (provider_is(provider, "TensorRT")) {
        OrtTensorRTProviderOptionsV2* trt = nullptr;
        if (!ok(ort->CreateTensorRTProviderOptions(&trt)) || !trt) return false;

        // TensorRT compiles the graph for this exact GPU while the session opens,
        // which for a model this small costs more than the inference of many films
        // put together, so the engine is cached and reused from then on. The cache
        // is keyed to the model, the TensorRT build and the GPU, so an entry that
        // has gone stale under one of them is simply rebuilt. Asking for this
        // provider is also what first loads the TensorRT runtime library, so a
        // machine without one reports the missing nvinfer here and falls through.
        std::string device = std::to_string(cuda_device);
        std::string cache = trt_cache_dir(model);
        try_trt_option(trt, "device_id", device);
        try_trt_option(trt, "trt_engine_cache_enable", "1");
        try_trt_option(trt, "trt_engine_cache_path", cache);
        try_trt_option(trt, "trt_timing_cache_enable", "1");

        bool good = ok(ort->SessionOptionsAppendExecutionProvider_TensorRT_V2(opts, trt));
        ort->ReleaseTensorRTProviderOptions(trt);
        return good;
    }

    return ok(ort->SessionOptionsAppendExecutionProvider(opts, provider.c_str(), nullptr, nullptr, 0));
}

// Builds a live session on one provider, leaving it in `session` and its options in
// `options`. On any failure both are left null and the caller tries the next one.
static bool build_session(const std::filesystem::path& model, const std::string& provider) {
    if (!ok(ort->CreateSessionOptions(&options))) return false;
    // These three answer with an OrtStatus (ORT_API2_STATUS, which recent headers
    // mark warn_unused_result). A refusal here is never fatal - the session simply
    // keeps the default threading and optimization level - so ok() logs it and we
    // carry on, exactly as dropping the status did before, only noisily when wrong.
    ok(ort->SetIntraOpNumThreads(options, 1));
    ok(ort->SetInterOpNumThreads(options, 1));
    ok(ort->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL));

    if (!append_provider(options, provider, model)) {
        ort->ReleaseSessionOptions(options);
        options = nullptr;
        return false;
    }

    if (!ok(ort->CreateSession(env, model.c_str(), options, &session)) || !session) {
        session = nullptr;
        return false;
    }
    return true;
}

static void drop_session() {
    if (session) { ort->ReleaseSession(session); session = nullptr; }
    if (options) { ort->ReleaseSessionOptions(options); options = nullptr; }
}

// A single throwaway window proves the session can actually run, not just load. A GPU
// export whose dynamic batch axis the provider rejects fails here, so we fall back
// before the crew threads ever see it.
static bool validate_session() {
    Lanes probe;
    return silero_begin(probe, 1) > 0;
}

// LAPSE_VAD_BENCH=<windows> runs a synthetic stream and reports throughput, so GPU and
// CPU can be compared with the same binary. Goes to stderr, stdout stays clean. It
// benches the lane width the engine will actually run with, because for this model the
// per-call overhead is most of the cost and a bench pinned to one width measures a
// configuration nobody uses.
static void maybe_benchmark() {
    const char* flag = std::getenv("LAPSE_VAD_BENCH");
    if (!flag) return;
    int iterations = atoi(flag);
    if (iterations < 8) iterations = 8;
    if (iterations > 100000) iterations = 100000;

    Lanes run;
    if (silero_begin(run, silero_preferred_lanes()) <= 0) return;

    const int step = silero_step_width();
    std::vector<float> feed((size_t)run.count * step * SILERO_WINDOW, 0.01f);
    std::vector<float> got((size_t)run.count * step, 0.0f);

    auto start = std::chrono::steady_clock::now();
    bool alive = true;
    for (int i = 0; i < iterations && alive; i++)
        alive = silero_step(run, feed.data(), got.data());
    auto stop = std::chrono::steady_clock::now();
    if (!alive) return;

    double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    double windows = (double)iterations * run.count * step;
    double per_second = ms > 0.0 ? windows / (ms / 1000.0) : 0.0;
    say() << "Silero VAD bench (" << chosen_provider << "): " << iterations << " x "
          << run.count << " lanes x " << step << " windows = " << (long long)windows << " windows in " << ms
          << "ms, " << (windows > 0 ? ms / windows : 0.0) << "ms each, "
          << per_second << " windows/s\n";
}

// Bolts the device-side plumbing onto a run whose probe step already passed.
// Locked staging, the tensors that wrap it, and on CUDA one IoBinding with the
// state slots ping-ponging between two arena buffers, are all built once here
// because nothing about a run's shapes changes after this point. LAPSE_VAD_NOFAST
// skips the whole thing and leaves the run on the plain path, which is the A/B
// baseline for LAPSE_VAD_BENCH. Any miss - locked memory denied, an old runtime
// refusing a call, a provider that will not take a host output binding - drops
// the attempt whole and the run keeps working the slow way.
static FastPath* try_fast_path(Lanes& run) {
    if (std::getenv("LAPSE_VAD_NOFAST")) return nullptr;

    // Ask the model what it is before laying out a single byte: the shapes,
    // the dtypes and the score's rank all come off the session, not out of a
    // comment about which export this engine last met. A model that will not
    // declare itself stays on the plain path where the runtime allocates its
    // own everything.
    ModelProbe probe;
    if (!model_probe(run.count, probe)) return nullptr;

    FastPath* fast = new (std::nothrow) FastPath();
    if (!fast) return nullptr;

    fast->window_bytes = probe.window_elems * sizeof(float);
    fast->score_bytes = probe.score_elems * sizeof(float);
    fast->rate_bytes = sizeof(int64_t);   // the probe checked: exactly one int64
    fast->state_bytes = probe.state_elems * sizeof(float);

    // run.state is sized by the silero convention in silero_begin; the fast
    // path only takes a run over when the model's state is that size byte for
    // byte, so every wrap of run.state and every copy into and out of it is
    // exact by construction.
    if (fast->state_bytes != run.state.size() * sizeof(float)) {
        fast_teardown(fast);
        return nullptr;
    }

    fast->pin_window = (float*)pin_alloc(fast->window_bytes);
    fast->pin_score = (float*)pin_alloc(fast->score_bytes);
    fast->pin_rate = (int64_t*)pin_alloc(fast->rate_bytes);
    fast->pin_state_zero = (float*)pin_alloc(fast->state_bytes);
    if (!fast->pin_window || !fast->pin_score || !fast->pin_rate || !fast->pin_state_zero) {
        fast_teardown(fast);
        return nullptr;
    }
    std::memset(fast->pin_state_zero, 0, fast->state_bytes);
    fast->pin_rate[0] = SILERO_RATE;

    bool built =
        ok(ort->CreateTensorWithDataAsOrtValue(memory, fast->pin_window, fast->window_bytes, probe.window_dims, probe.window_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_window)) &&
        ok(ort->CreateTensorWithDataAsOrtValue(memory, fast->pin_rate, fast->rate_bytes, probe.rate_dims, probe.rate_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &fast->v_sr)) &&
        ok(ort->CreateTensorWithDataAsOrtValue(memory, fast->pin_state_zero, fast->state_bytes, probe.state_dims, probe.state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_state_zero));

    if (built && provider_is(chosen_provider, "CUDA")) {
        // The recurrent state is the one tensor of this model that only the
        // model itself ever reads. On CUDA it does not need the host at all:
        // two buffers in the provider's own arena swap roles every step, which
        // takes both its transfers off the critical path. The score crosses to
        // the host through one IoBinding, and IoBinding holds a bound value to
        // the rank the model declared for that output: [lanes] on the old
        // exports, [lanes, 1] on the v6 ones. Both are lanes floats end to
        // end, so the staging stays the same and only the label follows the
        // declaration the probe read off the session.
        built = built &&
            ok(ort->CreateMemoryInfo("Cuda", OrtDeviceAllocator, cuda_device, OrtMemTypeDefault, &fast->cuda_mem)) &&
            ok(ort->CreateAllocator(session, fast->cuda_mem, &fast->gpu_alloc));
        for (int i = 0; built && i < 2; i++) {
            fast->dev_state[i] = (float*)fast->gpu_alloc->Alloc(fast->gpu_alloc, fast->state_bytes);
            built = fast->dev_state[i] != nullptr &&
                ok(ort->CreateTensorWithDataAsOrtValue(fast->cuda_mem, fast->dev_state[i], fast->state_bytes, probe.state_dims, probe.state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_dev_state[i]));
        }
        built = built &&
            ok(ort->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &fast->host_mem)) &&
            ok(ort->CreateTensorWithDataAsOrtValue(fast->host_mem, fast->pin_score, fast->score_bytes, probe.score_dims, probe.score_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_score)) &&
            ok(ort->CreateIoBinding(session, &fast->binding)) &&
            ok(ort->BindInput(fast->binding, input_names[0], fast->v_window)) &&
            ok(ort->BindInput(fast->binding, input_names[2], fast->v_sr)) &&
            ok(ort->BindOutput(fast->binding, output_names[0], fast->v_score));
        if (built) fast->device_state = true;
    }

    if (built && !fast->device_state) {
        // Not CUDA (CPU or DirectML): the state stays where it lives, but it is
        // wrapped once now so the per-step tensor churn goes away anyway.
        built = ok(ort->CreateTensorWithDataAsOrtValue(memory, run.state.data(), fast->state_bytes, probe.state_dims, probe.state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_state_host));
    }

    if (!built) {
        fast_teardown(fast);
        return nullptr;
    }
    run.fast = fast;
    run.fast_kind = FAST_STOCK;
    return fast;
}

// CUDA with the state on the device: bind the two state slots for this step and
// let one RunWithBinding carry everything else. The slot written last step is
// the one read this step, and the one read this step is never the one being
// written, so the buffers never alias even mid-flight on the stream. pending_zero
// covers the first step after every reset, where the zeros are still on the host.
static bool device_run(Lanes& run, FastPath& fast, float* out) {
    int next = 1 - fast.active;
    OrtValue* incoming = fast.pending_zero ? fast.v_state_zero : fast.v_dev_state[fast.active];
    if (!ok(ort->BindInput(fast.binding, input_names[1], incoming))) return false;
    if (!ok(ort->BindOutput(fast.binding, output_names[1], fast.v_dev_state[next]))) return false;
    if (!ok(ort->RunWithBinding(session, nullptr, fast.binding))) return false;
    for (int l = 0; l < run.count; l++) out[l] = fast.pin_score[l];
    fast.active = next;
    fast.pending_zero = false;
    return true;
}

// Host state, persistent tensors: every provider that is not CUDA lands here,
// and it is where CUDA would land if the device setup above had declined. Same
// Run as the plain path, just with the input side wrapped once for good.
static bool host_run(Lanes& run, FastPath& fast, float* out) {
    OrtValue* in[3] = {fast.v_window, fast.v_state_host, fast.v_sr};
    OrtValue* res[2] = {nullptr, nullptr};
    bool done = false;
    if (ok(ort->Run(session, nullptr, input_names, in, 3, output_names, 2, res))) {
        float* score = nullptr;
        float* next = nullptr;
        if (ok(ort->GetTensorMutableData(res[0], (void**)&score)) &&
            ok(ort->GetTensorMutableData(res[1], (void**)&next))) {
            for (int l = 0; l < run.count; l++) out[l] = score[l];
            std::copy(next, next + run.state.size(), run.state.begin());
            done = true;
        }
    }
    for (OrtValue* value : res) if (value) ort->ReleaseValue(value);
    return done;
}

int silero_step_width() {
    return contract == SILERO_CONTRACT_SEQUENCE && seq_windows > 0 ? seq_windows : 1;
}

SileroContract silero_contract() {
    return contract;
}

// The sequence contract's answer to FastPath: everything one run touches every
// step, parked once instead of rebuilt per call. The shape of the problem is
// different enough that sharing the stock struct would only hide the
// differences - here the staging is a whole K-row block of one lane rather than
// one row per lane, and the recurrent state is not one buffer advanced in
// lockstep but one pair per lane, each pair a private history the model carries
// forward through every call this lane makes. On CUDA that per-lane pair
// ping-pongs between two arena buffers exactly like the stock device state
// does, so a lane's history never leaves the card either; the only transfers
// left per step are the K-row block in and the K scores out.
struct SeqFastPath {
    bool device_state = false;         // CUDA only: each lane's state stays on the device
    int active = 0;                    // which device slot the last step wrote, per-lane parity
    bool pending_zero = true;          // the next step reads the zero state

    size_t block_bytes = 0;            // one lane's K rows of context plus samples
    size_t score_bytes = 0;            // one lane's K scores
    size_t state_bytes = 0;            // one half of one lane's state, as the session declared it
    size_t pair_bytes = 0;             // h and c together, as the host buffers hold them

    float* pin_block = nullptr;        // page-locked input staging, one lane's K rows
    float* pin_score = nullptr;        // page-locked score output, K of them
    float* pin_state_zero = nullptr;   // page-locked zeros for both halves, first step after a reset

    OrtValue* v_block = nullptr;       // wraps pin_block, rebound per lane at bind time
    OrtValue* v_score = nullptr;       // wraps pin_score, K scores wide

    // Per lane the state is one h and one c; off the device they sit back to
    // back in run.state exactly as the plain path leaves them there, wrapped
    // once as v_h_host / v_c_host so silero_begin's reset reaches them too. On
    // CUDA each half has two device slots and a step ping-pongs between them,
    // so the buffer a step reads is never the one it writes even mid-flight on
    // the stream. Device vectors are indexed lane * 2 + slot, slot the active
    // parity.
    std::vector<OrtValue*> v_state_zero;      // per lane * 2: zero h, zero c (device first step)
    std::vector<OrtValue*> v_h_host;          // per lane: wraps this lane's h in run.state
    std::vector<OrtValue*> v_c_host;          // per lane: wraps this lane's c in run.state
    std::vector<OrtValue*> v_h_dev;           // per lane * 2: device h slots
    std::vector<OrtValue*> v_c_dev;           // per lane * 2: device c slots
    std::vector<float*> dev_h;                // per lane * 2: raw device h, for freeing
    std::vector<float*> dev_c;                // per lane * 2: raw device c, for freeing

    OrtIoBinding* binding = nullptr;
    OrtMemoryInfo* cuda_mem = nullptr;
    OrtMemoryInfo* host_mem = nullptr;
    OrtAllocator* gpu_alloc = nullptr;
};

static void seq_teardown(SeqFastPath* fast) {
    if (!fast) return;
    if (ort) {
        if (fast->binding) ort->ReleaseIoBinding(fast->binding);
        if (fast->v_block) ort->ReleaseValue(fast->v_block);
        if (fast->v_score) ort->ReleaseValue(fast->v_score);
        for (OrtValue* value : fast->v_state_zero) if (value) ort->ReleaseValue(value);
        for (OrtValue* value : fast->v_h_host) if (value) ort->ReleaseValue(value);
        for (OrtValue* value : fast->v_c_host) if (value) ort->ReleaseValue(value);
        for (OrtValue* value : fast->v_h_dev) if (value) ort->ReleaseValue(value);
        for (OrtValue* value : fast->v_c_dev) if (value) ort->ReleaseValue(value);
        if (fast->gpu_alloc) {
            for (float* block : fast->dev_h)
                // AllocatorFree answers with an OrtStatus; on a teardown there is
                // nothing left to do about a refusal but say so, which ok() does.
                if (block) ok(ort->AllocatorFree(fast->gpu_alloc, block));
            for (float* block : fast->dev_c)
                if (block) ok(ort->AllocatorFree(fast->gpu_alloc, block));
            ort->ReleaseAllocator(fast->gpu_alloc);
        }
        if (fast->host_mem) ort->ReleaseMemoryInfo(fast->host_mem);
        if (fast->cuda_mem) ort->ReleaseMemoryInfo(fast->cuda_mem);
    }
    pin_free(fast->pin_block, fast->block_bytes);
    pin_free(fast->pin_score, fast->score_bytes);
    pin_free(fast->pin_state_zero, fast->pair_bytes);
    delete fast;
}

// Bolts the sequence plumbing onto a run whose probe step already passed, for
// the K-windows-per-call export. Mirrors what try_fast_path is for the stock
// contract, with the shape of the work swapped: one K-row staging block that
// gets rebound to each lane in turn rather than one row per lane, and a state
// pair per lane that on CUDA never comes back to the host. The binding's I/O
// names are bound once; only the three tensors whose contents differ per lane
// - the block, the state pair, and the score landing pad - get rebound per
// lane. Any miss drops the attempt whole and the run keeps working the slow
// way, exactly like the stock fast path.
static SeqFastPath* try_seq_fast_path(Lanes& run) {
    if (std::getenv("LAPSE_VAD_NOFAST")) return nullptr;
    if (seq_windows < 1 || seq_state_elems < 1) return nullptr;

    SeqFastPath* fast = new (std::nothrow) SeqFastPath();
    if (!fast) return nullptr;

    const int K = seq_windows;
    const int width = SILERO_CONTEXT + SILERO_WINDOW;
    fast->block_bytes = (size_t)K * width * sizeof(float);
    fast->score_bytes = (size_t)K * sizeof(float);
    fast->state_bytes = seq_state_elems * sizeof(float);
    fast->pair_bytes = fast->state_bytes * 2;

    // run.state is sized by silero_begin as two halves per lane; the fast path
    // only takes a run over when the session's state is that size byte for
    // byte, so every wrap and copy is exact by construction.
    if (fast->pair_bytes * (size_t)run.count != run.state.size() * sizeof(float)) {
        seq_teardown(fast);
        return nullptr;
    }

    fast->pin_block = (float*)pin_alloc(fast->block_bytes);
    fast->pin_score = (float*)pin_alloc(fast->score_bytes);
    fast->pin_state_zero = (float*)pin_alloc(fast->pair_bytes);
    if (!fast->pin_block || !fast->pin_score || !fast->pin_state_zero) {
        seq_teardown(fast);
        return nullptr;
    }
    std::memset(fast->pin_state_zero, 0, fast->pair_bytes);

    fast->v_state_zero.assign((size_t)run.count * 2, nullptr);
    fast->v_h_host.assign(run.count, nullptr);
    fast->v_c_host.assign(run.count, nullptr);
    fast->v_h_dev.assign((size_t)run.count * 2, nullptr);
    fast->v_c_dev.assign((size_t)run.count * 2, nullptr);
    fast->dev_h.assign((size_t)run.count * 2, nullptr);
    fast->dev_c.assign((size_t)run.count * 2, nullptr);

    bool built = ok(ort->CreateTensorWithDataAsOrtValue(memory, fast->pin_block, fast->block_bytes,
        seq_window_dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_block));

    if (built && provider_is(chosen_provider, "CUDA")) {
        // The per-lane state is the one tensor of this model that only the
        // model itself ever reads. On CUDA it does not need the host at all:
        // two buffers in the provider's own arena swap roles every step, which
        // takes both its transfers off the critical path. h and c ping-pong
        // together, so a step reads h[i] and c[i] and writes h[i]'s and c[i]'s
        // partners, which never alias even mid-flight on the stream.
        built = built &&
            ok(ort->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &fast->host_mem)) &&
            ok(ort->CreateMemoryInfo("Cuda", OrtDeviceAllocator, cuda_device, OrtMemTypeDefault, &fast->cuda_mem)) &&
            ok(ort->CreateAllocator(session, fast->cuda_mem, &fast->gpu_alloc)) &&
            ok(ort->CreateTensorWithDataAsOrtValue(fast->host_mem, fast->pin_score, fast->score_bytes,
                seq_score_dims, seq_score_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_score)) &&
            ok(ort->CreateIoBinding(session, &fast->binding)) &&
            ok(ort->BindInput(fast->binding, seq_input_names[0], fast->v_block));

        // Two device slots per half per lane: a step reads the slot the last
        // step wrote and writes the other, exactly the ping-pong the stock
        // device state runs, with one pair of buffers per lane.
        for (int l = 0; built && l < run.count; l++) {
            for (int slot = 0; built && slot < 2; slot++) {
                size_t idx = (size_t)l * 2 + slot;
                fast->dev_h[idx] = (float*)fast->gpu_alloc->Alloc(fast->gpu_alloc, fast->state_bytes);
                fast->dev_c[idx] = (float*)fast->gpu_alloc->Alloc(fast->gpu_alloc, fast->state_bytes);
                built = fast->dev_h[idx] != nullptr && fast->dev_c[idx] != nullptr &&
                    ok(ort->CreateTensorWithDataAsOrtValue(fast->cuda_mem, fast->dev_h[idx], fast->state_bytes,
                        seq_state_dims, seq_state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_h_dev[idx])) &&
                    ok(ort->CreateTensorWithDataAsOrtValue(fast->cuda_mem, fast->dev_c[idx], fast->state_bytes,
                        seq_state_dims, seq_state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_c_dev[idx]));
            }
        }
        if (built) fast->device_state = true;
    }

    if (built) {
        for (int l = 0; built && l < run.count; l++) {
            // On CUDA the first step after a reset reads the pinned zeros, h and
            // c as two values over the same zeros because the model takes them
            // as two inputs. Off the device the pair lives in run.state, where
            // the plain path keeps it too, and is wrapped in place one value per
            // half, so no step rebuilds them and silero_begin's reset of that
            // vector reaches this path as well.
            if (fast->device_state) {
                built =
                    ok(ort->CreateTensorWithDataAsOrtValue(memory, fast->pin_state_zero, fast->state_bytes,
                        seq_state_dims, seq_state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_state_zero[l * 2])) &&
                    ok(ort->CreateTensorWithDataAsOrtValue(memory, fast->pin_state_zero + seq_state_elems, fast->state_bytes,
                        seq_state_dims, seq_state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_state_zero[l * 2 + 1]));
            } else {
                float* half = run.state.data() + (size_t)l * seq_state_elems * 2;
                built =
                    ok(ort->CreateTensorWithDataAsOrtValue(memory, half, fast->state_bytes,
                        seq_state_dims, seq_state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_h_host[l])) &&
                    ok(ort->CreateTensorWithDataAsOrtValue(memory, half + seq_state_elems, fast->state_bytes,
                        seq_state_dims, seq_state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_c_host[l]));
            }
        }
    }

    if (!built) {
        seq_teardown(fast);
        return nullptr;
    }
    run.fast = fast;
    run.fast_kind = FAST_SEQUENCE;
    return fast;
}

// One call, one lane: the model runs the K rows of this lane's block and hands
// back K scores and the next state pair. On CUDA the state pair ping-pongs
// through the device slots behind the binding; everywhere else the pair lives
// in the host buffer and the run reads and writes it in place, which is the
// same Run the plain path makes, just with the input side wrapped once for good.
static bool sequence_run(Lanes& run, SeqFastPath& fast, int lane, float* out) {
    const int K = seq_windows;
    float* half = run.state.data() + (size_t)lane * seq_state_elems * 2;

    bool done = false;
    if (fast.device_state) {
        const size_t read_slot = (size_t)lane * 2 + (size_t)fast.active;
        const size_t write_slot = (size_t)lane * 2 + (size_t)(1 - fast.active);
        OrtValue* incoming_h = fast.pending_zero ? fast.v_state_zero[lane * 2] : fast.v_h_dev[read_slot];
        OrtValue* incoming_c = fast.pending_zero ? fast.v_state_zero[lane * 2 + 1] : fast.v_c_dev[read_slot];
        if (ok(ort->BindInput(fast.binding, seq_input_names[0], fast.v_block)) &&
            ok(ort->BindInput(fast.binding, seq_input_names[1], incoming_h)) &&
            ok(ort->BindInput(fast.binding, seq_input_names[2], incoming_c)) &&
            ok(ort->BindOutput(fast.binding, seq_output_names[0], fast.v_score)) &&
            ok(ort->BindOutput(fast.binding, seq_output_names[1], fast.v_h_dev[write_slot])) &&
            ok(ort->BindOutput(fast.binding, seq_output_names[2], fast.v_c_dev[write_slot])) &&
            ok(ort->RunWithBinding(session, nullptr, fast.binding))) {
            float* score = nullptr;
            if (ok(ort->GetTensorMutableData(fast.v_score, (void**)&score))) {
                std::copy(score, score + K, out + (size_t)lane * K);
                done = true;
            }
        }
    } else {
        // The block, h and c are all wrapped once and reused; only the three
        // outputs are fresh per call, because the runtime allocates those. The
        // state copies land in run.state through the same pointers the input
        // values wrap, so the next call reads them without any rebinding.
        OrtValue* in[3] = {fast.v_block, fast.v_h_host[lane], fast.v_c_host[lane]};
        OrtValue* res[3] = {nullptr, nullptr, nullptr};
        if (ok(ort->Run(session, nullptr, seq_input_names, in, 3, seq_output_names, 3, res))) {
            float* score = nullptr;
            float* hn = nullptr;
            float* cn = nullptr;
            if (ok(ort->GetTensorMutableData(res[0], (void**)&score)) &&
                ok(ort->GetTensorMutableData(res[1], (void**)&hn)) &&
                ok(ort->GetTensorMutableData(res[2], (void**)&cn))) {
                std::copy(score, score + K, out + (size_t)lane * K);
                std::copy(hn, hn + seq_state_elems, half);
                std::copy(cn, cn + seq_state_elems, half + seq_state_elems);
                done = true;
            }
        }
        for (OrtValue* value : res) if (value) ort->ReleaseValue(value);
    }
    return done;
}

// The plain sequence path: the runtime allocates its own everything and every
// tensor is wrapped fresh per lane. The K rows of one lane are assembled into
// a scratch block first, because the context chaining below reads the tail of
// the row before it, and that row has to be the one that was actually fed.
static bool sequence_step_plain(Lanes& run, const float* pcm, float* out) {
    const int K = seq_windows;
    const int width = SILERO_CONTEXT + SILERO_WINDOW;
    std::vector<float> block((size_t)K * width);
    int64_t block_shape[2] = {K, width};

    for (int l = 0; l < run.count; l++) {
        float* half = run.state.data() + (size_t)l * seq_state_elems * 2;
        float* lane_context = run.context.data() + (size_t)l * SILERO_CONTEXT;

        for (int k = 0; k < K; k++) {
            float* row = block.data() + (size_t)k * width;
            const float* src = lane_context;
            if (k > 0) {
                // the row before it becomes this row's context: exactly what the
                // reference frame_blocks() does with contexts[1:]
                src = row - width + SILERO_WINDOW;
            }
            std::copy(src, src + SILERO_CONTEXT, row);
            std::copy(pcm + ((size_t)l * K + k) * SILERO_WINDOW,
                      pcm + ((size_t)l * K + k + 1) * SILERO_WINDOW, row + SILERO_CONTEXT);
        }

        OrtValue* in[3] = {nullptr, nullptr, nullptr};
        OrtValue* res[3] = {nullptr, nullptr, nullptr};
        bool ready =
            ok(ort->CreateTensorWithDataAsOrtValue(memory, block.data(), block.size() * sizeof(float),
                block_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0])) &&
            ok(ort->CreateTensorWithDataAsOrtValue(memory, half, seq_state_elems * sizeof(float),
                seq_state_dims, seq_state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1])) &&
            ok(ort->CreateTensorWithDataAsOrtValue(memory, half + seq_state_elems, seq_state_elems * sizeof(float),
                seq_state_dims, seq_state_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[2]));

        bool done = false;
        if (ready && ok(ort->Run(session, nullptr, seq_input_names, in, 3, seq_output_names, 3, res))) {
            float* score = nullptr;
            float* hn = nullptr;
            float* cn = nullptr;
            if (ok(ort->GetTensorMutableData(res[0], (void**)&score)) &&
                ok(ort->GetTensorMutableData(res[1], (void**)&hn)) &&
                ok(ort->GetTensorMutableData(res[2], (void**)&cn))) {
                std::copy(score, score + K, out + (size_t)l * K);
                std::copy(hn, hn + seq_state_elems, half);
                std::copy(cn, cn + seq_state_elems, half + seq_state_elems);
                done = true;
            }
        }
        for (OrtValue* value : res) if (value) ort->ReleaseValue(value);
        for (OrtValue* value : in) if (value) ort->ReleaseValue(value);
        if (!done) return false;

        // the tail of the last row fed becomes the first row's context next call
        std::copy(block.data() + (size_t)(K - 1) * width + SILERO_WINDOW,
                  block.data() + (size_t)K * width, lane_context);
    }
    return true;
}

// The sequence contract's step: assemble each lane's K rows, run the model once
// per lane, and write back the context that chains into the next call. The
// caller has laid out its audio and score arrays at K windows per lane, in time
// order, which is the whole difference between this and the stock contract.
static bool sequence_step(Lanes& run, const float* pcm, float* out) {
    SeqFastPath* fast = static_cast<SeqFastPath*>(run.fast);
    if (!fast) return sequence_step_plain(run, pcm, out);

    const int K = seq_windows;
    const int width = SILERO_CONTEXT + SILERO_WINDOW;

    for (int l = 0; l < run.count; l++) {
        float* row = fast->pin_block;
        const float* src = run.context.data() + (size_t)l * SILERO_CONTEXT;
        for (int k = 0; k < K; k++) {
            if (k > 0) src = row - width + SILERO_WINDOW;
            std::copy(src, src + SILERO_CONTEXT, row);
            std::copy(pcm + ((size_t)l * K + k) * SILERO_WINDOW,
                      pcm + ((size_t)l * K + k + 1) * SILERO_WINDOW, row + SILERO_CONTEXT);
            row += width;
        }
        if (!sequence_run(run, *fast, l, out)) return false;
        // the tail of this lane's last row becomes the head of its next call;
        // it must be saved now, before the next lane overwrites the staging block
        const float* last = row - width;
        std::copy(last + SILERO_WINDOW, last + width,
                  run.context.data() + (size_t)l * SILERO_CONTEXT);
    }
    fast->active = 1 - fast->active;
    fast->pending_zero = false;
    return true;
}

bool silero_open() {
    std::lock_guard<std::mutex> guard(open_mutex);
    if (session) return true;
    if (tried) return false;
    tried = true;

    library = load_runtime();
    if (!library) {
        say() << "Silero VAD: no onnxruntime library found, using libfvad\n";
        return false;
    }

    typedef const OrtApiBase* (*ApiBase)(void);
    ApiBase base = (ApiBase)library_symbol(library, "OrtGetApiBase");
    if (!base) {
        say() << "Silero VAD: onnxruntime has no entry point, using libfvad\n";
        return false;
    }

    const OrtApiBase* api = base();
    for (int version = ORT_API_VERSION; version >= 11 && !ort; version--)
        ort = api->GetApi(version);
    if (!ort) {
        say() << "Silero VAD: onnxruntime is too old, using libfvad\n";
        return false;
    }

    // The runtime is whatever the operator dropped beside the binary, and the GPU
    // provider is whatever that build was compiled against. A CUDA EP built before a
    // GPU's compute capability existed loads fine and lists itself in the providers,
    // then fails at the first kernel - which would otherwise look like "the GPU is
    // slow" while the CPU does all the work. The version string is what tells the
    // two apart from the log alone: sm_120 class cards need an onnxruntime built
    // against CUDA 12.8 or newer. GetVersionString has been in OrtApiBase since the
    // first C API, so an older runtime still answers it.
    const char* version_text = api->GetVersionString ? api->GetVersionString() : nullptr;
    say() << "Silero VAD: onnxruntime " << (version_text ? version_text : "unknown")
          << ", compiled against C API " << ORT_API_VERSION << "\n";

    std::filesystem::path model = find_model();
    if (model.empty()) {
        say() << "Silero VAD: no silero_vad.onnx found, using libfvad\n";
        return false;
    }

    const char* device_env = std::getenv("LAPSE_CUDA_DEVICE");
    if (device_env) cuda_device = atoi(device_env);
    const char* mem_env = std::getenv("LAPSE_CUDA_MEM_MB");
    if (mem_env) {
        int mb = atoi(mem_env);
        if (mb > 0) cuda_mem_bytes = (long long)mb * 1024 * 1024;
    }

    if (!ok(ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "lapse", &env))) return false;

    // The CPU memory info describes the host buffers we feed tensors from; the CUDA
    // and DirectML providers copy to and from the device themselves, so this is built
    // once regardless of which provider wins.
    if (!ok(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory))) {
        memory = nullptr;
        return false;
    }

    // Prefer a GPU execution provider when this onnxruntime build ships one, but let
    // LAPSE_VAD_CPU pin it back to the CPU. CUDA first, DirectML only on Windows where
    // it is a real provider, and the CPU provider always last as the sure thing.
    //
    // TensorRT stays opt-in behind LAPSE_VAD_TRT even when the runtime advertises it.
    // For a 512-sample stateful update its compiled engine is no faster than CUDA,
    // while building it costs seconds on first use and the TensorRT runtime is many
    // times the size of this program. When the flag is set it still goes in front of
    // CUDA, and the probe below catches an engine that fails to build or run so the
    // chain keeps falling through to it.
    std::vector<std::string> available = list_providers();
    std::vector<std::string> order;
    if (!std::getenv("LAPSE_VAD_CPU")) {
        if (std::getenv("LAPSE_VAD_TRT")) {
            std::string trt = provider_name(available, "TensorRT");
            if (!trt.empty()) order.push_back(trt);
        }
        std::string cuda = provider_name(available, "CUDA");
        if (!cuda.empty()) order.push_back(cuda);
#ifdef _WIN32
        std::string dml = provider_name(available, "Dml");
        if (!dml.empty()) order.push_back(dml);
#endif
    }
    order.push_back("");

    // One line naming everything this onnxruntime offers. When the GPU providers are
    // missing from it the reason is a DLL next to onnxruntime that failed to load,
    // which is the single most common cause of a quiet fall back to the CPU, so it is
    // worth having in the log even when nothing is wrong.
    std::string reported;
    for (const std::string& name : available) {
        if (!reported.empty()) reported += ", ";
        reported += name;
    }
    say() << "Silero VAD: onnxruntime offers " << (reported.empty() ? "nothing" : reported) << "\n";

    // The contract is read between building and validating, because it is the
    // answer that picks which code runs the first throwaway window. It is also a
    // property of the file, not of the provider that loaded it: a model that
    // answers to neither contract is rejected here instead of being fed stock
    // names and coming back as "Invalid input name: state" from every provider
    // in turn, which is what made an unsupported model read like a GPU fault.
    bool shape_rejected = false;
    for (const std::string& provider : order) {
        if (!build_session(model, provider)) {
            drop_session();
            continue;
        }
        if (!read_contract()) {
            shape_rejected = true;
            say() << "Silero VAD: " << model.string()
                  << " matches neither the stock contract (input/state/sr) nor the "
                  << "sequence contract (input/h/c), using libfvad\n";
            drop_session();
            break;
        }
        if (validate_session()) {
            chosen_provider = provider;
            gpu = !provider.empty();
            break;
        }
        // A GPU provider that is advertised but cannot run a throwaway window is the
        // single most expensive quiet failure this engine can have: everything after
        // it runs on the CPU at a fraction of the rate with no other signal. Say so.
        if (!provider.empty())
            say() << "Silero VAD: " << provider << " is offered but could not run the model, "
                  << "falling through - check the onnxruntime build against this GPU\n";
        drop_session();
    }

    if (!session) {
        if (memory) { ort->ReleaseMemoryInfo(memory); memory = nullptr; }
        if (!shape_rejected)
            say() << "Silero VAD: could not load the model, using libfvad\n";
        return false;
    }

    say() << "Silero VAD: " << model.string() << " ["
          << (gpu ? chosen_provider : std::string("CPU provider")) << "]"
          << (contract == SILERO_CONTRACT_SEQUENCE
                  ? " sequence contract, " + std::to_string(seq_windows) + " windows per call"
                  : " stock contract")
          << "\n";
    maybe_benchmark();
    return true;
}

bool silero_on_gpu() {
    return gpu;
}

// How wide one run should be. On the CPU a step is mostly the model itself, so a
// modest batch is enough and a huge one only adds staging traffic. On a GPU the
// opposite holds: the whole 512-sample update is launch-overhead bound - the kernels
// themselves are over in microseconds and what is paid for is the per-step dispatch -
// so the width of a step is what turns the same launches into throughput, and the
// cost per window keeps falling well past the old fixed batch of 8.
//
// The number is a request, not a promise. The model's batch axis has to be dynamic
// for anything above one to work at all, and silero_begin proves a probe step at the
// requested width before handing the run over, falling back when a model will not
// take it. The ceiling is where the per-lane pinned staging and the recurrent state
// still fit inside the CUDA arena alongside everything else on the card.
int silero_preferred_lanes() {
    if (!session) return 1;

    const char* forced = std::getenv("LAPSE_VAD_LANES");
    if (forced) {
        int lanes = atoi(forced);
        if (lanes > 0) return lanes > 256 ? 256 : lanes;
    }
    // The sequence contract batches along time, not lanes: its state is nailed to
    // one stream, so a wider run is just more Runs in series, each with a private
    // state pair. The K windows per call are the whole win; extra lanes only add
    // staging and device buffers. LAPSE_VAD_LANES above still overrides.
    if (contract == SILERO_CONTRACT_SEQUENCE) return 1;
    return gpu ? 64 : 8;
}

int silero_begin(Lanes& run, int lanes) {
    if (!session) return 0;
    if (lanes < 1) lanes = 1;

    const int step = silero_step_width();
    run.count = lanes;
    // The sequence contract keeps one h/c pair per lane, the stock one a single
    // [2, lanes, 128] block; either way the run owns the whole state as two
    // halves per lane and every fast path wraps it in place.
    const size_t per_lane_state = contract == SILERO_CONTRACT_SEQUENCE
                                      ? (size_t)2 * seq_state_elems
                                      : (size_t)SILERO_STATE_DIRS * SILERO_STATE_HIDDEN;
    run.state.assign(per_lane_state * lanes, 0.0f);
    run.context.assign(lanes * SILERO_CONTEXT, 0.0f);
    run.window.assign(lanes * (SILERO_CONTEXT + SILERO_WINDOW), 0.0f);

    // The probe feeds the same layout the real steps will: one block per lane,
    // each block step windows deep, which is one window for stock and K for the
    // sequence contract.
    std::vector<float> probe((size_t)lanes * step * SILERO_WINDOW, 0.0f);
    std::vector<float> out((size_t)lanes * step, 0.0f);
    if (silero_step(run, probe.data(), out.data())) {
        if (contract == SILERO_CONTRACT_SEQUENCE) {
            SeqFastPath* fast = try_seq_fast_path(run);
            // A device binding that survives setup but chokes at Run time would
            // fail the film, not just slow it, so spend one throwaway step proving
            // the whole fast path before anyone commits to it.
            if (fast && fast->device_state && !silero_step(run, probe.data(), out.data())) silero_end(run);

            run.state.assign(per_lane_state * lanes, 0.0f);
            run.context.assign(lanes * SILERO_CONTEXT, 0.0f);
            // whatever the probe left in the device buffers is stale from here on;
            // the next step reads the zeros again and overwrites a slot it never
            // reads. Off the device the wrapped state was just zeroed with the
            // vector above, which is why this flag only steers the device path.
            if (run.fast) static_cast<SeqFastPath*>(run.fast)->pending_zero = true;
            return lanes;
        }

        FastPath* fast = try_fast_path(run);
        // A device binding that survives setup but chokes at Run time would
        // fail the film, not just slow it, so spend one throwaway step proving
        // the whole fast path before anyone commits to it.
        if (fast && fast->device_state && !silero_step(run, probe.data(), out.data())) silero_end(run);

        run.state.assign(per_lane_state * lanes, 0.0f);
        run.context.assign(lanes * SILERO_CONTEXT, 0.0f);
        // whatever the probe left in the device buffers is stale from here on;
        // the next step reads the zeros again and overwrites a slot it never reads
        if (run.fast) static_cast<FastPath*>(run.fast)->pending_zero = true;
        return lanes;
    }

    // an older export with the batch axis nailed to one. fall back rather than die, it is only a speed thing
    if (lanes > 1) return silero_begin(run, 1);
    run.count = 0;
    return 0;
}

bool silero_step(Lanes& run, const float* pcm, float* out) {
    if (!session || run.count < 1) return false;
    // The sequence contract takes a different tensor - input/h/c, K rows of one
    // lane per call - and owns its staging, context chaining and per-lane state;
    // everything below is the stock [lanes, 576] + sr + state contract only.
    if (contract == SILERO_CONTRACT_SEQUENCE) return sequence_step(run, pcm, out);
    int width = SILERO_CONTEXT + SILERO_WINDOW;
    std::vector<float>& state = run.state;
    std::vector<float>& context = run.context;
    std::vector<float>& window = run.window;
    int lane_count = run.count;
    FastPath* fast = static_cast<FastPath*>(run.fast);

    // The window goes through page-locked staging on the fast path and through
    // the run's own vector otherwise; assembling it is the same loop either way.
    float* stage = (fast && fast->pin_window) ? fast->pin_window : window.data();

    for (int l = 0; l < lane_count; l++) {
        float* row = stage + (size_t)l * width;
        std::copy(context.begin() + (size_t)l * SILERO_CONTEXT,
                  context.begin() + (size_t)(l + 1) * SILERO_CONTEXT, row);
        std::copy(pcm + (size_t)l * SILERO_WINDOW, pcm + (size_t)(l + 1) * SILERO_WINDOW, row + SILERO_CONTEXT);
    }

    bool done = false;
    if (fast) {
        done = fast->device_state ? device_run(run, *fast, out) : host_run(run, *fast, out);
    } else {
        int64_t input_shape[2] = {lane_count, width};
        int64_t state_shape[3] = {SILERO_STATE_DIRS, lane_count, SILERO_STATE_HIDDEN};
        int64_t scalar_shape[1] = {1};
        int64_t rate = SILERO_RATE;

        OrtValue* in[3] = {nullptr, nullptr, nullptr};
        OrtValue* res[2] = {nullptr, nullptr};

        bool ready =
            ok(ort->CreateTensorWithDataAsOrtValue(memory, stage, window.size() * sizeof(float), input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0])) &&
            ok(ort->CreateTensorWithDataAsOrtValue(memory, state.data(), state.size() * sizeof(float), state_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1])) &&
            ok(ort->CreateTensorWithDataAsOrtValue(memory, &rate, sizeof(int64_t), scalar_shape, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in[2]));

        if (ready && ok(ort->Run(session, nullptr, input_names, in, 3, output_names, 2, res))) {
            float* score = nullptr;
            float* next = nullptr;
            if (ok(ort->GetTensorMutableData(res[0], (void**)&score)) &&
                ok(ort->GetTensorMutableData(res[1], (void**)&next))) {
                for (int l = 0; l < lane_count; l++) out[l] = score[l];
                std::copy(next, next + state.size(), state.begin());
                done = true;
            }
        }

        for (OrtValue* value : res) if (value) ort->ReleaseValue(value);
        for (OrtValue* value : in) if (value) ort->ReleaseValue(value);
    }

    if (!done) return false;

    // the tail of the window just fed becomes the head of the next one
    for (int l = 0; l < lane_count; l++) {
        const float* row = stage + (size_t)l * width;
        std::copy(row + width - SILERO_CONTEXT, row + width,
                  context.begin() + (size_t)l * SILERO_CONTEXT);
    }
    return true;
}

void silero_end(Lanes& run) {
    if (!run.fast) return;
    void* fast = run.fast;
    run.fast = nullptr;
    // Which struct the run.fast pointer points at is remembered per run: the two
    // contracts build different ones, and freeing the wrong one walks garbage.
    if (run.fast_kind == FAST_SEQUENCE) seq_teardown(static_cast<SeqFastPath*>(fast));
    else fast_teardown(static_cast<FastPath*>(fast));
    run.fast_kind = FAST_NONE;
}

void silero_close() {
    std::lock_guard<std::mutex> guard(open_mutex);
    if (!ort) return;
    if (memory) ort->ReleaseMemoryInfo(memory);
    drop_session();
    if (env) ort->ReleaseEnv(env);
    memory = nullptr;
    env = nullptr;
    ort = nullptr;
    gpu = false;
    chosen_provider.clear();
    contract = SILERO_CONTRACT_UNKNOWN;
    seq_windows = 0;
    seq_state_elems = 0;

    if (library) {
#ifdef _WIN32
        FreeLibrary(library);
#else
        dlclose(library);
#endif
        library = nullptr;
    }
    tried = false;
}
