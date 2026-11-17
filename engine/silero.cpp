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
#else
    #include <dlfcn.h>
    #include <sys/mman.h>
    typedef void* library_handle;
#endif

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cctype>
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
static int cuda_device = 0;
static long long cuda_mem_bytes = 512LL * 1024 * 1024;

// The model wants the tail of the previous chunk glued to the front of this one. Without it the numbers come back near zero for speech that is plainly there
static const int SILERO_CONTEXT = 64;

static const char* input_names[]  = {"input", "state", "sr"};
static const char* output_names[] = {"output", "stateN"};

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
// Windows locks a committed region, POSIX mlocks the mapping; both round up
// to a page so unlocking covers exactly what was locked.
static void* pin_alloc(size_t bytes) {
    size_t pages = (bytes + 4095) & ~(size_t)4095;
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
    if (posix_memalign(&p, 4096, pages) != 0) return nullptr;
    if (mlock(p, pages) != 0) {
        free(p);
        return nullptr;
    }
    return p;
#endif
}

static void pin_free(void* p, size_t bytes) {
    if (!p) return;
    size_t pages = (bytes + 4095) & ~(size_t)4095;
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
                if (block) ort->AllocatorFree(fast->gpu_alloc, block);
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

static std::filesystem::path binary_dir() {
#ifdef _WIN32
    wchar_t buffer[MAX_PATH];
    DWORD length = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    if (length > 0 && length < MAX_PATH) {
        std::filesystem::path dir = std::filesystem::path(buffer).parent_path();
        if (!dir.empty()) return dir;
    }
#else
    Dl_info info;
    if (dladdr((void*)&binary_dir, &info) && info.dli_fname) {
        std::filesystem::path dir = std::filesystem::path(info.dli_fname).parent_path();
        if (!dir.empty()) return dir;
    }
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
        ort->ReleaseAvailableProviders(raw, count);
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
    ort->SetIntraOpNumThreads(options, 1);
    ort->SetInterOpNumThreads(options, 1);
    ort->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);

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
// CPU can be compared with the same binary. Goes to stderr, stdout stays clean.
static void maybe_benchmark() {
    const char* flag = std::getenv("LAPSE_VAD_BENCH");
    if (!flag) return;
    int iterations = atoi(flag);
    if (iterations < 8) iterations = 8;
    if (iterations > 100000) iterations = 100000;

    Lanes run;
    if (silero_begin(run, 8) <= 0) return;

    std::vector<float> feed((size_t)run.count * SILERO_WINDOW, 0.01f);
    std::vector<float> got(run.count, 0.0f);

    auto start = std::chrono::steady_clock::now();
    bool alive = true;
    for (int i = 0; i < iterations && alive; i++)
        alive = silero_step(run, feed.data(), got.data());
    auto stop = std::chrono::steady_clock::now();
    if (!alive) return;

    double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    double windows = (double)iterations * run.count;
    double per_second = ms > 0.0 ? windows / (ms / 1000.0) : 0.0;
    say() << "Silero VAD bench (" << chosen_provider << "): " << iterations << " x "
          << run.count << " lanes = " << (long long)windows << " windows in " << ms
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

    int lanes = run.count;
    int width = SILERO_CONTEXT + SILERO_WINDOW;
    FastPath* fast = new (std::nothrow) FastPath();
    if (!fast) return nullptr;

    fast->window_bytes = (size_t)lanes * width * sizeof(float);
    fast->score_bytes = (size_t)lanes * sizeof(float);
    fast->rate_bytes = sizeof(int64_t);
    fast->state_bytes = (size_t)2 * lanes * 128 * sizeof(float);

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

    int64_t input_shape[2] = {lanes, width};
    int64_t state_shape[3] = {2, lanes, 128};
    int64_t score_shape[1] = {lanes};
    int64_t scalar_shape[1] = {1};

    bool built =
        ok(ort->CreateTensorWithDataAsOrtValue(memory, fast->pin_window, fast->window_bytes, input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_window)) &&
        ok(ort->CreateTensorWithDataAsOrtValue(memory, fast->pin_rate, fast->rate_bytes, scalar_shape, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &fast->v_sr)) &&
        ok(ort->CreateTensorWithDataAsOrtValue(memory, fast->pin_state_zero, fast->state_bytes, state_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_state_zero));

    if (built && provider_is(chosen_provider, "CUDA")) {
        // The recurrent state is the one tensor of this model that only the
        // model itself ever reads. On CUDA it does not need the host at all:
        // two buffers in the provider's own arena swap roles every step, which
        // takes both its transfers off the critical path.
        built = built &&
            ok(ort->CreateMemoryInfo("Cuda", OrtDeviceAllocator, cuda_device, OrtMemTypeDefault, &fast->cuda_mem)) &&
            ok(ort->CreateAllocator(session, fast->cuda_mem, &fast->gpu_alloc));
        for (int i = 0; built && i < 2; i++) {
            fast->dev_state[i] = (float*)fast->gpu_alloc->Alloc(fast->gpu_alloc, fast->state_bytes);
            built = fast->dev_state[i] != nullptr &&
                ok(ort->CreateTensorWithDataAsOrtValue(fast->cuda_mem, fast->dev_state[i], fast->state_bytes, state_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_dev_state[i]));
        }
        built = built &&
            ok(ort->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &fast->host_mem)) &&
            ok(ort->CreateTensorWithDataAsOrtValue(fast->host_mem, fast->pin_score, fast->score_bytes, score_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_score)) &&
            ok(ort->CreateIoBinding(session, &fast->binding)) &&
            ok(ort->BindInput(fast->binding, input_names[0], fast->v_window)) &&
            ok(ort->BindInput(fast->binding, input_names[2], fast->v_sr)) &&
            ok(ort->BindOutput(fast->binding, output_names[0], fast->v_score));
        if (built) fast->device_state = true;
    }

    if (built && !fast->device_state) {
        // Not CUDA (CPU or DirectML): the state stays where it lives, but it is
        // wrapped once now so the per-step tensor churn goes away anyway.
        built = ok(ort->CreateTensorWithDataAsOrtValue(memory, run.state.data(), run.state.size() * sizeof(float), state_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast->v_state_host));
    }

    if (!built) {
        fast_teardown(fast);
        return nullptr;
    }
    run.fast = fast;
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

    for (int version = ORT_API_VERSION; version >= 11 && !ort; version--)
        ort = base()->GetApi(version);
    if (!ort) {
        say() << "Silero VAD: onnxruntime is too old, using libfvad\n";
        return false;
    }

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

    for (const std::string& provider : order) {
        if (build_session(model, provider) && validate_session()) {
            chosen_provider = provider;
            gpu = !provider.empty();
            break;
        }
        drop_session();
    }

    if (!session) {
        if (memory) { ort->ReleaseMemoryInfo(memory); memory = nullptr; }
        say() << "Silero VAD: could not load the model, using libfvad\n";
        return false;
    }

    say() << "Silero VAD: " << model.string() << " ["
          << (gpu ? chosen_provider : std::string("CPU provider")) << "]\n";
    maybe_benchmark();
    return true;
}

bool silero_on_gpu() {
    return gpu;
}

int silero_begin(Lanes& run, int lanes) {
    if (!session) return 0;
    if (lanes < 1) lanes = 1;

    run.count = lanes;
    run.state.assign(2 * lanes * 128, 0.0f);
    run.context.assign(lanes * SILERO_CONTEXT, 0.0f);
    run.window.assign(lanes * (SILERO_CONTEXT + SILERO_WINDOW), 0.0f);

    std::vector<float> probe(lanes * SILERO_WINDOW, 0.0f);
    std::vector<float> out(lanes, 0.0f);
    if (silero_step(run, probe.data(), out.data())) {
        FastPath* fast = try_fast_path(run);
        // A device binding that survives setup but chokes at Run time would
        // fail the film, not just slow it, so spend one throwaway step proving
        // the whole fast path before anyone commits to it.
        if (fast && fast->device_state && !silero_step(run, probe.data(), out.data())) silero_end(run);

        run.state.assign(2 * lanes * 128, 0.0f);
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
        int64_t state_shape[3] = {2, lane_count, 128};
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
    FastPath* fast = static_cast<FastPath*>(run.fast);
    run.fast = nullptr;
    fast_teardown(fast);
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
