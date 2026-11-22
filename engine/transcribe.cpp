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

#include "transcribe.h"
#include "decoder.h"
#include "log.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <algorithm>
#include <cctype>
#include <iostream>

static std::string in_quotes(const std::string& s) {
    return "\"" + s + "\"";
}

// True when the model file is English-only, judged by its name ending in ".en" before the extension, ggml-base.en.bin.
static bool english_only(const std::string& model) {
    std::string stem = std::filesystem::path(model).stem().string();
    return stem.size() > 3 && stem.compare(stem.size() - 3, 3, ".en") == 0;
}

// The whole "-dtw <value>" piece of the command line, or the empty string when DTW is unset.
// whisper-cli takes a preset name here (tiny, base.en, large.v3.turbo...), we hand it over as given.
static std::string dtw_argument(const std::optional<std::string>& dtw) {
    if (!dtw || dtw->empty()) return "";
    return " -dtw " + in_quotes(*dtw);
}

// Read the whole number written after key inside json[from, upto), -1 when the key is missing or out of range.
static long json_number(const std::string& json, const std::string& key, size_t from, size_t upto) {
    size_t at = json.find(key, from);
    if (at == std::string::npos || at >= upto) return -1;
    return strtol(json.c_str() + at + key.size(), nullptr, 10);
}

std::vector<int> spoken_words(const std::string& wav, const std::string& model, std::optional<std::string> lang, std::optional<std::string> dtw) {
    const char* cli = std::getenv("LAPSE_WHISPER_CLI");
    int threads = worker_threads();
    std::string out = (std::filesystem::temp_directory_path() /
        ("lapse-words-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))).string();

    // A language the caller named wins; otherwise keep the old guess of "en" for a .en model and "auto" for the rest.
    std::string language = (lang && !lang->empty()) ? *lang : (english_only(model) ? "en" : "auto");

    // -dtw joins the command only when the caller set it, in its usual place, quoted like every other path.
    const bool dtw_on = dtw && !dtw->empty();
    std::string command = in_quotes(cli ? cli : "whisper-cli") + " -m " + in_quotes(model) + " -f " + in_quotes(wav) +
        " -ojf -of " + in_quotes(out) + dtw_argument(dtw) + " -nfa -np -t " + std::to_string(threads) +
        " -l " + in_quotes(language);
#ifdef _WIN32
    command = "\"" + command + " > NUL 2>&1\"";
#else
    command += " > /dev/null 2>&1";
#endif

    say() << "Listening for words with " << model << ", this takes a while\n";
    int status = std::system(command.c_str());

    std::ifstream in(out + ".json");
    std::stringstream all;
    all << in.rdbuf();
    std::string json = all.str();
    in.close();
    std::error_code ec;
    std::filesystem::remove(out + ".json", ec);

    if (status != 0 || json.empty()) {
        std::cerr << "whisper-cli did not run, put it on the PATH or point LAPSE_WHISPER_CLI at it\n";
        return {};
    }

    // Each token of the transcription carries its own moment: t_dtw in centiseconds, only meaningful when DTW
    // ran, and offsets.from in milliseconds, which -ojf writes out either way. We read the one that matches how
    // we ran whisper-cli, and never let a lookup cross into the token that follows.
    std::vector<int> words;
    size_t at = 0;
    while ((at = json.find("\"text\": \"", at)) != std::string::npos) {
        at += 9;
        size_t end = json.find('"', at);
        std::string text = json.substr(at, end - at);
        size_t stop = json.find("\"text\": \"", end);
        if (stop == std::string::npos) stop = json.size();

        bool word = text.rfind("[_", 0) != 0 && text.rfind("<|", 0) != 0 &&
                    std::any_of(text.begin(), text.end(), [](char c) { return isalnum((unsigned char)c); });
        if (!word) continue;

        if (dtw_on) {
            long centis = json_number(json, "\"t_dtw\": ", end, stop);
            if (centis >= 0) words.push_back((int)(centis * 10));
        } else {
            size_t offsets = json.find("\"offsets\"", end);
            if (offsets == std::string::npos || offsets >= stop) continue;
            long millis = json_number(json, "\"from\": ", offsets, stop);
            if (millis >= 0) words.push_back((int)millis);
        }
    }
    std::sort(words.begin(), words.end());
    return words;
}
