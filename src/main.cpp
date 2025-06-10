#include "common-whisper.h"
#include "whisper.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <chrono>

struct whisper_basic_params {
  std::string model_path;
  std::string audio_dir;
  std::string references_path;
  int n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
  std::string language = "auto";
};

void print_usage(int /*argc*/, char **argv,
                 const whisper_basic_params &params) {
  fprintf(stderr, "使用方法: %s [選項] model_path audio_path\n\n", argv[0]);
  fprintf(stderr, "選項:\n");
  fprintf(stderr, "  -h, --help             顯示此幫助信息並退出\n");
  fprintf(stderr, "  -t N, --threads N      [%-7d] 使用的線程數\n",
          params.n_threads);
  fprintf(stderr,
          "  -l LANG, --language LANG [%-7s] 設定語言 (auto = 自動偵測)\n",
          params.language.c_str());
  fprintf(stderr, "\n");
  fprintf(stderr, "範例:\n");
  fprintf(stderr, "  %s path/to/model.bin path/to/audio.wav\n", argv[0]);
  fprintf(stderr,
          "  %s -t 8 --language zh path/to/model.bin path/to/audio.mp3\n",
          argv[0]);
  fprintf(stderr, "  %s --translate path/to/model.bin path/to/audio.wav\n",
          argv[0]);
  fprintf(stderr, "\n");
}

bool parse_params(int argc, char **argv, whisper_basic_params &params) {
  if (argc < 3) {
    fprintf(stderr, "錯誤: 需要指定模型路徑和音頻文件路徑\n");
    return false;
  }

  // 解析可選參數
  int positional_args = 0;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      return false; // 顯示幫助並退出
    } else if (arg == "-t" || arg == "--threads") {
      if (i + 1 < argc) {
        params.n_threads = std::stoi(argv[++i]);
      } else {
        fprintf(stderr, "錯誤: --threads 需要一個數值\n");
        return false;
      }
    } else if (arg == "-l" || arg == "--language") {
      if (i + 1 < argc) {
        params.language = argv[++i];
      } else {
        fprintf(stderr, "錯誤: --language 需要指定語言\n");
        return false;
      }
    } else {
      if (positional_args == 0) {
        params.model_path = arg;
      } else if (positional_args == 1) {
        params.audio_dir = arg;
      } else if (positional_args == 2) {
        params.references_path = arg;
      }
      positional_args++;
    }
  }

  if (params.model_path.empty() || params.audio_dir.empty()) {
    fprintf(stderr, "錯誤: 需要指定模型路徑和音頻文件路徑\n");
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  whisper_basic_params params;

  if (!parse_params(argc, argv, params)) {
    print_usage(argc, argv, params);
    return 1;
  }

  // 初始化 whisper
  struct whisper_context_params cparams = whisper_context_default_params();
  struct whisper_context *ctx =
      whisper_init_from_file_with_params(params.model_path.c_str(), cparams);

  if (ctx == nullptr) {
    std::cerr << "Error: cannot init whisper model: "
              << params.model_path.c_str() << "\n";
    return 1;
  }

  fprintf(stderr, "模型: %s\n", params.model_path.c_str());
  fprintf(stderr, "系統資訊: %s\n", whisper_print_system_info());
  fprintf(stderr, "\n");

  // 準備 whisper 參數
  struct whisper_full_params wparams =
      whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  wparams.print_progress = true;
  wparams.print_special = false;
  wparams.print_realtime = false;
  wparams.print_timestamps = true;
  wparams.translate = false;
  wparams.language = params.language.c_str();
  wparams.n_threads = params.n_threads;
  wparams.offset_ms = 0;
  wparams.no_context = true;


  std::ifstream references(params.references_path);
  if(!references.is_open()) {
    std::cerr << "Error: Could not open the file: " << params.references_path << "\n";
    return 1;
  }

  std::vector<std::filesystem::directory_entry> audio_files(
      std::filesystem::directory_iterator(params.audio_dir),
      std::filesystem::directory_iterator());
  std::sort(audio_files.begin(), audio_files.end());

  for (const auto &e : audio_files) {
    const auto& audio_path = e.path();
    std::string ref;
    if(!std::getline(references, ref)) {
      std::cerr << "Error: Could not get reference sentence\n";
      return 1;
    }

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(audio_path, pcmf32, pcmf32s, false)) {
      std::cerr << "Error: cannot read audio file: " << audio_path
                << "\n";
      return 1;
    }
    
    const float duration = float(pcmf32.size()) / WHISPER_SAMPLE_RATE;
    fprintf(stderr, "\n");
    std::cerr << "Audio file: " << audio_path << "\n";
    std::cerr << "Sample number: " << pcmf32.size() << "\n";
    std::cerr << "Duration: " << duration << "\n";

    auto starts_at = std::chrono::high_resolution_clock::now();
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
      std::cerr << "Error: failed to process audio\n";
      whisper_free(ctx);
      return 1;
    }
    
    const int n_segments = whisper_full_n_segments(ctx);
    std::string output = "";
    for (int i = 0; i < n_segments; ++i) {
      const char *text = whisper_full_get_segment_text(ctx, i);
      // const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
      // const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

      // printf("[%s --> %s] %s\n", to_timestamp(t0).c_str(),
      //        to_timestamp(t1).c_str(), text);
      output += text;
    }
    auto ends_at = std::chrono::high_resolution_clock::now();
    auto elapsed = ends_at - starts_at;
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / 1000.;
    std::cout << duration << "\t" << elapsed_secs << "\t" << ref << "\t" << output << "\n";

    printf("\n");
  }

  whisper_free(ctx);
  return 0;
}
