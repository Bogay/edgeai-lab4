#include "common-whisper.h"
#include "whisper.h"

#include <cstdio>
#include <string>
#include <vector>
#include <thread>
#include <fstream>

// 基本的命令行參數結構
struct whisper_basic_params {
    std::string model_path;   // 模型文件路徑
    std::string dataset_path;   // 資料集路徑
    int n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency()); // 使用的線程數
    bool translate = false;   // 是否翻譯
    std::string language = "auto"; // 使用的語言，auto為自動偵測
};

// 顯示使用說明
void print_usage(int /*argc*/, char **argv, const whisper_basic_params &params) {
    fprintf(stderr, "使用方法: %s [選項] model_path dataset_path\n\n", argv[0]);
    fprintf(stderr, "選項:\n");
    fprintf(stderr, "  -h, --help             顯示此幫助信息並退出\n");
    fprintf(stderr, "  -t N, --threads N      [%-7d] 使用的線程數\n", params.n_threads);
    fprintf(stderr, "  -l LANG, --language LANG [%-7s] 設定語言 (auto = 自動偵測)\n", params.language.c_str());
    fprintf(stderr, "\n");
    fprintf(stderr, "範例:\n");
    fprintf(stderr, "  %s path/to/model.bin path/to/dataset\n", argv[0]);
    fprintf(stderr, "  %s -t 8 --language zh path/to/model.bin path/to/dataset\n", argv[0]);
    fprintf(stderr, "  %s --translate path/to/model.bin path/to/dataset\n", argv[0]);
    fprintf(stderr, "\n");
}

// 解析命令行參數
bool parse_params(int argc, char **argv, whisper_basic_params &params) {
    // 至少需要模型和音頻文件兩個參數
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
            // 位置參數: 模型路徑和音頻文件路徑
            if (positional_args == 0) {
                params.model_path = arg;
            } else if (positional_args == 1) {
                params.dataset_path = arg;
            }
            positional_args++;
        }
    }

    // 檢查是否獲取到必要的參數
    if (params.model_path.empty() || params.dataset_path.empty()) {
        fprintf(stderr, "錯誤: 需要指定模型路徑和資料集路徑\n");
        return false;
    }

    return true;
}

int main(int argc, char **argv) {
    whisper_basic_params params;

    // 解析命令行參數
    if (!parse_params(argc, argv, params)) {
        print_usage(argc, argv, params);
        return 1;
    }

    fprintf(stderr, "設定: 線程數 = %d, 語言 = %s, %s模式\n", 
        params.n_threads, 
        params.language.c_str());

    // 初始化 whisper
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context *ctx = whisper_init_from_file_with_params(params.model_path.c_str(), cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "錯誤: 無法初始化 whisper 模型 '%s'\n", params.model_path.c_str());
        return 1;
    }

    fprintf(stderr, "模型: %s\n", params.model_path.c_str());
    fprintf(stderr, "系統資訊: %s\n", whisper_print_system_info());
    fprintf(stderr, "\n");

    // 準備 whisper 參數
    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress   = false;
    wparams.print_special    = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.translate        = params.translate;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;
    wparams.offset_ms        = 0;
    wparams.no_context       = true;

    fprintf(stderr, "\n");
    // 初始化 whisper

    // 載入資料集檔案列表 {dataset_path}/list.txt
    std::vector<std::string> audio_files;
    std::ifstream list_file(params.dataset_path + "/list.txt");
    std::string line;
    while (std::getline(list_file, line)) {
        audio_files.push_back(params.dataset_path + "/audios/" + line);
    }
    list_file.close();

    // create a file result.txt to save the results
    std::ofstream result_file(params.dataset_path + "/result.txt");
    if (!result_file.is_open()) {
        fprintf(stderr, "錯誤: 無法打開結果文件 '%s/result.txt'\n", params.dataset_path.c_str());
        whisper_free(ctx);
        return 1;
    }

    int duration = 0;
    // read all audio files
    for (const auto &audio_file : audio_files) {
        fprintf(stderr, "process: %d/%d\n", &audio_file - &audio_files[0] + 1, (int)audio_files.size());
        std::vector<float> pcmf32;               // 單聲道 F32 PCM
        std::vector<std::vector<float>> pcmf32s; // 立體聲 F32 PCM
        if (!read_audio_data(audio_file, pcmf32, pcmf32s, false)) {
            fprintf(stderr, "錯誤: 無法讀取音頻文件 '%s'\n", audio_file.c_str());
            return 1;
        }

        // get current time
        int start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            fprintf(stderr, "錯誤: 處理音頻失敗\n");
            whisper_free(ctx);
            return 1;
        }
        // get current time
        int end_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        duration += (end_time - start_time);
        fprintf(stderr, "處理時間: %d毫秒\n", end_time - start_time);

        // 輸出結果
        const int n_segments = whisper_full_n_segments(ctx);
        std::string full_text = "";
        for (int i = 0; i < n_segments; ++i) {
            const char *text = whisper_full_get_segment_text(ctx, i);
            full_text += text;
        }
        printf("結果:%s\n\n", full_text.c_str());
        result_file << end_time - start_time << "\t" << full_text << "\n";
        // flush the result file
        result_file.flush();
    }
}

