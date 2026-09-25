#include "vortexrt/build_config.hpp"
#include "vortexrt/gemma3.hpp"
#include "vortexrt/vulkan_context.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string model_path;
    std::string device;
    std::string prompt;
    std::uint32_t token_id = 2;
    std::size_t top_k = 8;
    std::size_t generate_tokens = 0;
};

std::uint32_t parse_u32(
    const std::string& text,
    const char* option) {

    std::size_t used = 0;
    const unsigned long long value =
        std::stoull(text, &used, 10);
    if (used != text.size() ||
        value >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            std::string(option) +
            " must be a uint32");
    }
    return static_cast<std::uint32_t>(value);
}

std::size_t parse_size(
    const std::string& text,
    const char* option) {

    std::size_t used = 0;
    const unsigned long long value =
        std::stoull(text, &used, 10);
    if (used != text.size() ||
        value >
            std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            std::string(option) +
            " is out of range");
    }
    return static_cast<std::size_t>(value);
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        throw std::runtime_error(
            "usage: vortexrt-gemma <model.gguf> "
            "[--token-id N] [--top-k N] [--generate N] "
            "[--device <index|name>]");
    }

    Options options{};
    options.model_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];

        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    std::string(name) +
                    " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--token-id") {
            options.token_id =
                parse_u32(
                    next("--token-id"),
                    "--token-id");
        } else if (arg == "--top-k") {
            options.top_k =
                parse_size(
                    next("--top-k"),
                    "--top-k");
        } else if (arg == "--prompt") {
            options.prompt = next("--prompt");
        } else if (arg == "--generate") {
            options.generate_tokens =
                parse_size(
                    next("--generate"),
                    "--generate");
            if (options.generate_tokens > 128) {
                throw std::runtime_error(
                    "--generate is capped at 128 during bring-up");
            }
        } else if (arg == "--device") {
            options.device = next("--device");
        } else if (arg == "--help" ||
                   arg == "-h") {
            std::cout
                << "usage: vortexrt-gemma <model.gguf> [options]\n"
                << "  --token-id N          input token id (default: 2/BOS)\n"
                << "  --top-k N             display top logits; 0 skips LM head\n"
                << "  --prompt TEXT         tokenize/prefill a text prompt before generation\n"
                << "  --generate N          greedy-generate N tokens with KV cache\n"
                << "  --device <index|name> Vulkan device selector\n";
            std::exit(0);
        } else {
            throw std::runtime_error(
                "unknown argument: " + arg);
        }
    }

    return options;
}

std::string escape_piece(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        if (c == '\n') {
            out << "\\n";
        } else if (c == '\r') {
            out << "\\r";
        } else if (c == '\t') {
            out << "\\t";
        } else if (c >= 32 && c < 127) {
            out << static_cast<char>(c);
        } else {
            out << "\\x"
                << std::hex
                << std::setw(2)
                << std::setfill('0')
                << static_cast<unsigned>(c)
                << std::dec
                << std::setfill(' ');
        }
    }
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options =
            parse_options(argc, argv);

        const auto runtime_start =
            std::chrono::steady_clock::now();

        vortexrt::VulkanContext context(
            options.device);

        const auto model_start =
            std::chrono::steady_clock::now();

        vortexrt::Gemma3Model model(
            options.model_path,
            context,
            vortexrt::build_config::q8_matvec_spv,
            vortexrt::build_config::q8_matvec_u8_spv,
            vortexrt::build_config::argmax_spv,
            vortexrt::build_config::q8_matvec_32_spv,
            vortexrt::build_config::q8_matvec_u8_32_spv);

        const auto model_ready =
            std::chrono::steady_clock::now();

        const double model_load_ms =
            std::chrono::duration<double, std::milli>(
                model_ready - model_start).count();
        const double runtime_setup_ms =
            std::chrono::duration<double, std::milli>(
                model_ready - runtime_start).count();

        const auto& config = model.config();

        std::cout
            << "Vortex-RT Gemma 3 runtime\n"
            << "  GPU: "
            << context.capabilities().name << "\n"
            << "  Layers: " << config.block_count << "\n"
            << "  Hidden: "
            << config.embedding_length << "\n"
            << "  FFN: "
            << config.feed_forward_length << "\n"
            << "  Heads: "
            << config.head_count
            << " / KV " << config.head_count_kv
            << "\n"
            << "  Head dim: "
            << config.head_dim << "\n"
            << "  Sliding window: "
            << config.sliding_window << "\n"
            << "  Vocab: "
            << config.vocab_size << "\n"
            << "  Projection fusion: "
            << (model.projection_fusion_enabled()
                    ? "enabled"
                    : "disabled")
            << "\n"
            << "  Model load: "
            << model_load_ms << " ms\n"
            << "  Runtime setup: "
            << runtime_setup_ms << " ms\n";

        if (options.generate_tokens != 0) {
            std::cout
                << "  Greedy generation from BOS: "
                << options.generate_tokens
                << " token(s)\n";

            const auto prompt_tokens =
                model.tokenize(options.prompt, true);

            std::cout << "  Prompt token ids:";
            for (const auto id : prompt_tokens) {
                std::cout << " " << id;
            }
            std::cout << "\n";

            const auto generation_start =
                std::chrono::steady_clock::now();

            const auto generated =
                options.prompt.empty()
                    ? model.generate_greedy_from_bos(
                          options.generate_tokens)
                    : model.generate_greedy(
                          options.prompt,
                          options.generate_tokens);

            const auto generation_end =
                std::chrono::steady_clock::now();
            const double generation_ms =
                std::chrono::duration<double, std::milli>(
                    generation_end - generation_start).count();
            const double tokens_per_second =
                generation_ms > 0.0
                    ? (static_cast<double>(
                           generated.token_ids.size()) *
                       1000.0 / generation_ms)
                    : 0.0;

            std::cout << "  Generated token ids:";
            for (const auto id : generated.token_ids) {
                std::cout << " " << id;
            }
            std::cout << "\n";

            std::cout
                << "  Generation time: "
                << generation_ms << " ms\n"
                << "  Throughput: "
                << tokens_per_second << " tok/s\n"
                << "  Generated text: \""
                << escape_piece(generated.text)
                << "\"\n"
                << "TEXT_BEGIN\n"
                << generated.text
                << "\nTEXT_END\n"
                << "  Validation: OK\n";
            return 0;
        }

        std::cout
            << "  Token: " << options.token_id << "\n";

        const auto forward_start =
            std::chrono::steady_clock::now();

        const auto result =
            model.run_single_token(
                options.token_id,
                options.top_k);

        const auto forward_end =
            std::chrono::steady_clock::now();
        const double forward_ms =
            std::chrono::duration<double, std::milli>(
                forward_end - forward_start).count();

        double checksum = 0.0;
        double sum_squares = 0.0;
        float max_abs = 0.0f;

        for (std::size_t i = 0;
             i < result.hidden.size();
             ++i) {
            const float value =
                result.hidden[i];
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "final hidden state is non-finite");
            }

            checksum +=
                static_cast<double>(value) *
                static_cast<double>(i + 1);
            sum_squares +=
                static_cast<double>(value) *
                static_cast<double>(value);
            max_abs =
                std::max(
                    max_abs,
                    std::fabs(value));
        }

        const double rms =
            std::sqrt(
                sum_squares /
                static_cast<double>(
                    result.hidden.size()));

        std::cout
            << "  Forward time: "
            << forward_ms << " ms\n"
            << "  Final hidden RMS: "
            << rms << "\n"
            << "  Final hidden max abs: "
            << max_abs << "\n"
            << "  Final hidden checksum: "
            << std::setprecision(12)
            << checksum << "\n";

        if (!result.top_tokens.empty()) {
            std::cout << "  Top logits:\n";
            for (std::size_t i = 0;
                 i < result.top_tokens.size();
                 ++i) {
                const auto& token =
                    result.top_tokens[i];

                std::cout
                    << "    " << i + 1
                    << ": id=" << token.token_id
                    << " logit="
                    << token.logit
                    << " piece=\""
                    << escape_piece(token.piece)
                    << "\"\n";
            }
        }

        std::cout << "  Validation: OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr
            << "vortexrt-gemma: "
            << e.what() << "\n";
        return 1;
    }
}
