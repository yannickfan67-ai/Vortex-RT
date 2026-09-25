#include "vortexrt/build_config.hpp"
#include "vortexrt/gemma3.hpp"
#include "vortexrt/vulkan_context.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string device;
    std::string prompt;
    std::uint32_t tokens = 4;
};

std::uint32_t parse_u32(
    const std::string& text,
    const char* name) {

    std::size_t used = 0;
    const unsigned long long value =
        std::stoull(text, &used, 10);

    if (used != text.size() ||
        value == 0 ||
        value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            std::string(name) + " must be a positive uint32");
    }

    return static_cast<std::uint32_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options options{};

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value =
            [&](const char* option) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(
                        std::string(option) + " requires a value");
                }
                return argv[++i];
            };

        if (arg == "--model" || arg == "-m") {
            options.model = require_value("--model");
        } else if (arg == "--device") {
            options.device = require_value("--device");
        } else if (arg == "--prompt" || arg == "-p") {
            options.prompt = require_value("--prompt");
        } else if (arg == "--tokens" || arg == "-n") {
            options.tokens =
                parse_u32(
                    require_value("--tokens"),
                    "--tokens");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: vortexrt-gemma --model <model.gguf> [options]\n"
                << "  -m, --model <path>       Gemma 3 270M Q8_0 GGUF\n"
                << "  -p, --prompt <text>      prompt text (default: BOS only)\n"
                << "  -n, --tokens <count>     greedy new tokens (default: 4)\n"
                << "      --device <index|name> Vulkan device selector\n";
            std::exit(0);
        } else {
            throw std::runtime_error(
                "unknown argument: " + arg);
        }
    }

    if (options.model.empty()) {
        throw std::runtime_error("--model is required");
    }

    return options;
}

std::string escape_for_log(const std::string& text) {
    std::string output;

    for (unsigned char c : text) {
        if (c == '\n') {
            output += "\\n";
        } else if (c == '\r') {
            output += "\\r";
        } else if (c == '\t') {
            output += "\\t";
        } else if (c < 32 || c == 127) {
            static constexpr char hex[] =
                "0123456789ABCDEF";
            output += "\\x";
            output.push_back(hex[(c >> 4) & 0x0f]);
            output.push_back(hex[c & 0x0f]);
        } else {
            output.push_back(static_cast<char>(c));
        }
    }

    return output;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options =
            parse_options(argc, argv);

        vortexrt::VulkanContext context(options.device);

        std::cout << "Vortex-RT Gemma 3 Vulkan runner\n";
        std::cout << "  Device: "
                  << context.capabilities().name << "\n";
        std::cout << "  Loading: "
                  << options.model << "\n";

        vortexrt::Gemma3Model model(
            context,
            options.model,
            vortexrt::build_config::q8_matvec_spv);

        const std::vector<std::uint32_t> prompt_tokens =
            model.tokenize(options.prompt, true);

        std::cout << "  Model: "
                  << model.model_name() << "\n";
        std::cout << "  Vocabulary: "
                  << model.vocab_size() << "\n";
        std::cout << "  Prompt: "
                  << std::quoted(options.prompt) << "\n";
        std::cout << "  Prompt tokens:";
        for (std::uint32_t token : prompt_tokens) {
            std::cout << " " << token;
        }
        std::cout << "\n";

        vortexrt::Gemma3GenerationStats stats{};
        const std::vector<std::uint32_t> generated =
            model.generate(
                options.prompt,
                options.tokens,
                &stats);

        std::string continuation;

        std::cout << "  Generated tokens:\n";
        for (std::uint32_t token : generated) {
            const std::string piece =
                model.decode_token(token);
            continuation += piece;

            std::cout
                << "    " << token
                << "  \""
                << escape_for_log(piece)
                << "\"\n";
        }

        std::cout << "Generated continuation: "
                  << continuation << "\n";
        std::cout << "Generated text: "
                  << options.prompt
                  << continuation << "\n";
        std::cout << "MatVec dispatches: "
                  << stats.matvec_dispatches << "\n";
        std::cout << "Total generation time: "
                  << std::fixed
                  << std::setprecision(3)
                  << stats.total_seconds
                  << " s\n";

        if (stats.total_seconds > 0.0) {
            std::cout << "Generated token throughput: "
                      << std::setprecision(3)
                      << (static_cast<double>(generated.size()) /
                          stats.total_seconds)
                      << " tok/s\n";
        }

        std::cout << "Generation: OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr
            << "vortexrt-gemma: "
            << e.what()
            << "\n";
        return 1;
    }
}
