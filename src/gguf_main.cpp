#include "vortexrt/gguf.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string escape_preview(const std::string& value, std::size_t max_bytes = 96) {
    std::ostringstream out;
    const auto count = std::min(value.size(), max_bytes);

    for (std::size_t i = 0; i < count; ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
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
                << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<unsigned>(c)
                << std::dec << std::setfill(' ');
        }
    }

    if (value.size() > count) {
        out << "...";
    }

    return out.str();
}

void print_value(const vortexrt::GgufValue& value) {
    if (value.is_array()) {
        std::cout << "array<"
                  << vortexrt::gguf_value_type_name(value.array_element_type)
                  << ">[" << value.array.size() << "]";

        if (!value.array.empty() &&
            value.array_element_type == vortexrt::GgufValueType::String) {
            const auto first = value.array.front().as_string();
            if (first) {
                std::cout << " first=\"" << escape_preview(*first, 48) << "\"";
            }
        }
        return;
    }

    if (const auto text = value.as_string()) {
        std::cout << "\"" << escape_preview(*text) << "\"";
        return;
    }
    if (const auto number = value.as_u64()) {
        std::cout << *number;
        return;
    }
    if (const auto number = value.as_i64()) {
        std::cout << *number;
        return;
    }
    if (const auto number = value.as_f64()) {
        std::cout << *number;
        return;
    }
    if (const auto boolean = value.as_bool()) {
        std::cout << (*boolean ? "true" : "false");
        return;
    }

    std::cout << "<unprintable>";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: vortexrt-gguf-info <model.gguf> [--expect-arch gemma3]\n";
        return 2;
    }

    try {
        std::string expected_arch;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--expect-arch") {
                expected_arch = argv[++i];
            }
        }

        const vortexrt::GgufFile file(argv[1]);

        std::cout << "Vortex-RT GGUF probe\n";
        std::cout << "  Version: " << file.version() << "\n";
        std::cout << "  File bytes: " << file.file_size() << "\n";
        std::cout << "  Tensor data offset: " << file.data_offset() << "\n";
        std::cout << "  Alignment: " << file.alignment() << "\n";
        std::cout << "  Metadata entries: " << file.metadata().size() << "\n";
        std::cout << "  Tensors: " << file.tensors().size() << "\n";

        const auto architecture = file.metadata_string("general.architecture");
        if (architecture) {
            std::cout << "  Architecture: " << *architecture << "\n";
        }

        if (!expected_arch.empty() && architecture != expected_arch) {
            throw std::runtime_error(
                "architecture mismatch: expected " + expected_arch +
                ", got " + architecture.value_or("<missing>"));
        }

        std::cout << "\nMetadata:\n";
        for (const auto& kv : file.metadata()) {
            std::cout << "  " << kv.key << " = ";
            print_value(kv.value);
            std::cout << "\n";
        }

        std::uint64_t q8_count = 0;
        std::cout << "\nTensors:\n";
        for (const auto& tensor : file.tensors()) {
            if (tensor.type == 8) {
                ++q8_count;
            }

            std::cout << "  " << tensor.name << " [";
            for (std::size_t d = 0; d < tensor.dimensions.size(); ++d) {
                if (d != 0) {
                    std::cout << " x ";
                }
                std::cout << tensor.dimensions[d];
            }
            std::cout << "] "
                      << vortexrt::ggml_type_name(tensor.type)
                      << "(" << tensor.type << ")"
                      << " offset=" << tensor.offset << "\n";
        }

        std::cout << "\nSummary:\n";
        std::cout << "  Q8_0 tensors: " << q8_count << "\n";

        if (const auto* tokens = file.find_metadata("tokenizer.ggml.tokens");
            tokens != nullptr && tokens->is_array()) {
            std::cout << "  Tokenizer tokens readable: " << tokens->array.size() << "\n";

            const std::size_t show = std::min<std::size_t>(tokens->array.size(), 8);
            for (std::size_t i = 0; i < show; ++i) {
                if (const auto token = tokens->array[i].as_string()) {
                    std::cout << "    token[" << i << "] = \""
                              << escape_preview(*token, 64) << "\"\n";
                }
            }
        } else {
            std::cout << "  Tokenizer tokens readable: no\n";
        }

        std::cout << "  Validation: OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "vortexrt-gguf-info: " << e.what() << "\n";
        return 1;
    }
}
