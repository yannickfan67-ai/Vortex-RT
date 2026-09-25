#pragma once

#include "vortexrt/gguf.hpp"
#include "vortexrt/q8_matvec.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vortexrt {

struct Gemma3Config {
    std::uint32_t context_length = 0;
    std::uint32_t embedding_length = 0;
    std::uint32_t block_count = 0;
    std::uint32_t feed_forward_length = 0;
    std::uint32_t head_count = 0;
    std::uint32_t head_count_kv = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t value_dim = 0;
    std::uint32_t vocab_size = 0;
    float rms_epsilon = 1.0e-6f;
};

struct Gemma3TopToken {
    std::uint32_t token_id = 0;
    float logit = 0.0f;
    std::string piece;
};

struct Gemma3SingleTokenResult {
    std::vector<float> hidden;
    std::vector<Gemma3TopToken> top_tokens;
};

class Gemma3Model {
public:
    Gemma3Model(
        const std::string& gguf_path,
        VulkanContext& context,
        const std::string& q8_matvec_spirv);

    [[nodiscard]] const Gemma3Config& config() const noexcept {
        return config_;
    }

    [[nodiscard]] Gemma3SingleTokenResult run_single_token(
        std::uint32_t token_id,
        std::size_t top_k = 8);

private:
    [[nodiscard]] std::vector<float> run_q8_matvec(
        const std::string& tensor_name,
        const std::vector<float>& input);

    [[nodiscard]] std::vector<float> rms_norm(
        const std::vector<float>& input,
        const std::string& weight_name) const;

    [[nodiscard]] std::vector<float> token_embedding(
        std::uint32_t token_id) const;

    [[nodiscard]] std::string token_piece(
        std::uint32_t token_id) const;

    static void add_inplace(
        std::vector<float>& dst,
        const std::vector<float>& src);

    static float gelu_tanh(float x);

    GgufFile gguf_;
    VulkanContext& context_;
    Q8MatVecPipeline q8_pipeline_;
    Gemma3Config config_{};
};

} // namespace vortexrt
