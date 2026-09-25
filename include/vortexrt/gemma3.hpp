#pragma once

#include "vortexrt/argmax.hpp"
#include "vortexrt/gguf.hpp"
#include "vortexrt/q8_matvec.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
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
    std::uint32_t sliding_window = 0;
    float rms_epsilon = 1.0e-6f;
    float global_rope_base = 1.0e6f;
    float local_rope_base = 1.0e4f;
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

struct Gemma3GenerationResult {
    std::vector<std::uint32_t> token_ids;
    std::string text;
};

class Gemma3Model {
public:
    Gemma3Model(
        const std::string& gguf_path,
        VulkanContext& context,
        const std::string& q8_matvec_spirv,
        const std::string& q8_matvec_u8_spirv = {},
        const std::string& argmax_spirv = {},
        const std::string& q8_matvec_32_spirv = {},
        const std::string& q8_matvec_u8_32_spirv = {},
        const std::string& gelu_mul_spirv = {},
        const std::string& q8_matvec_16_spirv = {},
        const std::string& q8_matvec_u8_16_spirv = {},
        const std::string& q8_matvec_8_spirv = {},
        const std::string& q8_matvec_u8_8_spirv = {},
        std::uint32_t q8_lane_override = 0);

    [[nodiscard]] const Gemma3Config& config() const noexcept {
        return config_;
    }

    [[nodiscard]] bool projection_fusion_enabled() const noexcept {
        return fuse_projections_;
    }

    void set_projection_fusion_enabled(bool enabled) noexcept {
        fuse_projections_ = enabled;
    }

    [[nodiscard]] std::uint32_t narrow_q8_lane_count() const noexcept {
        return narrow_q8_lane_count_;
    }

    [[nodiscard]] bool fused_ffn_enabled() const noexcept {
        const auto* pipeline =
            q8_pipeline_narrow_
                ? q8_pipeline_narrow_.get()
                : &q8_pipeline_;
        return fuse_projections_ &&
            pipeline->ffn_available();
    }

    [[nodiscard]] Gemma3SingleTokenResult run_single_token(
        std::uint32_t token_id,
        std::size_t top_k = 8);

    void reset_cache();

    [[nodiscard]] Gemma3SingleTokenResult decode_token(
        std::uint32_t token_id,
        std::uint32_t position,
        std::size_t top_k = 1);

    [[nodiscard]] std::vector<std::uint32_t> tokenize(
        const std::string& text,
        bool add_bos = true) const;

    [[nodiscard]] Gemma3GenerationResult generate_greedy(
        const std::string& prompt,
        std::size_t max_new_tokens);

    [[nodiscard]] Gemma3GenerationResult generate_greedy_from_bos(
        std::size_t max_new_tokens);

private:
    struct LayerKvCache {
        std::deque<std::vector<float>> keys;
        std::deque<std::vector<float>> values;
    };

    [[nodiscard]] Q8MatVecPipeline& q8_pipeline_for(
        std::size_t input_dim) noexcept;

    [[nodiscard]] std::vector<float> run_q8_matvec(
        const std::string& tensor_name,
        const std::vector<float>& input);

    [[nodiscard]] std::size_t run_q8_matvec_to_output(
        const std::string& tensor_name,
        const std::vector<float>& input);

    [[nodiscard]] std::array<std::vector<float>, 2> run_q8_pair(
        const std::array<std::string, 2>& tensor_names,
        const std::vector<float>& input);

    [[nodiscard]] std::array<std::vector<float>, 3> run_q8_triplet(
        const std::array<std::string, 3>& tensor_names,
        const std::vector<float>& input);

    [[nodiscard]] std::vector<float> rms_norm(
        const std::vector<float>& input,
        const std::string& weight_name) const;

    [[nodiscard]] std::vector<float> rms_norm_heads(
        const std::vector<float>& input,
        std::uint32_t head_count,
        const std::string& weight_name) const;

    [[nodiscard]] std::vector<float> token_embedding(
        std::uint32_t token_id) const;

    [[nodiscard]] std::vector<Gemma3TopToken> top_logits(
        const std::vector<float>& hidden,
        std::size_t top_k);

    [[nodiscard]] std::string token_piece(
        std::uint32_t token_id) const;

    [[nodiscard]] std::string decode_piece(
        std::uint32_t token_id) const;

    [[nodiscard]] bool is_global_layer(
        std::uint32_t layer) const noexcept;

    static void apply_rope(
        std::vector<float>& values,
        std::uint32_t head_count,
        std::uint32_t head_dim,
        const std::vector<float>& rope_cos,
        const std::vector<float>& rope_sin);

    [[nodiscard]] static std::vector<float> build_rope_inverse_frequencies(
        std::uint32_t head_dim,
        float theta);

    static void build_rope_table(
        const std::vector<float>& inverse_frequencies,
        std::uint32_t position,
        std::vector<float>& rope_cos,
        std::vector<float>& rope_sin);

    static void add_inplace(
        std::vector<float>& dst,
        const std::vector<float>& src);

    static float gelu_tanh(float x);

    GgufFile gguf_;
    VulkanContext& context_;
    Q8MatVecPipeline q8_pipeline_;
    std::unique_ptr<Q8MatVecPipeline> q8_pipeline_narrow_;
    std::unique_ptr<ArgmaxPipeline> argmax_pipeline_;
    Gemma3Config config_{};
    bool fuse_projections_ = true;
    bool force_q8_lane_override_ = false;
    std::uint32_t narrow_q8_lane_count_ = 64;

    std::vector<float> local_rope_inverse_frequencies_;
    std::vector<float> global_rope_inverse_frequencies_;
    std::unordered_map<std::string, std::vector<float>> f32_weights_;
    std::unique_ptr<Buffer> weights_arena_;
    std::unique_ptr<Buffer> activation_input_;
    std::unique_ptr<Buffer> activation_output_;
    std::unique_ptr<Buffer> staging_input_;
    std::unique_ptr<Buffer> staging_output_;
    std::unique_ptr<Buffer> argmax_output_;
    std::vector<LayerKvCache> kv_cache_;
    std::uint32_t next_position_ = 0;
};

} // namespace vortexrt
