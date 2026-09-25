#pragma once

#include "vortexrt/buffer.hpp"
#include "vortexrt/gguf.hpp"
#include "vortexrt/q8_matvec.hpp"
#include "vortexrt/vulkan_context.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vortexrt {

struct Gemma3GenerationStats {
    std::uint64_t matvec_dispatches = 0;
    double total_seconds = 0.0;
};

class Gemma3Model {
public:
    Gemma3Model(
        VulkanContext& context,
        const std::filesystem::path& model_path,
        const std::string& q8_matvec_spirv);

    Gemma3Model(const Gemma3Model&) = delete;
    Gemma3Model& operator=(const Gemma3Model&) = delete;

    [[nodiscard]] std::vector<std::uint32_t> tokenize(
        const std::string& text,
        bool add_bos = true) const;

    [[nodiscard]] std::string decode_token(
        std::uint32_t token_id) const;

    [[nodiscard]] std::vector<std::uint32_t> generate(
        const std::string& prompt,
        std::uint32_t max_new_tokens,
        Gemma3GenerationStats* stats = nullptr);

    [[nodiscard]] const std::string& model_name() const noexcept {
        return model_name_;
    }

    [[nodiscard]] std::uint32_t vocab_size() const noexcept {
        return vocab_size_;
    }

private:
    struct TensorRef {
        std::uint32_t offset = 0;
        std::uint32_t input_dim = 0;
        std::uint32_t output_dim = 0;
    };

    struct LayerWeights {
        TensorRef q;
        TensorRef k;
        TensorRef v;
        TensorRef o;
        TensorRef ffn_gate;
        TensorRef ffn_up;
        TensorRef ffn_down;

        std::vector<float> attn_norm;
        std::vector<float> post_attention_norm;
        std::vector<float> ffn_norm;
        std::vector<float> post_ffw_norm;
        std::vector<float> q_norm;
        std::vector<float> k_norm;

        bool sliding = true;
    };

    struct LayerCache {
        std::vector<float> keys;
        std::vector<float> values;
    };

    [[nodiscard]] TensorRef require_q8_matrix(
        const std::string& name,
        std::uint32_t expected_input,
        std::uint32_t expected_output) const;

    [[nodiscard]] std::vector<float> require_f32_vector(
        const std::string& name,
        std::uint32_t expected_size) const;

    [[nodiscard]] std::vector<float> dequant_embedding(
        std::uint32_t token_id) const;

    [[nodiscard]] std::vector<float> matvec(
        const TensorRef& matrix,
        const std::vector<float>& input);

    [[nodiscard]] std::vector<float> rms_norm(
        const std::vector<float>& input,
        const std::vector<float>& weight,
        std::uint32_t block_size) const;

    void apply_rope(
        std::vector<float>& values,
        std::uint32_t head_count,
        std::uint32_t position,
        double base) const;

    [[nodiscard]] std::vector<float> attend(
        std::uint32_t layer_index,
        const std::vector<float>& query,
        const std::vector<float>& key,
        const std::vector<float>& value,
        std::uint32_t position);

    [[nodiscard]] std::optional<std::uint32_t> forward_token(
        std::uint32_t token_id,
        std::uint32_t position,
        bool compute_logits);

    void reset_cache();

    [[nodiscard]] static float half_to_float(std::uint16_t value);
    [[nodiscard]] static float gelu_tanh(float x);
    [[nodiscard]] static std::string sentencepiece_normalize(
        const std::string& text);
    [[nodiscard]] static std::vector<std::string> utf8_symbols(
        const std::string& text);
    [[nodiscard]] bool tokenizer_mergeable(std::uint32_t token_id) const;

    VulkanContext& context_;
    GgufFile gguf_;
    std::vector<std::uint8_t> tensor_data_;

    std::unique_ptr<Buffer> model_buffer_;
    std::unique_ptr<Buffer> input_buffer_;
    std::unique_ptr<Buffer> output_buffer_;
    std::unique_ptr<Q8MatVecPipeline> q8_pipeline_;

    std::string model_name_;
    std::uint32_t hidden_size_ = 0;
    std::uint32_t intermediate_size_ = 0;
    std::uint32_t layer_count_ = 0;
    std::uint32_t head_count_ = 0;
    std::uint32_t kv_head_count_ = 0;
    std::uint32_t head_dim_ = 0;
    std::uint32_t vocab_size_ = 0;
    std::uint32_t sliding_window_ = 0;
    float rms_eps_ = 1e-6f;
    double global_rope_base_ = 1'000'000.0;
    double local_rope_base_ = 10'000.0;
    float attention_scale_ = 1.0f / 16.0f;

    TensorRef token_embedding_;
    std::vector<float> output_norm_;
    std::vector<LayerWeights> layers_;
    std::vector<LayerCache> cache_;

    std::vector<std::string> tokens_;
    std::vector<float> token_scores_;
    std::vector<std::int32_t> token_types_;
    std::unordered_map<std::string, std::uint32_t> token_to_id_;

    std::uint32_t bos_token_id_ = 2;
    std::uint32_t eos_token_id_ = 1;
    std::uint32_t unk_token_id_ = 3;

    std::uint64_t matvec_dispatches_ = 0;
};

} // namespace vortexrt
