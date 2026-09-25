#include "vortexrt/gemma3.hpp"

#include "vortexrt/buffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vortexrt {
namespace {

std::uint32_t require_u32(
    const GgufFile& gguf,
    const std::string& key) {

    const auto value = gguf.metadata_u64(key);
    if (!value ||
        *value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Missing/invalid Gemma 3 metadata: " + key);
    }
    return static_cast<std::uint32_t>(*value);
}

float require_f32(
    const GgufFile& gguf,
    const std::string& key) {

    const auto* value = gguf.find_metadata(key);
    if (value == nullptr) {
        throw std::runtime_error(
            "Missing Gemma 3 metadata: " + key);
    }
    const auto number = value->as_f64();
    if (!number || !std::isfinite(*number)) {
        throw std::runtime_error(
            "Invalid Gemma 3 floating metadata: " + key);
    }
    return static_cast<float>(*number);
}

const GgufTensorInfo* require_tensor(
    const GgufFile& gguf,
    const std::string& name) {

    const auto* tensor = gguf.find_tensor(name);
    if (tensor == nullptr) {
        throw std::runtime_error(
            "Missing Gemma 3 tensor: " + name);
    }
    return tensor;
}

std::string layer_tensor(
    std::uint32_t layer,
    const char* suffix) {

    return "blk." + std::to_string(layer) + "." + suffix;
}

} // namespace

Gemma3Model::Gemma3Model(
    const std::string& gguf_path,
    VulkanContext& context,
    const std::string& q8_matvec_spirv)
    : gguf_(gguf_path),
      context_(context),
      q8_pipeline_(context, q8_matvec_spirv) {

    const auto architecture =
        gguf_.metadata_string("general.architecture");
    if (architecture != std::optional<std::string>{"gemma3"}) {
        throw std::runtime_error(
            "Gemma3Model requires general.architecture=gemma3");
    }

    config_.context_length =
        require_u32(gguf_, "gemma3.context_length");
    config_.embedding_length =
        require_u32(gguf_, "gemma3.embedding_length");
    config_.block_count =
        require_u32(gguf_, "gemma3.block_count");
    config_.feed_forward_length =
        require_u32(gguf_, "gemma3.feed_forward_length");
    config_.head_count =
        require_u32(gguf_, "gemma3.attention.head_count");
    config_.head_count_kv =
        require_u32(gguf_, "gemma3.attention.head_count_kv");
    config_.head_dim =
        require_u32(gguf_, "gemma3.attention.key_length");
    config_.value_dim =
        require_u32(gguf_, "gemma3.attention.value_length");
    config_.rms_epsilon =
        require_f32(
            gguf_,
            "gemma3.attention.layer_norm_rms_epsilon");

    const auto* embedding =
        require_tensor(gguf_, "token_embd.weight");
    if (embedding->type != 8 ||
        embedding->dimensions.size() != 2 ||
        embedding->dimensions[0] != config_.embedding_length ||
        embedding->dimensions[1] >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Unsupported token_embd.weight layout");
    }

    config_.vocab_size =
        static_cast<std::uint32_t>(embedding->dimensions[1]);

    if (config_.embedding_length == 0 ||
        config_.block_count == 0 ||
        config_.feed_forward_length == 0 ||
        config_.head_count == 0 ||
        config_.head_count_kv == 0 ||
        config_.head_dim == 0 ||
        config_.value_dim == 0 ||
        config_.vocab_size == 0) {
        throw std::runtime_error(
            "Gemma 3 configuration contains zero dimensions");
    }

    if (config_.head_count % config_.head_count_kv != 0) {
        throw std::runtime_error(
            "Gemma 3 head_count must be divisible by head_count_kv");
    }

    const std::uint64_t q_width =
        static_cast<std::uint64_t>(config_.head_count) *
        config_.head_dim;
    if (q_width >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Gemma 3 attention width exceeds uint32");
    }

    if (config_.head_count_kv != 1 ||
        config_.value_dim != config_.head_dim) {
        throw std::runtime_error(
            "Single-token bring-up currently requires one KV head "
            "and equal key/value head dimensions");
    }

    const auto* tokens =
        gguf_.find_metadata("tokenizer.ggml.tokens");
    if (tokens == nullptr ||
        !tokens->is_array() ||
        tokens->array.size() != config_.vocab_size) {
        throw std::runtime_error(
            "Tokenizer vocabulary does not match token embedding");
    }
}

std::vector<float> Gemma3Model::run_q8_matvec(
    const std::string& tensor_name,
    const std::vector<float>& input) {

    const auto* tensor =
        require_tensor(gguf_, tensor_name);

    if (tensor->type != 8 ||
        tensor->dimensions.size() != 2 ||
        tensor->dimensions[0] != input.size() ||
        tensor->dimensions[0] >
            std::numeric_limits<std::uint32_t>::max() ||
        tensor->dimensions[1] >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Unsupported Q8_0 matrix layout: " + tensor_name);
    }

    const auto encoded =
        gguf_.read_tensor_bytes(*tensor);
    if (encoded.empty()) {
        throw std::runtime_error(
            "Q8_0 tensor payload is empty: " + tensor_name);
    }

    const auto input_bytes =
        static_cast<VkDeviceSize>(
            input.size() * sizeof(float));
    const auto output_elements =
        static_cast<std::size_t>(tensor->dimensions[1]);
    const auto output_bytes =
        static_cast<VkDeviceSize>(
            output_elements * sizeof(float));

    Buffer weights(
        context_,
        static_cast<VkDeviceSize>(encoded.size()),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    Buffer gpu_input(
        context_,
        input_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    Buffer gpu_output(
        context_,
        output_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    weights.upload(encoded.data(), encoded.size());
    gpu_input.upload(
        input.data(),
        input.size() * sizeof(float));

    q8_pipeline_.run(
        weights,
        gpu_input,
        gpu_output,
        0,
        static_cast<std::uint32_t>(tensor->dimensions[0]),
        static_cast<std::uint32_t>(tensor->dimensions[1]));

    std::vector<float> output(output_elements);
    gpu_output.download(
        output.data(),
        output.size() * sizeof(float));

    return output;
}

std::vector<float> Gemma3Model::rms_norm(
    const std::vector<float>& input,
    const std::string& weight_name) const {

    const auto* tensor =
        require_tensor(gguf_, weight_name);
    const auto weights =
        gguf_.read_f32_tensor(*tensor);

    if (weights.size() != input.size()) {
        throw std::runtime_error(
            "RMSNorm shape mismatch: " + weight_name);
    }

    double sum_squares = 0.0;
    for (const float value : input) {
        sum_squares +=
            static_cast<double>(value) *
            static_cast<double>(value);
    }

    const double mean_square =
        sum_squares /
        static_cast<double>(input.size());
    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(mean_square) +
            config_.rms_epsilon);

    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        // Gemma 3 GGUF conversion already folds the model's
        // (1 + weight) RMSNorm convention into the stored weight.
        output[i] = input[i] * scale * weights[i];
    }
    return output;
}

std::vector<float> Gemma3Model::token_embedding(
    std::uint32_t token_id) const {

    if (token_id >= config_.vocab_size) {
        throw std::runtime_error(
            "Token id is outside Gemma 3 vocabulary");
    }

    const auto* embedding =
        require_tensor(gguf_, "token_embd.weight");
    auto values =
        gguf_.read_q8_0_row(*embedding, token_id);

    if (values.size() != config_.embedding_length) {
        throw std::runtime_error(
            "Token embedding row has wrong width");
    }

    const float scale =
        std::sqrt(
            static_cast<float>(config_.embedding_length));
    for (auto& value : values) {
        value *= scale;
    }

    return values;
}

std::string Gemma3Model::token_piece(
    std::uint32_t token_id) const {

    const auto* tokens =
        gguf_.find_metadata("tokenizer.ggml.tokens");
    if (tokens == nullptr ||
        !tokens->is_array() ||
        token_id >= tokens->array.size()) {
        return {};
    }

    return tokens->array[token_id]
        .as_string()
        .value_or(std::string{});
}

void Gemma3Model::add_inplace(
    std::vector<float>& dst,
    const std::vector<float>& src) {

    if (dst.size() != src.size()) {
        throw std::runtime_error(
            "Residual add shape mismatch");
    }

    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] += src[i];
    }
}

float Gemma3Model::gelu_tanh(float x) {
    constexpr float kSqrt2OverPi =
        0.7978845608028654f;
    constexpr float kCubic =
        0.044715f;

    const float cubic = x * x * x;
    return 0.5f * x *
        (1.0f +
         std::tanh(
             kSqrt2OverPi *
             (x + kCubic * cubic)));
}

Gemma3SingleTokenResult Gemma3Model::run_single_token(
    std::uint32_t token_id,
    std::size_t top_k) {

    std::vector<float> hidden =
        token_embedding(token_id);

    const std::uint32_t heads_per_kv =
        config_.head_count / config_.head_count_kv;

    for (std::uint32_t layer = 0;
         layer < config_.block_count;
         ++layer) {

        const auto attn_input =
            rms_norm(
                hidden,
                layer_tensor(layer, "attn_norm.weight"));

        // Position zero has exactly one causal key/value. The
        // softmax therefore equals 1.0 for every query head, so
        // Q/K/RoPE do not affect this single-token bring-up path.
        const auto value =
            run_q8_matvec(
                layer_tensor(layer, "attn_v.weight"),
                attn_input);

        if (value.size() != config_.value_dim) {
            throw std::runtime_error(
                "Gemma 3 V projection has unexpected width");
        }

        std::vector<float> attention_concat;
        attention_concat.reserve(
            static_cast<std::size_t>(
                config_.head_count) *
            config_.value_dim);

        for (std::uint32_t kv = 0;
             kv < config_.head_count_kv;
             ++kv) {
            const std::size_t base =
                static_cast<std::size_t>(kv) *
                config_.value_dim;

            for (std::uint32_t repeat = 0;
                 repeat < heads_per_kv;
                 ++repeat) {
                attention_concat.insert(
                    attention_concat.end(),
                    value.begin() +
                        static_cast<std::ptrdiff_t>(base),
                    value.begin() +
                        static_cast<std::ptrdiff_t>(
                            base + config_.value_dim));
            }
        }

        auto attention_output =
            run_q8_matvec(
                layer_tensor(
                    layer,
                    "attn_output.weight"),
                attention_concat);

        attention_output =
            rms_norm(
                attention_output,
                layer_tensor(
                    layer,
                    "post_attention_norm.weight"));

        add_inplace(attention_output, hidden);
        hidden = std::move(attention_output);

        const auto ffn_input =
            rms_norm(
                hidden,
                layer_tensor(layer, "ffn_norm.weight"));

        auto gate =
            run_q8_matvec(
                layer_tensor(layer, "ffn_gate.weight"),
                ffn_input);
        auto up =
            run_q8_matvec(
                layer_tensor(layer, "ffn_up.weight"),
                ffn_input);

        if (gate.size() != up.size() ||
            gate.size() != config_.feed_forward_length) {
            throw std::runtime_error(
                "Gemma 3 FFN projection width mismatch");
        }

        for (std::size_t i = 0; i < gate.size(); ++i) {
            gate[i] = gelu_tanh(gate[i]) * up[i];
        }

        auto ffn_output =
            run_q8_matvec(
                layer_tensor(layer, "ffn_down.weight"),
                gate);

        ffn_output =
            rms_norm(
                ffn_output,
                layer_tensor(
                    layer,
                    "post_ffw_norm.weight"));

        add_inplace(ffn_output, hidden);
        hidden = std::move(ffn_output);

        for (const float value_out : hidden) {
            if (!std::isfinite(value_out)) {
                throw std::runtime_error(
                    "Non-finite value after Gemma 3 block " +
                    std::to_string(layer));
            }
        }
    }

    hidden =
        rms_norm(hidden, "output_norm.weight");

    Gemma3SingleTokenResult result{};
    result.hidden = hidden;

    if (top_k == 0) {
        return result;
    }

    auto logits =
        run_q8_matvec(
            "token_embd.weight",
            hidden);

    if (logits.size() != config_.vocab_size) {
        throw std::runtime_error(
            "Gemma 3 tied output logits have wrong width");
    }

    top_k = std::min<std::size_t>(
        top_k,
        logits.size());

    std::vector<std::uint32_t> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0u);

    std::partial_sort(
        ids.begin(),
        ids.begin() +
            static_cast<std::ptrdiff_t>(top_k),
        ids.end(),
        [&](std::uint32_t a, std::uint32_t b) {
            return logits[a] > logits[b];
        });

    result.top_tokens.reserve(top_k);
    for (std::size_t i = 0; i < top_k; ++i) {
        const auto id = ids[i];
        if (!std::isfinite(logits[id])) {
            throw std::runtime_error(
                "Non-finite Gemma 3 logit");
        }

        result.top_tokens.push_back(
            Gemma3TopToken{
                id,
                logits[id],
                token_piece(id),
            });
    }

    return result;
}

} // namespace vortexrt
