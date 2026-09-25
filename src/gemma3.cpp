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

    if (const auto sliding =
            gguf_.metadata_u64("gemma3.attention.sliding_window");
        sliding &&
        *sliding <= std::numeric_limits<std::uint32_t>::max()) {
        config_.sliding_window =
            static_cast<std::uint32_t>(*sliding);
    }

    if (const auto* rope =
            gguf_.find_metadata("gemma3.rope.freq_base");
        rope != nullptr) {
        if (const auto number = rope->as_f64();
            number && std::isfinite(*number) && *number > 0.0) {
            config_.global_rope_base =
                static_cast<float>(*number);
        }
    }

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
        static_cast<std::uint32_t>(
            embedding->dimensions[1]);

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

    if (config_.head_count_kv != 1 ||
        config_.value_dim != config_.head_dim) {
        throw std::runtime_error(
            "Current Gemma 3 decoder requires one KV head "
            "and equal key/value head dimensions");
    }

    const std::uint64_t q_width =
        static_cast<std::uint64_t>(config_.head_count) *
        config_.head_dim;
    if (q_width >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Gemma 3 attention width exceeds uint32");
    }

    const auto* tokens =
        gguf_.find_metadata("tokenizer.ggml.tokens");
    if (tokens == nullptr ||
        !tokens->is_array() ||
        tokens->array.size() != config_.vocab_size) {
        throw std::runtime_error(
            "Tokenizer vocabulary does not match token embedding");
    }

    kv_cache_.resize(config_.block_count);
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

    auto it = q8_weights_.find(tensor_name);
    if (it == q8_weights_.end()) {
        const auto encoded =
            gguf_.read_tensor_bytes(*tensor);
        if (encoded.empty()) {
            throw std::runtime_error(
                "Q8_0 tensor payload is empty: " + tensor_name);
        }

        CachedQ8Weight cached{};
        cached.tensor = tensor;
        cached.buffer = std::make_unique<Buffer>(
            context_,
            static_cast<VkDeviceSize>(encoded.size()),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        cached.buffer->upload(encoded.data(), encoded.size());

        it = q8_weights_.emplace(
            tensor_name,
            std::move(cached)).first;
    }

    const auto input_bytes =
        static_cast<VkDeviceSize>(
            input.size() * sizeof(float));
    const auto output_elements =
        static_cast<std::size_t>(
            tensor->dimensions[1]);
    const auto output_bytes =
        static_cast<VkDeviceSize>(
            output_elements * sizeof(float));

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

    gpu_input.upload(
        input.data(),
        input.size() * sizeof(float));

    q8_pipeline_.run(
        *it->second.buffer,
        gpu_input,
        gpu_output,
        0,
        static_cast<std::uint32_t>(
            tensor->dimensions[0]),
        static_cast<std::uint32_t>(
            tensor->dimensions[1]));

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
        // Gemma GGUF conversion already folds the original
        // (1 + weight) RMSNorm convention into stored weights.
        output[i] =
            input[i] * scale * weights[i];
    }
    return output;
}

std::vector<float> Gemma3Model::rms_norm_heads(
    const std::vector<float>& input,
    std::uint32_t head_count,
    const std::string& weight_name) const {

    const auto* tensor =
        require_tensor(gguf_, weight_name);
    const auto weights =
        gguf_.read_f32_tensor(*tensor);

    if (weights.size() != config_.head_dim ||
        input.size() !=
            static_cast<std::size_t>(head_count) *
            config_.head_dim) {
        throw std::runtime_error(
            "Q/K RMSNorm shape mismatch: " + weight_name);
    }

    std::vector<float> output(input.size());

    for (std::uint32_t head = 0;
         head < head_count;
         ++head) {
        const std::size_t base =
            static_cast<std::size_t>(head) *
            config_.head_dim;

        double sum_squares = 0.0;
        for (std::uint32_t i = 0;
             i < config_.head_dim;
             ++i) {
            const float value = input[base + i];
            sum_squares +=
                static_cast<double>(value) *
                static_cast<double>(value);
        }

        const float scale =
            1.0f /
            std::sqrt(
                static_cast<float>(
                    sum_squares /
                    static_cast<double>(
                        config_.head_dim)) +
                config_.rms_epsilon);

        for (std::uint32_t i = 0;
             i < config_.head_dim;
             ++i) {
            output[base + i] =
                input[base + i] *
                scale *
                weights[i];
        }
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
            static_cast<float>(
                config_.embedding_length));
    for (auto& value : values) {
        value *= scale;
    }

    return values;
}

std::vector<Gemma3TopToken> Gemma3Model::top_logits(
    const std::vector<float>& hidden,
    std::size_t top_k) {

    if (top_k == 0) {
        return {};
    }

    auto logits =
        run_q8_matvec(
            "token_embd.weight",
            hidden);

    if (logits.size() != config_.vocab_size) {
        throw std::runtime_error(
            "Gemma 3 tied output logits have wrong width");
    }

    top_k =
        std::min<std::size_t>(
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

    std::vector<Gemma3TopToken> result;
    result.reserve(top_k);

    for (std::size_t i = 0; i < top_k; ++i) {
        const auto id = ids[i];
        if (!std::isfinite(logits[id])) {
            throw std::runtime_error(
                "Non-finite Gemma 3 logit");
        }
        result.push_back(
            Gemma3TopToken{
                id,
                logits[id],
                token_piece(id),
            });
    }

    return result;
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

std::string Gemma3Model::decode_piece(
    std::uint32_t token_id) const {

    if (token_id == 0 ||
        token_id == 1 ||
        token_id == 2 ||
        token_id == 3) {
        return {};
    }

    std::string piece = token_piece(token_id);

    if (piece.size() == 6 &&
        piece[0] == '<' &&
        piece[1] == '0' &&
        piece[2] == 'x' &&
        piece[5] == '>') {
        const auto hex_value = [](char c) -> int {
            if (c >= '0' && c <= '9') {
                return c - '0';
            }
            if (c >= 'a' && c <= 'f') {
                return c - 'a' + 10;
            }
            if (c >= 'A' && c <= 'F') {
                return c - 'A' + 10;
            }
            return -1;
        };

        const int hi = hex_value(piece[3]);
        const int lo = hex_value(piece[4]);
        if (hi >= 0 && lo >= 0) {
            return std::string(
                1,
                static_cast<char>((hi << 4) | lo));
        }
    }

    const std::string marker = "\\xE2\\x96\\x81";
    std::size_t pos = 0;
    while ((pos = piece.find(marker, pos)) !=
           std::string::npos) {
        piece.replace(pos, marker.size(), " ");
        ++pos;
    }

    return piece;
}

bool Gemma3Model::is_global_layer(
    std::uint32_t layer) const noexcept {

    // Gemma 3 uses five local sliding layers followed by one
    // global layer, repeated throughout the decoder.
    return ((layer + 1u) % 6u) == 0u;
}

void Gemma3Model::apply_rope(
    std::vector<float>& values,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t position,
    float theta) {

    if (head_dim == 0 ||
        (head_dim % 2u) != 0u ||
        values.size() !=
            static_cast<std::size_t>(head_count) *
            head_dim ||
        !(theta > 0.0f)) {
        throw std::runtime_error(
            "Invalid Gemma 3 RoPE shape/configuration");
    }

    const std::uint32_t half = head_dim / 2u;

    for (std::uint32_t head = 0;
         head < head_count;
         ++head) {
        const std::size_t base =
            static_cast<std::size_t>(head) *
            head_dim;

        for (std::uint32_t i = 0;
             i < half;
             ++i) {
            const float exponent =
                (2.0f * static_cast<float>(i)) /
                static_cast<float>(head_dim);
            const float inverse_frequency =
                1.0f / std::pow(theta, exponent);
            const float angle =
                static_cast<float>(position) *
                inverse_frequency;
            const float c = std::cos(angle);
            const float s = std::sin(angle);

            const float a = values[base + i];
            const float b =
                values[base + half + i];

            values[base + i] =
                a * c - b * s;
            values[base + half + i] =
                b * c + a * s;
        }
    }
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

void Gemma3Model::reset_cache() {
    for (auto& layer : kv_cache_) {
        layer.keys.clear();
        layer.values.clear();
    }
    next_position_ = 0;
}

Gemma3SingleTokenResult Gemma3Model::decode_token(
    std::uint32_t token_id,
    std::uint32_t position,
    std::size_t top_k) {

    if (position != next_position_) {
        throw std::runtime_error(
            "Gemma 3 decode positions must be contiguous");
    }
    if (position >= config_.context_length) {
        throw std::runtime_error(
            "Gemma 3 context length exceeded");
    }

    try {
        std::vector<float> hidden =
            token_embedding(token_id);

        for (std::uint32_t layer = 0;
             layer < config_.block_count;
             ++layer) {

            const auto attn_input =
                rms_norm(
                    hidden,
                    layer_tensor(
                        layer,
                        "attn_norm.weight"));

            auto query =
                run_q8_matvec(
                    layer_tensor(
                        layer,
                        "attn_q.weight"),
                    attn_input);
            auto key =
                run_q8_matvec(
                    layer_tensor(
                        layer,
                        "attn_k.weight"),
                    attn_input);
            const auto value =
                run_q8_matvec(
                    layer_tensor(
                        layer,
                        "attn_v.weight"),
                    attn_input);

            query =
                rms_norm_heads(
                    query,
                    config_.head_count,
                    layer_tensor(
                        layer,
                        "attn_q_norm.weight"));
            key =
                rms_norm_heads(
                    key,
                    config_.head_count_kv,
                    layer_tensor(
                        layer,
                        "attn_k_norm.weight"));

            const float rope_base =
                is_global_layer(layer)
                    ? config_.global_rope_base
                    : config_.local_rope_base;

            apply_rope(
                query,
                config_.head_count,
                config_.head_dim,
                position,
                rope_base);
            apply_rope(
                key,
                config_.head_count_kv,
                config_.head_dim,
                position,
                rope_base);

            auto& cache = kv_cache_[layer];
            cache.keys.push_back(std::move(key));
            cache.values.push_back(value);

            if (!is_global_layer(layer) &&
                config_.sliding_window != 0 &&
                cache.keys.size() >
                    config_.sliding_window) {
                cache.keys.erase(cache.keys.begin());
                cache.values.erase(cache.values.begin());
            }

            if (cache.keys.size() !=
                    cache.values.size() ||
                cache.keys.empty()) {
                throw std::runtime_error(
                    "Gemma 3 KV cache state is invalid");
            }

            const std::size_t history =
                cache.keys.size();
            const float attention_scale =
                1.0f /
                std::sqrt(
                    static_cast<float>(
                        config_.head_dim));

            std::vector<float> attention(
                static_cast<std::size_t>(
                    config_.head_count) *
                config_.value_dim,
                0.0f);

            std::vector<float> scores(history);

            for (std::uint32_t head = 0;
                 head < config_.head_count;
                 ++head) {
                const std::size_t q_base =
                    static_cast<std::size_t>(head) *
                    config_.head_dim;

                float max_score =
                    -std::numeric_limits<float>::infinity();

                for (std::size_t t = 0;
                     t < history;
                     ++t) {
                    double dot = 0.0;
                    const auto& cached_key =
                        cache.keys[t];

                    for (std::uint32_t d = 0;
                         d < config_.head_dim;
                         ++d) {
                        dot +=
                            static_cast<double>(
                                query[q_base + d]) *
                            static_cast<double>(
                                cached_key[d]);
                    }

                    scores[t] =
                        static_cast<float>(dot) *
                        attention_scale;
                    max_score =
                        std::max(
                            max_score,
                            scores[t]);
                }

                double denominator = 0.0;
                for (std::size_t t = 0;
                     t < history;
                     ++t) {
                    const float exp_value =
                        std::exp(
                            scores[t] -
                            max_score);
                    scores[t] = exp_value;
                    denominator += exp_value;
                }

                if (!(denominator > 0.0) ||
                    !std::isfinite(denominator)) {
                    throw std::runtime_error(
                        "Gemma 3 attention softmax failed");
                }

                const std::size_t out_base =
                    static_cast<std::size_t>(head) *
                    config_.value_dim;

                for (std::size_t t = 0;
                     t < history;
                     ++t) {
                    const float probability =
                        static_cast<float>(
                            scores[t] /
                            denominator);
                    const auto& cached_value =
                        cache.values[t];

                    for (std::uint32_t d = 0;
                         d < config_.value_dim;
                         ++d) {
                        attention[out_base + d] +=
                            probability *
                            cached_value[d];
                    }
                }
            }

            auto attention_output =
                run_q8_matvec(
                    layer_tensor(
                        layer,
                        "attn_output.weight"),
                    attention);

            attention_output =
                rms_norm(
                    attention_output,
                    layer_tensor(
                        layer,
                        "post_attention_norm.weight"));

            add_inplace(
                attention_output,
                hidden);
            hidden =
                std::move(attention_output);

            const auto ffn_input =
                rms_norm(
                    hidden,
                    layer_tensor(
                        layer,
                        "ffn_norm.weight"));

            auto gate =
                run_q8_matvec(
                    layer_tensor(
                        layer,
                        "ffn_gate.weight"),
                    ffn_input);
            auto up =
                run_q8_matvec(
                    layer_tensor(
                        layer,
                        "ffn_up.weight"),
                    ffn_input);

            if (gate.size() != up.size() ||
                gate.size() !=
                    config_.feed_forward_length) {
                throw std::runtime_error(
                    "Gemma 3 FFN projection width mismatch");
            }

            for (std::size_t i = 0;
                 i < gate.size();
                 ++i) {
                gate[i] =
                    gelu_tanh(gate[i]) *
                    up[i];
            }

            auto ffn_output =
                run_q8_matvec(
                    layer_tensor(
                        layer,
                        "ffn_down.weight"),
                    gate);

            ffn_output =
                rms_norm(
                    ffn_output,
                    layer_tensor(
                        layer,
                        "post_ffw_norm.weight"));

            add_inplace(ffn_output, hidden);
            hidden =
                std::move(ffn_output);

            for (const float value_out : hidden) {
                if (!std::isfinite(value_out)) {
                    throw std::runtime_error(
                        "Non-finite value after Gemma 3 block " +
                        std::to_string(layer));
                }
            }
        }

        hidden =
            rms_norm(
                hidden,
                "output_norm.weight");

        Gemma3SingleTokenResult result{};
        result.hidden = hidden;
        result.top_tokens =
            top_logits(hidden, top_k);

        ++next_position_;
        return result;
    } catch (...) {
        reset_cache();
        throw;
    }
}

Gemma3SingleTokenResult Gemma3Model::run_single_token(
    std::uint32_t token_id,
    std::size_t top_k) {

    reset_cache();
    auto result =
        decode_token(token_id, 0, top_k);
    reset_cache();
    return result;
}

Gemma3GenerationResult Gemma3Model::generate_greedy_from_bos(
    std::size_t max_new_tokens) {

    reset_cache();

    Gemma3GenerationResult generated{};
    std::uint32_t current_token = 2;

    for (std::size_t i = 0;
         i < max_new_tokens;
         ++i) {
        const auto decoded =
            decode_token(
                current_token,
                static_cast<std::uint32_t>(i),
                1);

        if (decoded.top_tokens.empty()) {
            throw std::runtime_error(
                "Gemma 3 greedy sampler received no logits");
        }

        const std::uint32_t next_token =
            decoded.top_tokens.front().token_id;

        generated.token_ids.push_back(next_token);

        if (next_token == 1) {
            break;
        }

        generated.text +=
            decode_piece(next_token);
        current_token = next_token;
    }

    return generated;
}

} // namespace vortexrt
