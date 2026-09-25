#include "vortexrt/gemma3.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace vortexrt {
namespace {

constexpr std::uint32_t kQ8BlockElements = 32;
constexpr std::uint32_t kQ8BlockBytes = 34;
constexpr std::size_t kUploadChunkBytes = 16u * 1024u * 1024u;

std::uint32_t checked_u32(
    std::uint64_t value,
    const std::string& what) {

    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(what + " exceeds uint32 range");
    }
    return static_cast<std::uint32_t>(value);
}

std::uint32_t require_meta_u32(
    const GgufFile& gguf,
    const std::string& key) {

    const auto value = gguf.metadata_u64(key);
    if (!value) {
        throw std::runtime_error("missing integer GGUF metadata: " + key);
    }
    return checked_u32(*value, key);
}

double require_meta_f64(
    const GgufFile& gguf,
    const std::string& key) {

    const auto* value = gguf.find_metadata(key);
    if (value == nullptr) {
        throw std::runtime_error("missing float GGUF metadata: " + key);
    }

    if (const auto f = value->as_f64()) {
        return *f;
    }
    if (const auto u = value->as_u64()) {
        return static_cast<double>(*u);
    }
    if (const auto i = value->as_i64()) {
        return static_cast<double>(*i);
    }

    throw std::runtime_error("GGUF metadata is not numeric: " + key);
}

bool is_special_piece(const std::string& piece) {
    return piece.size() >= 2 &&
           piece.front() == '<' &&
           piece.back() == '>';
}

int hex_digit(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return 10 + c - 'a';
    }
    if (c >= 'A' && c <= 'F') {
        return 10 + c - 'A';
    }
    return -1;
}

std::string byte_token(std::uint8_t byte) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string token = "<0x00>";
    token[3] = kHex[(byte >> 4) & 0x0f];
    token[4] = kHex[byte & 0x0f];
    return token;
}

void replace_all(
    std::string& text,
    const std::string& needle,
    const std::string& replacement) {

    std::size_t position = 0;
    while ((position = text.find(needle, position)) != std::string::npos) {
        text.replace(position, needle.size(), replacement);
        position += replacement.size();
    }
}

} // namespace

Gemma3Model::Gemma3Model(
    VulkanContext& context,
    const std::filesystem::path& model_path,
    const std::string& q8_matvec_spirv)
    : context_(context),
      gguf_(model_path) {

    const auto architecture = gguf_.metadata_string("general.architecture");
    if (architecture != "gemma3") {
        throw std::runtime_error(
            "Vortex-RT Gemma3 runner requires general.architecture=gemma3");
    }

    model_name_ =
        gguf_.metadata_string("general.name").value_or("Gemma 3");

    hidden_size_ =
        require_meta_u32(gguf_, "gemma3.embedding_length");
    intermediate_size_ =
        require_meta_u32(gguf_, "gemma3.feed_forward_length");
    layer_count_ =
        require_meta_u32(gguf_, "gemma3.block_count");
    head_count_ =
        require_meta_u32(gguf_, "gemma3.attention.head_count");
    kv_head_count_ =
        require_meta_u32(gguf_, "gemma3.attention.head_count_kv");
    head_dim_ =
        require_meta_u32(gguf_, "gemma3.attention.key_length");
    sliding_window_ =
        require_meta_u32(gguf_, "gemma3.attention.sliding_window");
    rms_eps_ = static_cast<float>(
        require_meta_f64(
            gguf_,
            "gemma3.attention.layer_norm_rms_epsilon"));
    global_rope_base_ =
        require_meta_f64(gguf_, "gemma3.rope.freq_base");

    if (hidden_size_ != 640 ||
        intermediate_size_ != 2048 ||
        layer_count_ != 18 ||
        head_count_ != 4 ||
        kv_head_count_ != 1 ||
        head_dim_ != 256 ||
        sliding_window_ != 512) {
        throw std::runtime_error(
            "this initial Vortex-RT Gemma3 path is specialized for Gemma 3 270M");
    }

    if (head_count_ * head_dim_ != 1024) {
        throw std::runtime_error("unexpected Gemma3 query projection width");
    }

    if (const auto* tokens = gguf_.find_metadata("tokenizer.ggml.tokens");
        tokens != nullptr && tokens->is_array()) {

        vocab_size_ = checked_u32(
            tokens->array.size(),
            "tokenizer vocabulary size");
        tokens_.reserve(tokens->array.size());

        for (const auto& value : tokens->array) {
            const auto text = value.as_string();
            if (!text) {
                throw std::runtime_error(
                    "tokenizer.ggml.tokens contains non-string value");
            }
            tokens_.push_back(*text);
        }
    } else {
        throw std::runtime_error("GGUF tokenizer token array is missing");
    }

    if (const auto* scores = gguf_.find_metadata("tokenizer.ggml.scores");
        scores != nullptr && scores->is_array()) {

        if (scores->array.size() != tokens_.size()) {
            throw std::runtime_error("tokenizer score count mismatch");
        }

        token_scores_.reserve(scores->array.size());
        for (const auto& value : scores->array) {
            const auto score = value.as_f64();
            if (!score) {
                throw std::runtime_error(
                    "tokenizer.ggml.scores contains non-float value");
            }
            token_scores_.push_back(static_cast<float>(*score));
        }
    } else {
        token_scores_.assign(tokens_.size(), 0.0f);
    }

    if (const auto* types = gguf_.find_metadata("tokenizer.ggml.token_type");
        types != nullptr && types->is_array()) {

        if (types->array.size() != tokens_.size()) {
            throw std::runtime_error("tokenizer type count mismatch");
        }

        token_types_.reserve(types->array.size());
        for (const auto& value : types->array) {
            const auto type = value.as_i64();
            if (!type ||
                *type < std::numeric_limits<std::int32_t>::min() ||
                *type > std::numeric_limits<std::int32_t>::max()) {
                throw std::runtime_error(
                    "tokenizer.ggml.token_type contains invalid value");
            }
            token_types_.push_back(static_cast<std::int32_t>(*type));
        }
    } else {
        token_types_.assign(tokens_.size(), 1);
    }

    bos_token_id_ = require_meta_u32(
        gguf_,
        "tokenizer.ggml.bos_token_id");
    eos_token_id_ = require_meta_u32(
        gguf_,
        "tokenizer.ggml.eos_token_id");
    unk_token_id_ = require_meta_u32(
        gguf_,
        "tokenizer.ggml.unknown_token_id");

    token_to_id_.reserve(tokens_.size() * 2);
    for (std::uint32_t i = 0; i < tokens_.size(); ++i) {
        token_to_id_.emplace(tokens_[i], i);
    }

    std::ifstream model(model_path, std::ios::binary);
    if (!model) {
        throw std::runtime_error(
            "failed to reopen model data: " + model_path.string());
    }

    if (gguf_.data_offset() > gguf_.file_size()) {
        throw std::runtime_error("invalid GGUF data offset");
    }

    const std::uint64_t tensor_bytes =
        gguf_.file_size() - gguf_.data_offset();

    if (tensor_bytes == 0 ||
        tensor_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("invalid GGUF tensor data size");
    }

    tensor_data_.resize(static_cast<std::size_t>(tensor_bytes));
    model.seekg(
        static_cast<std::streamoff>(gguf_.data_offset()),
        std::ios::beg);
    model.read(
        reinterpret_cast<char*>(tensor_data_.data()),
        static_cast<std::streamsize>(tensor_data_.size()));

    if (!model) {
        throw std::runtime_error("failed to read GGUF tensor data");
    }

    token_embedding_ = require_q8_matrix(
        "token_embd.weight",
        hidden_size_,
        vocab_size_);

    output_norm_ = require_f32_vector(
        "output_norm.weight",
        hidden_size_);

    layers_.reserve(layer_count_);
    cache_.resize(layer_count_);

    for (std::uint32_t layer_index = 0;
         layer_index < layer_count_;
         ++layer_index) {

        const std::string prefix =
            "blk." + std::to_string(layer_index) + ".";

        LayerWeights layer{};
        layer.q = require_q8_matrix(
            prefix + "attn_q.weight",
            hidden_size_,
            head_count_ * head_dim_);
        layer.k = require_q8_matrix(
            prefix + "attn_k.weight",
            hidden_size_,
            kv_head_count_ * head_dim_);
        layer.v = require_q8_matrix(
            prefix + "attn_v.weight",
            hidden_size_,
            kv_head_count_ * head_dim_);
        layer.o = require_q8_matrix(
            prefix + "attn_output.weight",
            head_count_ * head_dim_,
            hidden_size_);

        layer.ffn_gate = require_q8_matrix(
            prefix + "ffn_gate.weight",
            hidden_size_,
            intermediate_size_);
        layer.ffn_up = require_q8_matrix(
            prefix + "ffn_up.weight",
            hidden_size_,
            intermediate_size_);
        layer.ffn_down = require_q8_matrix(
            prefix + "ffn_down.weight",
            intermediate_size_,
            hidden_size_);

        layer.attn_norm = require_f32_vector(
            prefix + "attn_norm.weight",
            hidden_size_);
        layer.post_attention_norm = require_f32_vector(
            prefix + "post_attention_norm.weight",
            hidden_size_);
        layer.ffn_norm = require_f32_vector(
            prefix + "ffn_norm.weight",
            hidden_size_);
        layer.post_ffw_norm = require_f32_vector(
            prefix + "post_ffw_norm.weight",
            hidden_size_);
        layer.q_norm = require_f32_vector(
            prefix + "attn_q_norm.weight",
            head_dim_);
        layer.k_norm = require_f32_vector(
            prefix + "attn_k_norm.weight",
            head_dim_);

        layer.sliding = ((layer_index + 1u) % 6u) != 0u;
        layers_.push_back(std::move(layer));
    }

    model_buffer_ = std::make_unique<Buffer>(
        context_,
        tensor_data_.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    for (std::size_t offset = 0;
         offset < tensor_data_.size();
         offset += kUploadChunkBytes) {

        const std::size_t bytes = std::min(
            kUploadChunkBytes,
            tensor_data_.size() - offset);

        model_buffer_->upload(
            tensor_data_.data() + offset,
            bytes,
            static_cast<VkDeviceSize>(offset));
    }

    const std::uint32_t max_input_dim = std::max({
        hidden_size_,
        intermediate_size_,
        head_count_ * head_dim_});

    input_buffer_ = std::make_unique<Buffer>(
        context_,
        static_cast<VkDeviceSize>(max_input_dim) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    output_buffer_ = std::make_unique<Buffer>(
        context_,
        static_cast<VkDeviceSize>(vocab_size_) * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    q8_pipeline_ = std::make_unique<Q8MatVecPipeline>(
        context_,
        q8_matvec_spirv);

    attention_scale_ =
        1.0f / std::sqrt(static_cast<float>(head_dim_));
}

Gemma3Model::TensorRef Gemma3Model::require_q8_matrix(
    const std::string& name,
    std::uint32_t expected_input,
    std::uint32_t expected_output) const {

    const auto* tensor = gguf_.find_tensor(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing GGUF tensor: " + name);
    }

    if (tensor->type != 8 ||
        tensor->dimensions.size() != 2 ||
        tensor->dimensions[0] != expected_input ||
        tensor->dimensions[1] != expected_output) {
        throw std::runtime_error(
            "unexpected Q8_0 tensor shape/type: " + name);
    }

    return TensorRef{
        checked_u32(tensor->offset, name + " offset"),
        expected_input,
        expected_output,
    };
}

std::vector<float> Gemma3Model::require_f32_vector(
    const std::string& name,
    std::uint32_t expected_size) const {

    const auto* tensor = gguf_.find_tensor(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing GGUF tensor: " + name);
    }

    if (tensor->type != 0 ||
        tensor->dimensions.size() != 1 ||
        tensor->dimensions[0] != expected_size) {
        throw std::runtime_error(
            "unexpected F32 tensor shape/type: " + name);
    }

    const std::uint64_t bytes =
        static_cast<std::uint64_t>(expected_size) * sizeof(float);

    if (tensor->offset > tensor_data_.size() ||
        bytes > tensor_data_.size() - tensor->offset) {
        throw std::runtime_error(
            "F32 tensor lies outside model data: " + name);
    }

    std::vector<float> result(expected_size);
    std::memcpy(
        result.data(),
        tensor_data_.data() + tensor->offset,
        static_cast<std::size_t>(bytes));
    return result;
}

float Gemma3Model::half_to_float(std::uint16_t value) {
    const std::uint32_t sign =
        static_cast<std::uint32_t>(value & 0x8000u) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1fu;
    std::uint32_t mantissa = value & 0x03ffu;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= 0x03ffu;
            bits =
                sign |
                (static_cast<std::uint32_t>(e + 127) << 23) |
                (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits =
            sign |
            ((exponent - 15u + 127u) << 23) |
            (mantissa << 13);
    }

    return std::bit_cast<float>(bits);
}

std::vector<float> Gemma3Model::dequant_embedding(
    std::uint32_t token_id) const {

    if (token_id >= vocab_size_) {
        throw std::runtime_error("token id outside vocabulary");
    }

    if ((hidden_size_ % kQ8BlockElements) != 0) {
        throw std::runtime_error("embedding width is not Q8_0 block aligned");
    }

    const std::uint32_t blocks =
        hidden_size_ / kQ8BlockElements;
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(blocks) * kQ8BlockBytes;
    const std::uint64_t row_offset =
        static_cast<std::uint64_t>(token_embedding_.offset) +
        static_cast<std::uint64_t>(token_id) * row_bytes;

    if (row_offset > tensor_data_.size() ||
        row_bytes > tensor_data_.size() - row_offset) {
        throw std::runtime_error("embedding row lies outside model data");
    }

    std::vector<float> result(hidden_size_);
    const auto* row = tensor_data_.data() + row_offset;

    for (std::uint32_t block = 0; block < blocks; ++block) {
        const auto* source =
            row + static_cast<std::size_t>(block) * kQ8BlockBytes;

        std::uint16_t scale_bits = 0;
        std::memcpy(&scale_bits, source, sizeof(scale_bits));
        const float scale = half_to_float(scale_bits);

        for (std::uint32_t i = 0; i < kQ8BlockElements; ++i) {
            const int raw = source[2 + i];
            const int quantized =
                raw >= 128 ? raw - 256 : raw;
            result[block * kQ8BlockElements + i] =
                scale * static_cast<float>(quantized);
        }
    }

    return result;
}

std::vector<float> Gemma3Model::matvec(
    const TensorRef& matrix,
    const std::vector<float>& input) {

    if (input.size() != matrix.input_dim) {
        throw std::runtime_error("Gemma3 matvec input dimension mismatch");
    }

    input_buffer_->upload(
        input.data(),
        input.size() * sizeof(float));

    q8_pipeline_->run(
        *model_buffer_,
        *input_buffer_,
        *output_buffer_,
        matrix.offset,
        matrix.input_dim,
        matrix.output_dim);

    std::vector<float> output(matrix.output_dim);
    output_buffer_->download(
        output.data(),
        output.size() * sizeof(float));

    ++matvec_dispatches_;
    return output;
}

std::vector<float> Gemma3Model::rms_norm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::uint32_t block_size) const {

    if (block_size == 0 ||
        input.empty() ||
        input.size() % block_size != 0 ||
        weight.size() != block_size) {
        throw std::runtime_error("invalid Gemma3 RMSNorm shape");
    }

    std::vector<float> output(input.size());

    for (std::size_t base = 0;
         base < input.size();
         base += block_size) {

        double sum_squares = 0.0;
        for (std::uint32_t i = 0; i < block_size; ++i) {
            const double x = input[base + i];
            sum_squares += x * x;
        }

        const float inv_rms = 1.0f / std::sqrt(
            static_cast<float>(
                sum_squares / static_cast<double>(block_size)) +
            rms_eps_);

        for (std::uint32_t i = 0; i < block_size; ++i) {
            output[base + i] =
                input[base + i] *
                inv_rms *
                weight[i];
        }
    }

    return output;
}

void Gemma3Model::apply_rope(
    std::vector<float>& values,
    std::uint32_t head_count,
    std::uint32_t position,
    double base) const {

    if (values.size() !=
        static_cast<std::size_t>(head_count) * head_dim_) {
        throw std::runtime_error("invalid Gemma3 RoPE shape");
    }

    const std::uint32_t half = head_dim_ / 2;

    for (std::uint32_t head = 0; head < head_count; ++head) {
        const std::size_t head_base =
            static_cast<std::size_t>(head) * head_dim_;

        for (std::uint32_t i = 0; i < half; ++i) {
            const double exponent =
                (2.0 * static_cast<double>(i)) /
                static_cast<double>(head_dim_);
            const double inverse_frequency =
                1.0 / std::pow(base, exponent);
            const double angle =
                static_cast<double>(position) * inverse_frequency;

            const float cosine =
                static_cast<float>(std::cos(angle));
            const float sine =
                static_cast<float>(std::sin(angle));

            const float first = values[head_base + i];
            const float second =
                values[head_base + half + i];

            values[head_base + i] =
                first * cosine - second * sine;
            values[head_base + half + i] =
                second * cosine + first * sine;
        }
    }
}

std::vector<float> Gemma3Model::attend(
    std::uint32_t layer_index,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    std::uint32_t position) {

    if (layer_index >= layers_.size()) {
        throw std::runtime_error("attention layer index out of range");
    }

    if (query.size() !=
            static_cast<std::size_t>(head_count_) * head_dim_ ||
        key.size() != head_dim_ ||
        value.size() != head_dim_) {
        throw std::runtime_error("attention Q/K/V shape mismatch");
    }

    LayerCache& cache = cache_[layer_index];
    cache.keys.insert(cache.keys.end(), key.begin(), key.end());
    cache.values.insert(cache.values.end(), value.begin(), value.end());

    const std::size_t cached_tokens =
        cache.keys.size() / head_dim_;

    if (cached_tokens != static_cast<std::size_t>(position) + 1 ||
        cache.values.size() != cache.keys.size()) {
        throw std::runtime_error("Gemma3 KV cache position mismatch");
    }

    std::size_t begin_token = 0;
    if (layers_[layer_index].sliding &&
        cached_tokens > sliding_window_) {
        begin_token = cached_tokens - sliding_window_;
    }

    std::vector<float> output(
        static_cast<std::size_t>(head_count_) * head_dim_,
        0.0f);

    std::vector<float> scores(
        cached_tokens - begin_token,
        0.0f);

    for (std::uint32_t head = 0; head < head_count_; ++head) {
        float max_score = -std::numeric_limits<float>::infinity();

        for (std::size_t token = begin_token;
             token < cached_tokens;
             ++token) {

            double dot = 0.0;
            const std::size_t key_base = token * head_dim_;
            const std::size_t query_base =
                static_cast<std::size_t>(head) * head_dim_;

            for (std::uint32_t d = 0; d < head_dim_; ++d) {
                dot +=
                    static_cast<double>(query[query_base + d]) *
                    static_cast<double>(cache.keys[key_base + d]);
            }

            const float score =
                static_cast<float>(dot) * attention_scale_;
            scores[token - begin_token] = score;
            max_score = std::max(max_score, score);
        }

        double denominator = 0.0;
        for (float& score : scores) {
            score = std::exp(score - max_score);
            denominator += score;
        }

        if (!(denominator > 0.0) ||
            !std::isfinite(denominator)) {
            throw std::runtime_error("Gemma3 attention softmax failed");
        }

        const std::size_t output_base =
            static_cast<std::size_t>(head) * head_dim_;

        for (std::size_t token = begin_token;
             token < cached_tokens;
             ++token) {

            const float probability =
                static_cast<float>(
                    static_cast<double>(
                        scores[token - begin_token]) /
                    denominator);

            const std::size_t value_base = token * head_dim_;

            for (std::uint32_t d = 0; d < head_dim_; ++d) {
                output[output_base + d] +=
                    probability *
                    cache.values[value_base + d];
            }
        }
    }

    return output;
}

float Gemma3Model::gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoeff = 0.044715f;

    const float x3 = x * x * x;
    return 0.5f * x *
        (1.0f +
         std::tanh(
             kSqrt2OverPi *
             (x + kCoeff * x3)));
}

std::optional<std::uint32_t> Gemma3Model::forward_token(
    std::uint32_t token_id,
    std::uint32_t position,
    bool compute_logits) {

    std::vector<float> hidden = dequant_embedding(token_id);

    const float embedding_scale =
        std::sqrt(static_cast<float>(hidden_size_));
    for (float& value : hidden) {
        value *= embedding_scale;
    }

    for (std::uint32_t layer_index = 0;
         layer_index < layer_count_;
         ++layer_index) {

        const LayerWeights& layer = layers_[layer_index];

        const std::vector<float> residual_attention = hidden;
        const std::vector<float> attention_input =
            rms_norm(
                hidden,
                layer.attn_norm,
                hidden_size_);

        std::vector<float> query =
            matvec(layer.q, attention_input);
        std::vector<float> key =
            matvec(layer.k, attention_input);
        std::vector<float> value =
            matvec(layer.v, attention_input);

        query = rms_norm(
            query,
            layer.q_norm,
            head_dim_);
        key = rms_norm(
            key,
            layer.k_norm,
            head_dim_);

        const double rope_base =
            layer.sliding
                ? local_rope_base_
                : global_rope_base_;

        apply_rope(
            query,
            head_count_,
            position,
            rope_base);
        apply_rope(
            key,
            kv_head_count_,
            position,
            rope_base);

        std::vector<float> attention_output =
            attend(
                layer_index,
                query,
                key,
                value,
                position);

        attention_output =
            matvec(layer.o, attention_output);

        attention_output =
            rms_norm(
                attention_output,
                layer.post_attention_norm,
                hidden_size_);

        hidden.resize(hidden_size_);
        for (std::uint32_t i = 0; i < hidden_size_; ++i) {
            hidden[i] =
                residual_attention[i] +
                attention_output[i];
        }

        const std::vector<float> residual_ffn = hidden;
        const std::vector<float> ffn_input =
            rms_norm(
                hidden,
                layer.ffn_norm,
                hidden_size_);

        std::vector<float> gate =
            matvec(layer.ffn_gate, ffn_input);
        const std::vector<float> up =
            matvec(layer.ffn_up, ffn_input);

        for (std::uint32_t i = 0;
             i < intermediate_size_;
             ++i) {
            gate[i] =
                gelu_tanh(gate[i]) *
                up[i];
        }

        std::vector<float> ffn_output =
            matvec(layer.ffn_down, gate);

        ffn_output =
            rms_norm(
                ffn_output,
                layer.post_ffw_norm,
                hidden_size_);

        for (std::uint32_t i = 0; i < hidden_size_; ++i) {
            hidden[i] =
                residual_ffn[i] +
                ffn_output[i];
        }
    }

    hidden = rms_norm(
        hidden,
        output_norm_,
        hidden_size_);

    if (!compute_logits) {
        return std::nullopt;
    }

    const std::vector<float> logits =
        matvec(token_embedding_, hidden);

    if (logits.size() != vocab_size_) {
        throw std::runtime_error("Gemma3 logits size mismatch");
    }

    const auto best = std::max_element(
        logits.begin(),
        logits.end());

    if (best == logits.end() ||
        !std::isfinite(*best)) {
        throw std::runtime_error("Gemma3 logits are invalid");
    }

    return static_cast<std::uint32_t>(
        std::distance(logits.begin(), best));
}

void Gemma3Model::reset_cache() {
    for (LayerCache& layer : cache_) {
        layer.keys.clear();
        layer.values.clear();
    }
}

bool Gemma3Model::tokenizer_mergeable(
    std::uint32_t token_id) const {

    if (token_id >= token_types_.size()) {
        return false;
    }

    const std::int32_t type = token_types_[token_id];

    // GGML tokenizer token types:
    // 1 normal, 2 unknown, 3 control, 4 user-defined,
    // 5 unused, 6 byte.
    return type == 1 || type == 4;
}

std::string Gemma3Model::sentencepiece_normalize(
    const std::string& text) {

    const std::string space_marker = "\xE2\x96\x81";
    std::string output;
    output.reserve(text.size() + 8);

    for (unsigned char c : text) {
        if (c == ' ') {
            output += space_marker;
        } else {
            output.push_back(static_cast<char>(c));
        }
    }

    return output;
}

std::vector<std::string> Gemma3Model::utf8_symbols(
    const std::string& text) {

    std::vector<std::string> symbols;

    for (std::size_t i = 0; i < text.size();) {
        const unsigned char first =
            static_cast<unsigned char>(text[i]);

        std::size_t length = 1;
        if ((first & 0x80u) == 0) {
            length = 1;
        } else if ((first & 0xe0u) == 0xc0u) {
            length = 2;
        } else if ((first & 0xf0u) == 0xe0u) {
            length = 3;
        } else if ((first & 0xf8u) == 0xf0u) {
            length = 4;
        }

        if (i + length > text.size()) {
            length = 1;
        }

        symbols.emplace_back(text.substr(i, length));
        i += length;
    }

    return symbols;
}

std::vector<std::uint32_t> Gemma3Model::tokenize(
    const std::string& text,
    bool add_bos) const {

    std::vector<std::uint32_t> result;

    if (add_bos) {
        result.push_back(bos_token_id_);
    }

    if (text.empty()) {
        return result;
    }

    std::vector<std::string> symbols =
        utf8_symbols(sentencepiece_normalize(text));

    while (symbols.size() >= 2) {
        float best_score =
            -std::numeric_limits<float>::infinity();
        std::size_t best_index = symbols.size();

        for (std::size_t i = 0;
             i + 1 < symbols.size();
             ++i) {

            const std::string merged =
                symbols[i] + symbols[i + 1];
            const auto found =
                token_to_id_.find(merged);

            if (found == token_to_id_.end() ||
                !tokenizer_mergeable(found->second)) {
                continue;
            }

            const float score =
                token_scores_[found->second];

            if (score > best_score) {
                best_score = score;
                best_index = i;
            }
        }

        if (best_index == symbols.size()) {
            break;
        }

        symbols[best_index] +=
            symbols[best_index + 1];
        symbols.erase(
            symbols.begin() +
            static_cast<std::ptrdiff_t>(best_index + 1));
    }

    for (const std::string& symbol : symbols) {
        const auto found = token_to_id_.find(symbol);

        if (found != token_to_id_.end() &&
            tokenizer_mergeable(found->second)) {
            result.push_back(found->second);
            continue;
        }

        for (unsigned char byte : symbol) {
            const auto byte_found =
                token_to_id_.find(byte_token(byte));

            if (byte_found != token_to_id_.end()) {
                result.push_back(byte_found->second);
            } else {
                result.push_back(unk_token_id_);
            }
        }
    }

    return result;
}

std::string Gemma3Model::decode_token(
    std::uint32_t token_id) const {

    if (token_id >= tokens_.size()) {
        throw std::runtime_error("decode token id outside vocabulary");
    }

    if (token_id == eos_token_id_ ||
        token_id == bos_token_id_) {
        return {};
    }

    const std::string& piece = tokens_[token_id];

    if (piece.size() == 6 &&
        piece.rfind("<0x", 0) == 0 &&
        piece.back() == '>') {

        const int high = hex_digit(piece[3]);
        const int low = hex_digit(piece[4]);

        if (high >= 0 && low >= 0) {
            return std::string(
                1,
                static_cast<char>((high << 4) | low));
        }
    }

    if (token_id < token_types_.size()) {
        const std::int32_t type = token_types_[token_id];
        if (type == 3 || type == 5) {
            return {};
        }
    }

    if (is_special_piece(piece)) {
        return {};
    }

    std::string output = piece;
    replace_all(
        output,
        "\xE2\x96\x81",
        " ");
    return output;
}

std::vector<std::uint32_t> Gemma3Model::generate(
    const std::string& prompt,
    std::uint32_t max_new_tokens,
    Gemma3GenerationStats* stats) {

    if (max_new_tokens == 0) {
        return {};
    }

    const auto started = std::chrono::steady_clock::now();
    const std::uint64_t dispatch_start = matvec_dispatches_;

    reset_cache();

    const std::vector<std::uint32_t> prompt_tokens =
        tokenize(prompt, true);

    if (prompt_tokens.empty()) {
        throw std::runtime_error("Gemma3 prompt produced no tokens");
    }

    std::optional<std::uint32_t> next;

    for (std::size_t i = 0;
         i < prompt_tokens.size();
         ++i) {

        const bool need_logits =
            i + 1 == prompt_tokens.size();

        next = forward_token(
            prompt_tokens[i],
            static_cast<std::uint32_t>(i),
            need_logits);
    }

    if (!next) {
        throw std::runtime_error("Gemma3 failed to produce first logits");
    }

    std::vector<std::uint32_t> generated;
    generated.reserve(max_new_tokens);

    std::uint32_t position =
        static_cast<std::uint32_t>(prompt_tokens.size());

    for (std::uint32_t step = 0;
         step < max_new_tokens;
         ++step) {

        const std::uint32_t token = *next;
        if (token == eos_token_id_) {
            break;
        }

        generated.push_back(token);

        if (step + 1 >= max_new_tokens) {
            break;
        }

        next = forward_token(
            token,
            position,
            true);
        ++position;

        if (!next) {
            throw std::runtime_error("Gemma3 decode produced no logits");
        }
    }

    const auto ended = std::chrono::steady_clock::now();

    if (stats != nullptr) {
        stats->matvec_dispatches =
            matvec_dispatches_ - dispatch_start;
        stats->total_seconds =
            std::chrono::duration<double>(
                ended - started).count();
    }

    return generated;
}

} // namespace vortexrt
