#include "vortexrt/gemma3.hpp"

#include "vortexrt/buffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
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
    const std::string& q8_matvec_spirv,
    const std::string& q8_matvec_u8_spirv,
    const std::string& argmax_spirv,
    const std::string& q8_matvec_32_spirv,
    const std::string& q8_matvec_u8_32_spirv,
    const std::string& gelu_mul_spirv,
    const std::string& q8_matvec_16_spirv,
    const std::string& q8_matvec_u8_16_spirv,
    const std::string& q8_matvec_8_spirv,
    const std::string& q8_matvec_u8_8_spirv,
    std::uint32_t q8_lane_override)
    : gguf_(gguf_path),
      context_(context),
      q8_pipeline_(
          context,
          q8_matvec_spirv,
          q8_matvec_u8_spirv,
          gelu_mul_spirv) {

    fuse_projections_ =
        context_.capabilities().device_type !=
        VK_PHYSICAL_DEVICE_TYPE_CPU;

    const auto subgroup_size =
        context_.capabilities().subgroup_size;

    std::uint32_t selected_lanes =
        q8_lane_override;

    if (selected_lanes == 0) {
        switch (subgroup_size) {
        case 8u:
            selected_lanes = 8u;
            break;
        case 16u:
            selected_lanes = 16u;
            break;
        case 32u:
            selected_lanes = 32u;
            break;
        default:
            selected_lanes = 64u;
            break;
        }
    }

    const auto make_narrow_pipeline =
        [&](std::uint32_t lanes,
            const std::string& spirv,
            const std::string& u8_spirv) {
            if (spirv.empty()) {
                throw std::runtime_error(
                    "Requested Q8 lane variant is unavailable");
            }
            q8_pipeline_narrow_ =
                std::make_unique<Q8MatVecPipeline>(
                    context_,
                    spirv,
                    u8_spirv,
                    gelu_mul_spirv);
            narrow_q8_lane_count_ =
                lanes;
        };

    switch (selected_lanes) {
    case 64u:
        narrow_q8_lane_count_ = 64u;
        break;
    case 32u:
        make_narrow_pipeline(
            32u,
            q8_matvec_32_spirv,
            q8_matvec_u8_32_spirv);
        break;
    case 16u:
        make_narrow_pipeline(
            16u,
            q8_matvec_16_spirv,
            q8_matvec_u8_16_spirv);
        break;
    case 8u:
        make_narrow_pipeline(
            8u,
            q8_matvec_8_spirv,
            q8_matvec_u8_8_spirv);
        break;
    default:
        throw std::runtime_error(
            "Q8 lane override must be 0, 8, 16, 32, or 64");
    }

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

    const auto cache_f32 = [&](const std::string& name) {
        const auto* tensor = require_tensor(gguf_, name);
        f32_weights_.emplace(
            name,
            gguf_.read_f32_tensor(*tensor));
    };

    cache_f32("output_norm.weight");

    constexpr const char* kNormSuffixes[] = {
        "attn_norm.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "post_attention_norm.weight",
        "ffn_norm.weight",
        "post_ffw_norm.weight",
    };

    for (std::uint32_t layer = 0;
         layer < config_.block_count;
         ++layer) {
        for (const char* suffix : kNormSuffixes) {
            cache_f32(
                layer_tensor(
                    layer,
                    suffix));
        }
    }

    const std::uint64_t tensor_data_bytes =
        gguf_.file_size() - gguf_.data_offset();

    if (tensor_data_bytes == 0 ||
        tensor_data_bytes >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Current Q8 shader requires GGUF tensor data below 4 GiB");
    }

    weights_arena_ = std::make_unique<Buffer>(
        context_,
        static_cast<VkDeviceSize>(tensor_data_bytes),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Upload the GGUF tensor-data section once.  Chunking keeps the
    // persistent host-visible Vulkan staging arena modest instead of
    // growing it to the full model size.
    std::ifstream model_file(
        gguf_path,
        std::ios::binary);
    if (!model_file) {
        throw std::runtime_error(
            "Failed to reopen GGUF for weight arena upload");
    }

    model_file.seekg(
        static_cast<std::streamoff>(gguf_.data_offset()),
        std::ios::beg);
    if (!model_file) {
        throw std::runtime_error(
            "Failed to seek to GGUF tensor data");
    }

    constexpr std::size_t kUploadChunk =
        32u * 1024u * 1024u;
    std::vector<std::byte> upload_chunk(kUploadChunk);

    std::uint64_t uploaded = 0;
    while (uploaded < tensor_data_bytes) {
        const auto remaining =
            tensor_data_bytes - uploaded;
        const std::size_t bytes =
            static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    remaining,
                    kUploadChunk));

        model_file.read(
            reinterpret_cast<char*>(
                upload_chunk.data()),
            static_cast<std::streamsize>(bytes));
        if (model_file.gcount() !=
            static_cast<std::streamsize>(bytes)) {
            throw std::runtime_error(
                "Short read while uploading GGUF weight arena");
        }

        weights_arena_->upload(
            upload_chunk.data(),
            bytes,
            static_cast<VkDeviceSize>(uploaded));
        uploaded += bytes;
    }

    const std::uint32_t max_input_elements =
        std::max({
            config_.embedding_length,
            config_.feed_forward_length,
            config_.head_count * config_.head_dim,
        });

    activation_input_ = std::make_unique<Buffer>(
        context_,
        static_cast<VkDeviceSize>(max_input_elements) *
            sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    activation_output_ = std::make_unique<Buffer>(
        context_,
        static_cast<VkDeviceSize>(config_.vocab_size) *
            sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    staging_input_ = std::make_unique<Buffer>(
        context_,
        static_cast<VkDeviceSize>(max_input_elements) *
            sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    staging_output_ = std::make_unique<Buffer>(
        context_,
        static_cast<VkDeviceSize>(config_.vocab_size) *
            sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (!argmax_spirv.empty()) {
        argmax_pipeline_ =
            std::make_unique<ArgmaxPipeline>(
                context_,
                argmax_spirv);
        argmax_output_ =
            std::make_unique<Buffer>(
                context_,
                sizeof(ArgmaxResult),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
}

Q8MatVecPipeline& Gemma3Model::q8_pipeline_for(
    std::size_t input_dim) noexcept {

    // 640-wide and 1024-wide Gemma projections contain only
    // 20 and 32 Q8 blocks respectively.  On subgroup-32-or-
    // smaller devices the 32-lane variant avoids wasting half
    // or more of a 64-lane workgroup.  The 2048-wide FFN down
    // projection keeps the 64-lane path.
    if (q8_pipeline_narrow_ &&
        input_dim <= 1024u) {
        return *q8_pipeline_narrow_;
    }

    return q8_pipeline_;
}

std::size_t Gemma3Model::run_q8_matvec_to_output(
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
            std::numeric_limits<std::uint32_t>::max() ||
        tensor->offset >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Unsupported Q8_0 matrix layout: " + tensor_name);
    }

    if (!weights_arena_) {
        throw std::runtime_error(
            "Gemma 3 weight arena is not initialized");
    }

    q8_pipeline_for(input.size()).run_staged(
        *weights_arena_,
        *activation_input_,
        *activation_output_,
        *staging_input_,
        *staging_output_,
        input.data(),
        input.size() * sizeof(float),
        nullptr,
        0,
        static_cast<std::uint32_t>(
            tensor->offset),
        static_cast<std::uint32_t>(
            tensor->dimensions[0]),
        static_cast<std::uint32_t>(
            tensor->dimensions[1]));

    return static_cast<std::size_t>(
        tensor->dimensions[1]);
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
            std::numeric_limits<std::uint32_t>::max() ||
        tensor->offset >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "Unsupported Q8_0 matrix layout: " + tensor_name);
    }

    if (!weights_arena_) {
        throw std::runtime_error(
            "Gemma 3 weight arena is not initialized");
    }

    const auto output_elements =
        static_cast<std::size_t>(
            tensor->dimensions[1]);

    std::vector<float> output(output_elements);

    q8_pipeline_for(input.size()).run_staged(
        *weights_arena_,
        *activation_input_,
        *activation_output_,
        *staging_input_,
        *staging_output_,
        input.data(),
        input.size() * sizeof(float),
        output.data(),
        output.size() * sizeof(float),
        static_cast<std::uint32_t>(
            tensor->offset),
        static_cast<std::uint32_t>(
            tensor->dimensions[0]),
        static_cast<std::uint32_t>(
            tensor->dimensions[1]));

    return output;
}

std::array<std::vector<float>, 2> Gemma3Model::run_q8_pair(
    const std::array<std::string, 2>& tensor_names,
    const std::vector<float>& input) {

    if (!weights_arena_) {
        throw std::runtime_error(
            "Gemma 3 weight arena is not initialized");
    }

    std::array<std::uint32_t, 2> offsets{};
    std::array<std::uint32_t, 2> output_dims{};
    std::size_t total_elements = 0;

    for (std::size_t i = 0;
         i < tensor_names.size();
         ++i) {

        const auto* tensor =
            require_tensor(
                gguf_,
                tensor_names[i]);

        if (tensor->type != 8 ||
            tensor->dimensions.size() != 2 ||
            tensor->dimensions[0] != input.size() ||
            tensor->dimensions[0] >
                std::numeric_limits<std::uint32_t>::max() ||
            tensor->dimensions[1] >
                std::numeric_limits<std::uint32_t>::max() ||
            tensor->offset >
                std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "Unsupported Q8_0 matrix layout: " +
                tensor_names[i]);
        }

        offsets[i] =
            static_cast<std::uint32_t>(
                tensor->offset);
        output_dims[i] =
            static_cast<std::uint32_t>(
                tensor->dimensions[1]);

        total_elements +=
            static_cast<std::size_t>(
                output_dims[i]);
    }

    std::vector<float> packed(total_elements);

    q8_pipeline_for(input.size()).run_staged_pair(
        *weights_arena_,
        *activation_input_,
        *activation_output_,
        *staging_input_,
        *staging_output_,
        input.data(),
        input.size() * sizeof(float),
        packed.data(),
        packed.size() * sizeof(float),
        offsets,
        static_cast<std::uint32_t>(
            input.size()),
        output_dims);

    std::array<std::vector<float>, 2> result{};
    std::size_t cursor = 0;

    for (std::size_t i = 0;
         i < result.size();
         ++i) {

        const std::size_t count =
            output_dims[i];

        result[i].assign(
            packed.begin() +
                static_cast<std::ptrdiff_t>(
                    cursor),
            packed.begin() +
                static_cast<std::ptrdiff_t>(
                    cursor + count));

        cursor += count;
    }

    return result;
}

std::array<std::vector<float>, 3> Gemma3Model::run_q8_triplet(
    const std::array<std::string, 3>& tensor_names,
    const std::vector<float>& input) {

    if (!weights_arena_) {
        throw std::runtime_error(
            "Gemma 3 weight arena is not initialized");
    }

    std::array<std::uint32_t, 3> offsets{};
    std::array<std::uint32_t, 3> output_dims{};
    std::size_t total_elements = 0;

    for (std::size_t i = 0;
         i < tensor_names.size();
         ++i) {

        const auto* tensor =
            require_tensor(
                gguf_,
                tensor_names[i]);

        if (tensor->type != 8 ||
            tensor->dimensions.size() != 2 ||
            tensor->dimensions[0] != input.size() ||
            tensor->dimensions[0] >
                std::numeric_limits<std::uint32_t>::max() ||
            tensor->dimensions[1] >
                std::numeric_limits<std::uint32_t>::max() ||
            tensor->offset >
                std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "Unsupported Q8_0 matrix layout: " +
                tensor_names[i]);
        }

        offsets[i] =
            static_cast<std::uint32_t>(
                tensor->offset);
        output_dims[i] =
            static_cast<std::uint32_t>(
                tensor->dimensions[1]);

        total_elements +=
            static_cast<std::size_t>(
                output_dims[i]);
    }

    std::vector<float> packed(total_elements);

    q8_pipeline_for(input.size()).run_staged_triplet(
        *weights_arena_,
        *activation_input_,
        *activation_output_,
        *staging_input_,
        *staging_output_,
        input.data(),
        input.size() * sizeof(float),
        packed.data(),
        packed.size() * sizeof(float),
        offsets,
        static_cast<std::uint32_t>(
            input.size()),
        output_dims);

    std::array<std::vector<float>, 3> result{};
    std::size_t cursor = 0;

    for (std::size_t i = 0;
         i < result.size();
         ++i) {

        const std::size_t count =
            output_dims[i];

        result[i].assign(
            packed.begin() +
                static_cast<std::ptrdiff_t>(
                    cursor),
            packed.begin() +
                static_cast<std::ptrdiff_t>(
                    cursor + count));

        cursor += count;
    }

    return result;
}

std::vector<float> Gemma3Model::rms_norm(
    const std::vector<float>& input,
    const std::string& weight_name) const {

    const auto cached =
        f32_weights_.find(weight_name);
    if (cached == f32_weights_.end()) {
        throw std::runtime_error(
            "Uncached RMSNorm tensor: " + weight_name);
    }
    const auto& weights = cached->second;

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

    const auto cached =
        f32_weights_.find(weight_name);
    if (cached == f32_weights_.end()) {
        throw std::runtime_error(
            "Uncached Q/K RMSNorm tensor: " + weight_name);
    }
    const auto& weights = cached->second;

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

    if (top_k == 1 &&
        argmax_pipeline_ &&
        argmax_output_) {

        const auto output_elements =
            run_q8_matvec_to_output(
                "token_embd.weight",
                hidden);

        if (output_elements !=
            config_.vocab_size) {
            throw std::runtime_error(
                "Gemma 3 tied output logits have wrong width");
        }

        argmax_pipeline_->run(
            *activation_output_,
            *argmax_output_,
            config_.vocab_size);

        ArgmaxResult reduced{};
        argmax_output_->download(
            &reduced,
            sizeof(reduced));

        if (reduced.index >=
                config_.vocab_size ||
            !std::isfinite(reduced.value)) {
            throw std::runtime_error(
                "GPU argmax produced invalid result");
        }

        return {
            Gemma3TopToken{
                reduced.index,
                reduced.value,
                token_piece(reduced.index),
            }
        };
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

    for (std::size_t i = 0;
         i < top_k;
         ++i) {
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

    const std::string marker = "▁";
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

            std::vector<float> query;
            std::vector<float> key;
            std::vector<float> value;

            if (fuse_projections_) {
                auto qkv =
                    run_q8_triplet(
                        std::array<std::string, 3>{
                            layer_tensor(
                                layer,
                                "attn_q.weight"),
                            layer_tensor(
                                layer,
                                "attn_k.weight"),
                            layer_tensor(
                                layer,
                                "attn_v.weight"),
                        },
                        attn_input);

                query = std::move(qkv[0]);
                key = std::move(qkv[1]);
                value = std::move(qkv[2]);
            } else {
                query =
                    run_q8_matvec(
                        layer_tensor(
                            layer,
                            "attn_q.weight"),
                        attn_input);
                key =
                    run_q8_matvec(
                        layer_tensor(
                            layer,
                            "attn_k.weight"),
                        attn_input);
                value =
                    run_q8_matvec(
                        layer_tensor(
                            layer,
                            "attn_v.weight"),
                        attn_input);
            }

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

            std::vector<float> ffn_output;

            auto& ffn_pipeline =
                q8_pipeline_for(
                    config_.embedding_length);

            if (fuse_projections_ &&
                ffn_pipeline.ffn_available()) {

                const auto gate_name =
                    layer_tensor(
                        layer,
                        "ffn_gate.weight");
                const auto up_name =
                    layer_tensor(
                        layer,
                        "ffn_up.weight");
                const auto down_name =
                    layer_tensor(
                        layer,
                        "ffn_down.weight");

                const auto* gate_tensor =
                    require_tensor(
                        gguf_,
                        gate_name);
                const auto* up_tensor =
                    require_tensor(
                        gguf_,
                        up_name);
                const auto* down_tensor =
                    require_tensor(
                        gguf_,
                        down_name);

                const auto valid_q8_matrix =
                    [](const GgufTensorInfo* tensor,
                       std::uint64_t input_dim,
                       std::uint64_t output_dim) {
                        return
                            tensor != nullptr &&
                            tensor->type == 8 &&
                            tensor->dimensions.size() == 2 &&
                            tensor->dimensions[0] == input_dim &&
                            tensor->dimensions[1] == output_dim &&
                            tensor->offset <=
                                std::numeric_limits<
                                    std::uint32_t>::max();
                    };

                if (!valid_q8_matrix(
                        gate_tensor,
                        config_.embedding_length,
                        config_.feed_forward_length) ||
                    !valid_q8_matrix(
                        up_tensor,
                        config_.embedding_length,
                        config_.feed_forward_length) ||
                    !valid_q8_matrix(
                        down_tensor,
                        config_.feed_forward_length,
                        config_.embedding_length)) {
                    throw std::runtime_error(
                        "Gemma 3 fused FFN tensor layout mismatch");
                }

                ffn_output.resize(
                    config_.embedding_length);

                ffn_pipeline.run_staged_ffn(
                    *weights_arena_,
                    *activation_input_,
                    *activation_output_,
                    *staging_input_,
                    *staging_output_,
                    ffn_input.data(),
                    ffn_input.size() * sizeof(float),
                    ffn_output.data(),
                    ffn_output.size() * sizeof(float),
                    static_cast<std::uint32_t>(
                        gate_tensor->offset),
                    static_cast<std::uint32_t>(
                        up_tensor->offset),
                    static_cast<std::uint32_t>(
                        down_tensor->offset),
                    config_.embedding_length,
                    config_.feed_forward_length,
                    config_.embedding_length);
            } else {
                std::vector<float> gate;
                std::vector<float> up;

                if (fuse_projections_) {
                    auto gate_up =
                        run_q8_pair(
                            std::array<std::string, 2>{
                                layer_tensor(
                                    layer,
                                    "ffn_gate.weight"),
                                layer_tensor(
                                    layer,
                                    "ffn_up.weight"),
                            },
                            ffn_input);

                    gate = std::move(gate_up[0]);
                    up = std::move(gate_up[1]);
                } else {
                    gate =
                        run_q8_matvec(
                            layer_tensor(
                                layer,
                                "ffn_gate.weight"),
                            ffn_input);
                    up =
                        run_q8_matvec(
                            layer_tensor(
                                layer,
                                "ffn_up.weight"),
                            ffn_input);
                }

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

                ffn_output =
                    run_q8_matvec(
                        layer_tensor(
                            layer,
                            "ffn_down.weight"),
                        gate);
            }

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

std::vector<std::uint32_t> Gemma3Model::tokenize(
    const std::string& text,
    bool add_bos) const {

    const auto* tokens =
        gguf_.find_metadata("tokenizer.ggml.tokens");
    const auto* scores =
        gguf_.find_metadata("tokenizer.ggml.scores");
    const auto* types =
        gguf_.find_metadata("tokenizer.ggml.token_type");

    if (tokens == nullptr ||
        scores == nullptr ||
        types == nullptr ||
        !tokens->is_array() ||
        !scores->is_array() ||
        !types->is_array() ||
        tokens->array.size() != config_.vocab_size ||
        scores->array.size() != config_.vocab_size ||
        types->array.size() != config_.vocab_size) {
        throw std::runtime_error(
            "Gemma 3 GGUF tokenizer metadata is incomplete");
    }

    struct TokenEntry {
        std::uint32_t id = 0;
        float score = 0.0f;
    };

    std::unordered_map<std::string, TokenEntry> pieces;
    pieces.reserve(config_.vocab_size);

    std::vector<std::uint32_t> byte_tokens(
        256,
        std::numeric_limits<std::uint32_t>::max());

    std::size_t max_piece_bytes = 0;

    const auto hex_value = [](char ch) -> int {
        if (ch >= '0' && ch <= '9') {
            return ch - '0';
        }
        if (ch >= 'a' && ch <= 'f') {
            return ch - 'a' + 10;
        }
        if (ch >= 'A' && ch <= 'F') {
            return ch - 'A' + 10;
        }
        return -1;
    };

    for (std::uint32_t id = 0;
         id < config_.vocab_size;
         ++id) {
        const auto piece =
            tokens->array[id].as_string();
        const auto score_value =
            scores->array[id].as_f64();
        const auto type_value =
            types->array[id].as_i64();

        if (!piece || !score_value || !type_value) {
            continue;
        }

        const int type =
            static_cast<int>(*type_value);

        if (type == 6 &&
            piece->size() == 6 &&
            (*piece)[0] == '<' &&
            (*piece)[1] == '0' &&
            (*piece)[2] == 'x' &&
            (*piece)[5] == '>') {
            const int hi =
                hex_value((*piece)[3]);
            const int lo =
                hex_value((*piece)[4]);
            if (hi >= 0 && lo >= 0) {
                byte_tokens[
                    static_cast<std::size_t>(
                        (hi << 4) | lo)] = id;
            }
            continue;
        }

        if (type != 1 && type != 4) {
            continue;
        }

        const float score =
            static_cast<float>(*score_value);
        const auto existing =
            pieces.find(*piece);
        if (existing == pieces.end() ||
            score > existing->second.score) {
            pieces[*piece] =
                TokenEntry{id, score};
        }

        max_piece_bytes =
            std::max(
                max_piece_bytes,
                piece->size());
    }

    bool add_space_prefix = false;
    if (const auto* setting =
            gguf_.find_metadata(
                "tokenizer.ggml.add_space_prefix");
        setting != nullptr) {
        add_space_prefix =
            setting->as_bool().value_or(false);
    }

    const std::string marker = "▁";
    std::string normalized;
    normalized.reserve(
        text.size() + marker.size());

    bool previous_space = false;
    for (const unsigned char ch : text) {
        const bool is_space =
            ch == ' ' ||
            ch == '\t' ||
            ch == '\r' ||
            ch == '\n';

        if (is_space) {
            if (!previous_space) {
                normalized += marker;
                previous_space = true;
            }
        } else {
            normalized.push_back(
                static_cast<char>(ch));
            previous_space = false;
        }
    }

    if (add_space_prefix &&
        !normalized.empty() &&
        normalized.rfind(marker, 0) != 0) {
        normalized.insert(0, marker);
    }

    const float negative_infinity =
        -std::numeric_limits<float>::infinity();

    std::vector<float> best(
        normalized.size() + 1,
        negative_infinity);
    std::vector<std::size_t> previous(
        normalized.size() + 1,
        std::numeric_limits<std::size_t>::max());
    std::vector<std::uint32_t> previous_token(
        normalized.size() + 1,
        std::numeric_limits<std::uint32_t>::max());

    best[0] = 0.0f;

    const auto unk =
        gguf_.metadata_u64(
            "tokenizer.ggml.unknown_token_id")
            .value_or(3);

    for (std::size_t position = 0;
         position < normalized.size();
         ++position) {
        if (!std::isfinite(best[position])) {
            continue;
        }

        bool matched = false;
        const std::size_t limit =
            std::min(
                max_piece_bytes,
                normalized.size() - position);

        for (std::size_t length = 1;
             length <= limit;
             ++length) {
            const auto it =
                pieces.find(
                    normalized.substr(
                        position,
                        length));
            if (it == pieces.end()) {
                continue;
            }

            matched = true;
            const std::size_t next =
                position + length;
            const float candidate =
                best[position] +
                it->second.score;

            if (candidate > best[next]) {
                best[next] = candidate;
                previous[next] = position;
                previous_token[next] =
                    it->second.id;
            }
        }

        if (!matched) {
            const auto byte =
                static_cast<unsigned char>(
                    normalized[position]);
            std::uint32_t token =
                byte_tokens[byte];

            if (token ==
                std::numeric_limits<std::uint32_t>::max()) {
                token =
                    static_cast<std::uint32_t>(
                        std::min<std::uint64_t>(
                            unk,
                            config_.vocab_size - 1u));
            }

            const std::size_t next =
                position + 1;
            const float candidate =
                best[position] - 100.0f;

            if (candidate > best[next]) {
                best[next] = candidate;
                previous[next] = position;
                previous_token[next] = token;
            }
        }
    }

    if (!normalized.empty() &&
        !std::isfinite(best.back())) {
        throw std::runtime_error(
            "Gemma 3 tokenizer could not segment prompt");
    }

    std::vector<std::uint32_t> reversed;
    std::size_t cursor = normalized.size();

    while (cursor != 0) {
        if (previous[cursor] ==
                std::numeric_limits<std::size_t>::max() ||
            previous_token[cursor] ==
                std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "Gemma 3 tokenizer backtracking failed");
        }

        reversed.push_back(
            previous_token[cursor]);
        cursor = previous[cursor];
    }

    std::reverse(
        reversed.begin(),
        reversed.end());

    std::vector<std::uint32_t> output;
    output.reserve(
        reversed.size() + (add_bos ? 1u : 0u));

    if (add_bos) {
        const auto bos =
            gguf_.metadata_u64(
                "tokenizer.ggml.bos_token_id")
                .value_or(2);
        if (bos >= config_.vocab_size) {
            throw std::runtime_error(
                "Gemma 3 BOS token id is out of range");
        }
        output.push_back(
            static_cast<std::uint32_t>(bos));
    }

    output.insert(
        output.end(),
        reversed.begin(),
        reversed.end());

    return output;
}

Gemma3GenerationResult Gemma3Model::generate_greedy(
    const std::string& prompt,
    std::size_t max_new_tokens) {

    reset_cache();

    Gemma3GenerationResult generated{};
    if (max_new_tokens == 0) {
        return generated;
    }

    const auto prompt_tokens =
        tokenize(prompt, true);
    if (prompt_tokens.empty()) {
        throw std::runtime_error(
            "Gemma 3 prompt produced no input tokens");
    }

    Gemma3SingleTokenResult decoded{};
    std::uint32_t position = 0;

    for (std::size_t i = 0;
         i < prompt_tokens.size();
         ++i) {
        const bool last =
            i + 1 == prompt_tokens.size();

        decoded =
            decode_token(
                prompt_tokens[i],
                position++,
                last ? 1u : 0u);
    }

    const auto eos =
        gguf_.metadata_u64(
            "tokenizer.ggml.eos_token_id")
            .value_or(1);

    std::uint32_t current_token = 0;

    for (std::size_t i = 0;
         i < max_new_tokens;
         ++i) {
        if (i != 0) {
            decoded =
                decode_token(
                    current_token,
                    position++,
                    1);
        }

        if (decoded.top_tokens.empty()) {
            throw std::runtime_error(
                "Gemma 3 greedy sampler received no logits");
        }

        current_token =
            decoded.top_tokens.front().token_id;
        generated.token_ids.push_back(
            current_token);

        if (current_token == eos) {
            break;
        }

        generated.text +=
            decode_piece(current_token);
    }

    return generated;
}

Gemma3GenerationResult Gemma3Model::generate_greedy_from_bos(
    std::size_t max_new_tokens) {

    return generate_greedy(
        std::string{},
        max_new_tokens);
}

} // namespace vortexrt
