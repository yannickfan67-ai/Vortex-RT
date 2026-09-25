#include "vortexrt/gguf.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace vortexrt {
namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747u;
constexpr std::uint64_t kMaxStringBytes = 1ull << 30;
constexpr std::uint64_t kMaxArrayElements = 1ull << 28;
constexpr std::uint32_t kMaxArrayDepth = 8;

class Reader {
public:
    explicit Reader(const std::filesystem::path& path)
        : stream_(path, std::ios::binary) {

        if (!stream_) {
            throw std::runtime_error("Failed to open GGUF file: " + path.string());
        }

        stream_.seekg(0, std::ios::end);
        const auto end = stream_.tellg();
        if (end < 0) {
            throw std::runtime_error("Failed to determine GGUF file size");
        }

        size_ = static_cast<std::uint64_t>(end);
        stream_.seekg(0, std::ios::beg);
    }

    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }

    [[nodiscard]] std::uint64_t position() {
        const auto pos = stream_.tellg();
        if (pos < 0) {
            throw std::runtime_error("Failed to query GGUF stream position");
        }
        return static_cast<std::uint64_t>(pos);
    }

    template <typename T>
    T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        raw(&value, sizeof(T));
        return value;
    }

    void raw(void* dst, std::size_t bytes) {
        if (bytes > std::numeric_limits<std::streamsize>::max()) {
            throw std::runtime_error("GGUF read is too large");
        }

        const auto pos = position();
        if (static_cast<std::uint64_t>(bytes) > size_ - pos) {
            throw std::runtime_error("Unexpected end of GGUF file");
        }

        stream_.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
        if (!stream_) {
            throw std::runtime_error("Failed while reading GGUF file");
        }
    }

    std::string string() {
        const auto length = pod<std::uint64_t>();
        if (length > kMaxStringBytes) {
            throw std::runtime_error("GGUF string length exceeds safety limit");
        }

        std::string value(static_cast<std::size_t>(length), '\0');
        if (length != 0) {
            raw(value.data(), static_cast<std::size_t>(length));
        }
        return value;
    }

private:
    std::ifstream stream_;
    std::uint64_t size_ = 0;
};

GgufValue read_value(Reader& reader, GgufValueType type, std::uint32_t depth);

GgufValue read_array(Reader& reader, std::uint32_t depth) {
    if (depth >= kMaxArrayDepth) {
        throw std::runtime_error("GGUF array nesting is too deep");
    }

    const auto raw_type = reader.pod<std::uint32_t>();
    if (raw_type > static_cast<std::uint32_t>(GgufValueType::Float64)) {
        throw std::runtime_error("GGUF array has invalid element type");
    }

    const auto element_type = static_cast<GgufValueType>(raw_type);
    const auto count = reader.pod<std::uint64_t>();

    if (count > kMaxArrayElements) {
        throw std::runtime_error("GGUF array length exceeds safety limit");
    }

    GgufValue result{};
    result.type = GgufValueType::Array;
    result.array_element_type = element_type;
    result.array.reserve(static_cast<std::size_t>(count));

    for (std::uint64_t i = 0; i < count; ++i) {
        result.array.push_back(read_value(reader, element_type, depth + 1));
    }

    return result;
}

GgufValue read_value(Reader& reader, GgufValueType type, std::uint32_t depth) {
    GgufValue result{};
    result.type = type;

    switch (type) {
        case GgufValueType::UInt8:
            result.scalar = static_cast<std::uint64_t>(reader.pod<std::uint8_t>());
            break;
        case GgufValueType::Int8:
            result.scalar = static_cast<std::int64_t>(reader.pod<std::int8_t>());
            break;
        case GgufValueType::UInt16:
            result.scalar = static_cast<std::uint64_t>(reader.pod<std::uint16_t>());
            break;
        case GgufValueType::Int16:
            result.scalar = static_cast<std::int64_t>(reader.pod<std::int16_t>());
            break;
        case GgufValueType::UInt32:
            result.scalar = static_cast<std::uint64_t>(reader.pod<std::uint32_t>());
            break;
        case GgufValueType::Int32:
            result.scalar = static_cast<std::int64_t>(reader.pod<std::int32_t>());
            break;
        case GgufValueType::Float32:
            result.scalar = static_cast<double>(reader.pod<float>());
            break;
        case GgufValueType::Bool: {
            const auto value = reader.pod<std::uint8_t>();
            if (value > 1) {
                throw std::runtime_error("GGUF boolean value is not 0/1");
            }
            result.scalar = value != 0;
            break;
        }
        case GgufValueType::String:
            result.scalar = reader.string();
            break;
        case GgufValueType::Array:
            return read_array(reader, depth);
        case GgufValueType::UInt64:
            result.scalar = reader.pod<std::uint64_t>();
            break;
        case GgufValueType::Int64:
            result.scalar = reader.pod<std::int64_t>();
            break;
        case GgufValueType::Float64:
            result.scalar = reader.pod<double>();
            break;
    }

    return result;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0) {
        throw std::runtime_error("GGUF alignment must be non-zero");
    }

    const auto remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

struct TypeLayout {
    std::uint64_t block_elements;
    std::uint64_t block_bytes;
};

std::optional<TypeLayout> type_layout(std::uint32_t type) {
    switch (type) {
        case 0:  return TypeLayout{1, 4};   // F32
        case 1:  return TypeLayout{1, 2};   // F16
        case 8:  return TypeLayout{32, 34}; // Q8_0
        case 24: return TypeLayout{1, 1};   // I8
        case 25: return TypeLayout{1, 2};   // I16
        case 26: return TypeLayout{1, 4};   // I32
        case 27: return TypeLayout{1, 8};   // I64
        case 28: return TypeLayout{1, 8};   // F64
        case 30: return TypeLayout{1, 2};   // BF16
        default: return std::nullopt;
    }
}

float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign =
        static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exponent = (h >> 10) & 31u;
    std::uint32_t mantissa = h & 1023u;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= 0x3ffu;
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

std::optional<std::uint64_t> tensor_bytes(const GgufTensorInfo& tensor) {
    const auto layout = type_layout(tensor.type);
    if (!layout) {
        return std::nullopt;
    }

    std::uint64_t elements = 1;
    for (const auto dim : tensor.dimensions) {
        if (dim == 0) {
            return std::uint64_t{0};
        }
        if (elements > std::numeric_limits<std::uint64_t>::max() / dim) {
            throw std::runtime_error("GGUF tensor element count overflow");
        }
        elements *= dim;
    }

    if ((elements % layout->block_elements) != 0) {
        throw std::runtime_error("GGUF tensor element count is not divisible by quant block size");
    }

    const auto blocks = elements / layout->block_elements;
    if (blocks > std::numeric_limits<std::uint64_t>::max() / layout->block_bytes) {
        throw std::runtime_error("GGUF tensor byte size overflow");
    }

    return blocks * layout->block_bytes;
}

} // namespace

std::optional<std::string> GgufValue::as_string() const {
    if (const auto* value = std::get_if<std::string>(&scalar)) {
        return *value;
    }
    return std::nullopt;
}

std::optional<std::uint64_t> GgufValue::as_u64() const {
    if (const auto* value = std::get_if<std::uint64_t>(&scalar)) {
        return *value;
    }
    if (const auto* signed_value = std::get_if<std::int64_t>(&scalar);
        signed_value != nullptr && *signed_value >= 0) {
        return static_cast<std::uint64_t>(*signed_value);
    }
    return std::nullopt;
}

std::optional<std::int64_t> GgufValue::as_i64() const {
    if (const auto* value = std::get_if<std::int64_t>(&scalar)) {
        return *value;
    }
    if (const auto* unsigned_value = std::get_if<std::uint64_t>(&scalar);
        unsigned_value != nullptr &&
        *unsigned_value <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return static_cast<std::int64_t>(*unsigned_value);
    }
    return std::nullopt;
}

std::optional<double> GgufValue::as_f64() const {
    if (const auto* value = std::get_if<double>(&scalar)) {
        return *value;
    }
    return std::nullopt;
}

std::optional<bool> GgufValue::as_bool() const {
    if (const auto* value = std::get_if<bool>(&scalar)) {
        return *value;
    }
    return std::nullopt;
}

GgufFile::GgufFile(const std::filesystem::path& path)
    : path_(path) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("Vortex-RT GGUF reader currently requires little-endian host");
    }

    Reader reader(path);
    file_size_ = reader.size();

    const auto magic = reader.pod<std::uint32_t>();
    if (magic != kGgufMagic) {
        throw std::runtime_error("Invalid GGUF magic");
    }

    version_ = reader.pod<std::uint32_t>();
    if (version_ < 2 || version_ > 3) {
        throw std::runtime_error("Unsupported GGUF version: " + std::to_string(version_));
    }

    const auto tensor_count = reader.pod<std::uint64_t>();
    const auto metadata_count = reader.pod<std::uint64_t>();

    if (tensor_count > 1'000'000ull || metadata_count > 1'000'000ull) {
        throw std::runtime_error("GGUF header counts exceed safety limits");
    }

    metadata_.reserve(static_cast<std::size_t>(metadata_count));

    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        GgufMetadata kv{};
        kv.key = reader.string();

        const auto raw_type = reader.pod<std::uint32_t>();
        if (raw_type > static_cast<std::uint32_t>(GgufValueType::Float64)) {
            throw std::runtime_error("GGUF metadata has invalid value type");
        }

        kv.value = read_value(reader, static_cast<GgufValueType>(raw_type), 0);
        metadata_.push_back(std::move(kv));
        metadata_index_.try_emplace(
            metadata_.back().key,
            metadata_.size() - 1u);
    }

    if (const auto* alignment = find_metadata("general.alignment")) {
        const auto value = alignment->as_u64();
        if (!value || *value == 0 || *value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Invalid general.alignment metadata");
        }
        alignment_ = static_cast<std::uint32_t>(*value);
    }

    if ((alignment_ & (alignment_ - 1u)) != 0u) {
        throw std::runtime_error("GGUF alignment must be a power of two");
    }

    tensors_.reserve(static_cast<std::size_t>(tensor_count));

    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensorInfo tensor{};
        tensor.name = reader.string();

        const auto dimensions = reader.pod<std::uint32_t>();
        if (dimensions == 0 || dimensions > 8) {
            throw std::runtime_error("GGUF tensor has invalid dimension count");
        }

        tensor.dimensions.reserve(dimensions);
        for (std::uint32_t d = 0; d < dimensions; ++d) {
            tensor.dimensions.push_back(reader.pod<std::uint64_t>());
        }

        tensor.type = reader.pod<std::uint32_t>();
        tensor.offset = reader.pod<std::uint64_t>();
        tensors_.push_back(std::move(tensor));
        tensor_index_.try_emplace(
            tensors_.back().name,
            tensors_.size() - 1u);
    }

    data_offset_ = align_up(reader.position(), alignment_);
    validate();

    payload_stream_.open(
        path_,
        std::ios::binary);
    if (!payload_stream_) {
        throw std::runtime_error(
            "Failed to open GGUF payload stream: " +
            path_.string());
    }
}

void GgufFile::read_bytes_at(
    std::uint64_t absolute_offset,
    void* destination,
    std::size_t bytes) const {

    if (bytes == 0) {
        return;
    }
    if (destination == nullptr) {
        throw std::runtime_error(
            "GGUF payload read received null destination");
    }
    if (absolute_offset > file_size_ ||
        bytes >
            file_size_ - absolute_offset ||
        bytes >
            static_cast<std::size_t>(
                std::numeric_limits<
                    std::streamsize>::max())) {
        throw std::runtime_error(
            "GGUF payload read lies outside file");
    }

    std::lock_guard<std::mutex> lock(
        payload_stream_mutex_);

    payload_stream_.clear();
    payload_stream_.seekg(
        static_cast<std::streamoff>(
            absolute_offset),
        std::ios::beg);

    if (!payload_stream_) {
        throw std::runtime_error(
            "Failed to seek GGUF payload stream");
    }

    payload_stream_.read(
        reinterpret_cast<char*>(
            destination),
        static_cast<std::streamsize>(
            bytes));

    if (!payload_stream_) {
        throw std::runtime_error(
            "Failed to read GGUF payload stream");
    }
}

const GgufValue* GgufFile::find_metadata(const std::string& key) const noexcept {
    const auto it =
        metadata_index_.find(key);
    if (it == metadata_index_.end()) {
        return nullptr;
    }
    return &metadata_[it->second].value;
}

const GgufTensorInfo* GgufFile::find_tensor(const std::string& name) const noexcept {
    const auto it =
        tensor_index_.find(name);
    if (it == tensor_index_.end()) {
        return nullptr;
    }
    return &tensors_[it->second];
}

std::optional<std::string> GgufFile::metadata_string(const std::string& key) const {
    const auto* value = find_metadata(key);
    return value ? value->as_string() : std::nullopt;
}

std::optional<std::uint64_t> GgufFile::metadata_u64(const std::string& key) const {
    const auto* value = find_metadata(key);
    return value ? value->as_u64() : std::nullopt;
}

std::optional<std::uint64_t> GgufFile::tensor_byte_size(
    const GgufTensorInfo& tensor) const {

    return tensor_bytes(tensor);
}

std::vector<std::byte> GgufFile::read_tensor_bytes(
    const GgufTensorInfo& tensor) const {

    const auto bytes = tensor_bytes(tensor);
    if (!bytes) {
        throw std::runtime_error(
            "Unsupported GGML tensor type for payload read: " +
            tensor.name);
    }
    if (*bytes > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()) ||
        *bytes > static_cast<std::uint64_t>(
            std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error(
            "Tensor payload is too large to materialize: " +
            tensor.name);
    }

    const std::uint64_t absolute = data_offset_ + tensor.offset;
    if (absolute > file_size_ || *bytes > file_size_ - absolute) {
        throw std::runtime_error(
            "Tensor payload is outside GGUF file: " +
            tensor.name);
    }

    std::vector<std::byte> data(
        static_cast<std::size_t>(
            *bytes));

    read_bytes_at(
        absolute,
        data.data(),
        data.size());

    return data;
}

std::vector<float> GgufFile::read_f32_tensor(
    const GgufTensorInfo& tensor) const {

    if (tensor.type != 0) {
        throw std::runtime_error(
            "Expected F32 tensor: " + tensor.name);
    }

    const auto bytes = read_tensor_bytes(tensor);
    if ((bytes.size() % sizeof(float)) != 0) {
        throw std::runtime_error(
            "F32 tensor byte size is invalid: " + tensor.name);
    }

    std::vector<float> values(bytes.size() / sizeof(float));
    if (!bytes.empty()) {
        std::memcpy(values.data(), bytes.data(), bytes.size());
    }
    return values;
}

std::vector<float> GgufFile::read_q8_0_row(
    const GgufTensorInfo& tensor,
    std::uint64_t row) const {

    if (tensor.type != 8) {
        throw std::runtime_error(
            "Expected Q8_0 tensor: " + tensor.name);
    }
    if (tensor.dimensions.empty() ||
        tensor.dimensions.front() == 0 ||
        (tensor.dimensions.front() % 32u) != 0u) {
        throw std::runtime_error(
            "Q8_0 tensor has unsupported row layout: " +
            tensor.name);
    }

    const std::uint64_t width = tensor.dimensions.front();
    std::uint64_t row_count = 1;
    for (std::size_t i = 1; i < tensor.dimensions.size(); ++i) {
        const auto dim = tensor.dimensions[i];
        if (dim != 0 &&
            row_count >
                std::numeric_limits<std::uint64_t>::max() / dim) {
            throw std::runtime_error(
                "Q8_0 tensor row count overflow: " +
                tensor.name);
        }
        row_count *= dim;
    }
    if (row >= row_count) {
        throw std::runtime_error(
            "Q8_0 row index out of range: " + tensor.name);
    }

    const std::uint64_t blocks = width / 32u;
    if (blocks >
        std::numeric_limits<std::uint64_t>::max() / 34u) {
        throw std::runtime_error(
            "Q8_0 row byte size overflow: " + tensor.name);
    }
    const std::uint64_t row_bytes = blocks * 34u;
    const std::uint64_t row_offset = row * row_bytes;

    const auto total_bytes = tensor_bytes(tensor);
    if (!total_bytes ||
        row_offset > *total_bytes ||
        row_bytes > *total_bytes - row_offset) {
        throw std::runtime_error(
            "Q8_0 row lies outside tensor payload: " +
            tensor.name);
    }

    const std::uint64_t absolute =
        data_offset_ + tensor.offset + row_offset;
    if (absolute > file_size_ ||
        row_bytes > file_size_ - absolute ||
        row_bytes >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error(
            "Q8_0 row lies outside GGUF file: " +
            tensor.name);
    }

    std::vector<std::uint8_t> encoded(
        static_cast<std::size_t>(
            row_bytes));

    read_bytes_at(
        absolute,
        encoded.data(),
        encoded.size());

    std::vector<float> values(static_cast<std::size_t>(width));
    for (std::uint64_t block = 0; block < blocks; ++block) {
        const std::size_t base =
            static_cast<std::size_t>(block * 34u);
        const std::uint16_t scale_bits =
            static_cast<std::uint16_t>(encoded[base]) |
            (static_cast<std::uint16_t>(encoded[base + 1]) << 8u);
        const float scale = f16_to_f32(scale_bits);

        for (std::size_t k = 0; k < 32; ++k) {
            const auto q =
                static_cast<std::int8_t>(encoded[base + 2 + k]);
            values[
                static_cast<std::size_t>(block * 32u) + k] =
                scale * static_cast<float>(q);
        }
    }

    return values;
}

void GgufFile::validate() const {
    if (data_offset_ > file_size_) {
        throw std::runtime_error("GGUF tensor data offset lies beyond end of file");
    }

    for (const auto& tensor : tensors_) {
        if ((tensor.offset % alignment_) != 0) {
            throw std::runtime_error(
                "GGUF tensor offset is not aligned: " + tensor.name);
        }

        if (tensor.offset > file_size_ - data_offset_) {
            throw std::runtime_error(
                "GGUF tensor offset lies beyond end of file: " + tensor.name);
        }

        if (const auto bytes = tensor_bytes(tensor)) {
            const auto absolute = data_offset_ + tensor.offset;
            if (*bytes > file_size_ - absolute) {
                throw std::runtime_error(
                    "GGUF tensor payload lies beyond end of file: " + tensor.name);
            }
        }
    }
}

const char* gguf_value_type_name(GgufValueType type) noexcept {
    switch (type) {
        case GgufValueType::UInt8: return "uint8";
        case GgufValueType::Int8: return "int8";
        case GgufValueType::UInt16: return "uint16";
        case GgufValueType::Int16: return "int16";
        case GgufValueType::UInt32: return "uint32";
        case GgufValueType::Int32: return "int32";
        case GgufValueType::Float32: return "float32";
        case GgufValueType::Bool: return "bool";
        case GgufValueType::String: return "string";
        case GgufValueType::Array: return "array";
        case GgufValueType::UInt64: return "uint64";
        case GgufValueType::Int64: return "int64";
        case GgufValueType::Float64: return "float64";
    }
    return "unknown";
}

const char* ggml_type_name(std::uint32_t type) noexcept {
    switch (type) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 9: return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        case 34: return "TQ1_0";
        case 35: return "TQ2_0";
        case 39: return "MXFP4";
        default: return "UNKNOWN";
    }
}

} // namespace vortexrt
