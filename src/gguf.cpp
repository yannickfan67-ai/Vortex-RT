#include "vortexrt/gguf.hpp"

#include <bit>
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

GgufFile::GgufFile(const std::filesystem::path& path) {
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
    }

    data_offset_ = align_up(reader.position(), alignment_);
    validate();
}

const GgufValue* GgufFile::find_metadata(const std::string& key) const noexcept {
    for (const auto& kv : metadata_) {
        if (kv.key == key) {
            return &kv.value;
        }
    }
    return nullptr;
}

const GgufTensorInfo* GgufFile::find_tensor(const std::string& name) const noexcept {
    for (const auto& tensor : tensors_) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

std::optional<std::string> GgufFile::metadata_string(const std::string& key) const {
    const auto* value = find_metadata(key);
    return value ? value->as_string() : std::nullopt;
}

std::optional<std::uint64_t> GgufFile::metadata_u64(const std::string& key) const {
    const auto* value = find_metadata(key);
    return value ? value->as_u64() : std::nullopt;
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
