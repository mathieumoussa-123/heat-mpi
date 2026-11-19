#include "heat.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace heat {

void write_binary_field(const std::string& file, const std::vector<double>& field) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                  "raw field format requires IEEE-754 64-bit doubles");
    std::ofstream output(file, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open field output: " + file);
    }
    const std::uint16_t endian_test = 1;
    const bool little_endian = *reinterpret_cast<const std::uint8_t*>(&endian_test) == 1;
    if (little_endian) {
        output.write(reinterpret_cast<const char*>(field.data()),
                     static_cast<std::streamsize>(field.size() * sizeof(double)));
    } else {
        // Explicitly serialize little-endian on a hypothetical big-endian host.
        for (const double number : field) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &number, 8);
            std::array<char, 8> bytes{};
            for (int k = 0; k < 8; ++k) {
                bytes[static_cast<std::size_t>(k)] = static_cast<char>((bits >> (8 * k)) & 0xFFU);
            }
            output.write(bytes.data(), 8);
        }
    }
    if (!output) {
        throw std::runtime_error("failed to write complete field: " + file);
    }
}

} // namespace heat
