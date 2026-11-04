#pragma once

#include <cstdint>
#include <limits>

namespace ftl {

using Lpn = uint32_t;      // logical page number (host view)
using Ppn = uint32_t;      // physical page number = block * pages_per_block + offset
using BlockId = uint32_t;
using Time = uint64_t;     // logical time: number of host page writes so far

constexpr Lpn kInvalidLpn = std::numeric_limits<Lpn>::max();
constexpr Ppn kInvalidPpn = std::numeric_limits<Ppn>::max();
constexpr BlockId kInvalidBlock = std::numeric_limits<BlockId>::max();
constexpr Time kNever = std::numeric_limits<Time>::max();

}  // namespace ftl
