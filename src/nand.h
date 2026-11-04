#pragma once

#include <vector>

#include "config.h"
#include "types.h"

namespace ftl {

enum class PageState : uint8_t { kFree, kValid, kInvalid };
enum class BlockState : uint8_t { kFree, kOpen, kClosed };

struct BlockInfo {
  BlockState state = BlockState::kFree;
  uint32_t write_ptr = 0;  // next page offset to program
  uint32_t valid = 0;
  uint32_t erase_count = 0;
  int stream = -1;         // write stream that filled this block
  Time open_time = 0;      // logical time when the block was opened
  uint64_t open_seq = 0;   // global open order (strict FIFO tie-break)
  Time max_data_time = 0;  // newest host write time of any page in the block
};

// NAND array: page-granular program, block-granular erase, no in-place update.
// Each page keeps OOB metadata (its LPN) plus simulation-only fields:
// a version tag for the shadow-map check and the original host write time.
class Nand {
 public:
  explicit Nand(const Geometry& geo);

  const Geometry& geo() const { return geo_; }
  uint32_t ppb() const { return geo_.pages_per_block; }
  uint32_t num_blocks() const { return geo_.num_blocks; }

  // Programs the next page of block b. Opens the block if it was free and
  // closes it when full. Returns the programmed physical page.
  Ppn program(BlockId b, Lpn lpn, uint64_t tag, Time data_time);
  void invalidate(Ppn ppn);
  void erase(BlockId b);

  BlockId block_of(Ppn ppn) const { return ppn / geo_.pages_per_block; }
  const BlockInfo& block(BlockId b) const { return blocks_[b]; }
  BlockInfo& block_mut(BlockId b) { return blocks_[b]; }

  PageState page_state(Ppn ppn) const { return page_state_[ppn]; }
  Lpn page_lpn(Ppn ppn) const { return page_lpn_[ppn]; }
  uint64_t page_tag(Ppn ppn) const { return page_tag_[ppn]; }
  Time page_data_time(Ppn ppn) const { return page_time_[ppn]; }

  uint64_t programs() const { return programs_; }
  uint64_t erases() const { return erases_; }

 private:
  Geometry geo_;
  std::vector<BlockInfo> blocks_;
  std::vector<PageState> page_state_;
  std::vector<Lpn> page_lpn_;
  std::vector<uint64_t> page_tag_;
  std::vector<Time> page_time_;
  uint64_t programs_ = 0;
  uint64_t erases_ = 0;
};

}  // namespace ftl
