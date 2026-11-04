#include "nand.h"

#include <stdexcept>

namespace ftl {

Nand::Nand(const Geometry& geo)
    : geo_(geo),
      blocks_(geo.num_blocks),
      page_state_(geo.physical_pages(), PageState::kFree),
      page_lpn_(geo.physical_pages(), kInvalidLpn),
      page_tag_(geo.physical_pages(), 0),
      page_time_(geo.physical_pages(), 0) {}

Ppn Nand::program(BlockId b, Lpn lpn, uint64_t tag, Time data_time) {
  BlockInfo& blk = blocks_[b];
  if (blk.state == BlockState::kClosed || blk.write_ptr >= ppb()) {
    throw std::logic_error("program on a full block");
  }
  blk.state = BlockState::kOpen;
  Ppn ppn = b * ppb() + blk.write_ptr;
  if (page_state_[ppn] != PageState::kFree) {
    throw std::logic_error("program on a non-free page");
  }
  page_state_[ppn] = PageState::kValid;
  page_lpn_[ppn] = lpn;
  page_tag_[ppn] = tag;
  page_time_[ppn] = data_time;
  if (data_time > blk.max_data_time) blk.max_data_time = data_time;
  ++blk.write_ptr;
  ++blk.valid;
  ++programs_;
  if (blk.write_ptr == ppb()) blk.state = BlockState::kClosed;
  return ppn;
}

void Nand::invalidate(Ppn ppn) {
  if (page_state_[ppn] != PageState::kValid) {
    throw std::logic_error("invalidate on a non-valid page");
  }
  page_state_[ppn] = PageState::kInvalid;
  --blocks_[block_of(ppn)].valid;
}

void Nand::erase(BlockId b) {
  BlockInfo& blk = blocks_[b];
  if (blk.valid != 0) throw std::logic_error("erase on a block with valid pages");
  Ppn first = b * ppb();
  for (uint32_t i = 0; i < ppb(); ++i) {
    page_state_[first + i] = PageState::kFree;
    page_lpn_[first + i] = kInvalidLpn;
  }
  uint32_t ec = blk.erase_count + 1;
  blk = BlockInfo{};
  blk.erase_count = ec;
  ++erases_;
}

}  // namespace ftl
