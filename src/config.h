#pragma once

#include <cstdint>
#include <string>

namespace ftl {

struct Geometry {
  uint32_t page_size = 4096;
  uint32_t pages_per_block = 256;
  uint32_t num_blocks = 1024;  // physical blocks

  uint64_t physical_pages() const {
    return static_cast<uint64_t>(pages_per_block) * num_blocks;
  }
};

struct FtlConfig {
  Geometry geo;
  // Over-provisioning, industry definition: (physical - user) / user.
  // 0.07 means user capacity = physical / 1.07.
  double op_ratio = 0.07;
  std::string gc_policy = "greedy";  // greedy | cost-benefit | fifo
  std::string placement = "none";    // none | sepgc | wcount | sepbit | oracle | fdp
  // GC runs when free blocks drop below this. 0 = auto (enough for every stream).
  uint32_t gc_free_threshold = 0;
  // Dynamic wear leveling: allocate the free block with the lowest erase count.
  bool wear_aware_alloc = false;
  // Static wear leveling: when max - min erase count exceeds delta, relocate
  // the least-erased closed block (cold data). 0 = off. Checked every
  // static_wl_period GC victims.
  uint32_t static_wl_delta = 0;
  uint32_t static_wl_period = 16;
  // Shadow map: verify every read returns the latest written version.
  bool verify = false;
  // DRAM-less mapping: fraction of the L2P table's map pages (4 bytes per
  // entry, one flash page each) that fit in the controller's cache.
  // 0 = the whole table is in DRAM (no map traffic).
  double map_cache_ratio = 0.0;
  // Power-loss recovery: save the L2P table every this many host writes
  // (0 = never). Before each save the FTL simulates a crash, rebuilds the
  // table from the checkpoint plus the OOB of blocks written since, and
  // checks it against the live table.
  uint64_t checkpoint_interval = 0;
};

}  // namespace ftl
