#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>

namespace ftl {

// Cached mapping table for a DRAM-less controller (DFTL-style). The full
// L2P table lives in flash as map pages, each holding the entries of
// entries_per_page consecutive LPNs. Only `capacity` map pages fit in the
// controller's SRAM / host memory buffer; they are kept in LRU order.
//   miss      -> read the map page from flash
//   eviction  -> write it back to flash if it was changed while cached
class MapCache {
 public:
  MapCache(uint32_t entries_per_page, uint64_t capacity_pages);

  // Look up (and, if dirty, update) the entry of lpn.
  void access(uint64_t lpn, bool dirty);

  uint64_t lookups() const { return lookups_; }
  uint64_t misses() const { return misses_; }          // map page reads
  uint64_t writebacks() const { return writebacks_; }  // map page writes
  uint64_t capacity_pages() const { return capacity_; }
  uint32_t entries_per_page() const { return per_page_; }
  void reset_stats() { lookups_ = misses_ = writebacks_ = 0; }

 private:
  struct Slot {
    uint64_t page;
    bool dirty;
  };
  uint32_t per_page_;
  uint64_t capacity_;
  std::list<Slot> lru_;  // front = most recently used
  std::unordered_map<uint64_t, std::list<Slot>::iterator> where_;
  uint64_t lookups_ = 0, misses_ = 0, writebacks_ = 0;
};

}  // namespace ftl
