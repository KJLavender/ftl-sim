#include "map_cache.h"

#include <stdexcept>

namespace ftl {

MapCache::MapCache(uint32_t entries_per_page, uint64_t capacity_pages)
    : per_page_(entries_per_page), capacity_(capacity_pages) {
  if (per_page_ == 0 || capacity_ == 0) {
    throw std::invalid_argument("map cache needs at least one page of one entry");
  }
  where_.reserve(capacity_ * 2);
}

void MapCache::access(uint64_t lpn, bool dirty) {
  ++lookups_;
  const uint64_t page = lpn / per_page_;
  auto it = where_.find(page);
  if (it != where_.end()) {
    lru_.splice(lru_.begin(), lru_, it->second);
    it->second->dirty |= dirty;
    return;
  }
  ++misses_;
  if (lru_.size() >= capacity_) {
    const Slot& victim = lru_.back();
    if (victim.dirty) ++writebacks_;
    where_.erase(victim.page);
    lru_.pop_back();
  }
  lru_.push_front({page, dirty});
  where_[page] = lru_.begin();
}

}  // namespace ftl
