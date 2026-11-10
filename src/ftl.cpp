#include "ftl.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ftl {

Ftl::Ftl(const FtlConfig& cfg)
    : Ftl(cfg, make_victim_policy(cfg.gc_policy), make_placement(cfg.placement)) {}

Ftl::Ftl(const FtlConfig& cfg, std::unique_ptr<VictimPolicy> victim,
         std::unique_ptr<Placement> placement)
    : cfg_(cfg),
      nand_(cfg.geo),
      victim_(std::move(victim)),
      placement_(std::move(placement)) {
  const uint64_t physical = cfg.geo.physical_pages();
  logical_pages_ = logical_pages_for(cfg);

  // Static WL gets one extra internal stream for relocated cold data.
  const uint32_t streams = placement_->num_streams() + (cfg.static_wl_delta ? 1 : 0);
  wl_stream_ = cfg.static_wl_delta ? static_cast<int>(placement_->num_streams()) : -1;
  // Each stream may need a fresh block while one victim is being migrated.
  gc_threshold_ = cfg.gc_free_threshold ? cfg.gc_free_threshold : streams + 2;
  const uint64_t reserved =
      static_cast<uint64_t>(gc_threshold_ + cfg.gc_soft_extra + streams) * cfg.geo.pages_per_block;
  if (logical_pages_ + reserved > physical) {
    throw std::invalid_argument(
        "over-provisioning too small for " + std::to_string(streams) +
        " streams: need at least " + std::to_string(gc_threshold_ + streams) +
        " spare blocks");
  }

  l2p_.assign(logical_pages_, kInvalidPpn);
  if (cfg.verify) shadow_.assign(logical_pages_, 0);
  open_.assign(streams, kInvalidBlock);
  for (BlockId b = 0; b < nand_.num_blocks(); ++b) free_.push_back(b);
  placement_->init(logical_pages_);
  block_last_prog_.assign(nand_.num_blocks(), 0);

  if (cfg.map_cache_ratio > 0) {
    // 4-byte entries, one flash page per map page.
    const uint32_t per_page = cfg.geo.page_size / 4;
    const uint64_t map_pages = (logical_pages_ + per_page - 1) / per_page;
    const uint64_t cached = std::max<uint64_t>(
        1, static_cast<uint64_t>(static_cast<double>(map_pages) * cfg.map_cache_ratio + 0.5));
    map_ = std::make_unique<MapCache>(per_page, cached);
  }
  if (cfg.latency.load > 0) lat_ = std::make_unique<LatencyModel>(cfg.latency);
}

void Ftl::map_access(Lpn lpn, bool dirty) {
  if (!map_) return;
  const uint64_t reads = map_->misses(), writes = map_->writebacks();
  map_->access(lpn, dirty);
  ++stats_.map_lookups;
  stats_.map_reads += map_->misses() - reads;
  stats_.map_writes += map_->writebacks() - writes;
}

uint64_t Ftl::logical_pages_for(const FtlConfig& cfg) {
  return static_cast<uint64_t>(cfg.geo.physical_pages() / (1.0 + cfg.op_ratio));
}

void Ftl::write(Lpn lpn, uint32_t req_pages, uint32_t time_s) {
  if (lpn >= logical_pages_) throw std::out_of_range("write lpn out of range");
  ++now_;
  ++stats_.host_writes;
  const Stats before = stats_;
  map_access(lpn, true);

  Ppn old = l2p_[lpn];
  Time prev_time = old == kInvalidPpn ? kNever : nand_.page_data_time(old);
  int stream = placement_->host_stream({lpn, now_, prev_time, req_pages, time_s});
  // Invalidate first so GC triggered by this write never copies stale data.
  unmap(lpn);
  if (cfg_.gc_soft_extra) pace_gc();

  uint64_t tag = ++tag_seq_;
  l2p_[lpn] = append(stream, lpn, tag, now_);
  if (cfg_.verify) shadow_[lpn] = tag;
  if (lat_) {
    // Everything GC (and the map cache) did inside this write delays it.
    lat_->on_write(stats_.gc_copies + stats_.wl_copies - before.gc_copies - before.wl_copies,
                   stats_.erases - before.erases, stats_.map_reads - before.map_reads,
                   stats_.map_writes - before.map_writes);
  }

  if (cfg_.checkpoint_interval && ++writes_since_cp_ >= cfg_.checkpoint_interval) {
    // Worst case for recovery: power fails just before the next checkpoint.
    const Rebuild r = rebuild_check();
    ++stats_.rebuilds;
    stats_.rebuild_pages += r.pages;
    stats_.rebuild_pages_max = std::max(stats_.rebuild_pages_max, r.pages);
    stats_.rebuild_mismatches += r.mismatches;
    checkpoint();
    writes_since_cp_ = 0;
  }
}

void Ftl::checkpoint() {
  cp_ppn_ = l2p_;
  cp_tag_.assign(logical_pages_, 0);
  for (Lpn l = 0; l < logical_pages_; ++l) {
    if (l2p_[l] != kInvalidPpn) cp_tag_[l] = nand_.page_tag(l2p_[l]);
  }
  cp_prog_ = prog_seq_;
  have_cp_ = true;
  ++stats_.checkpoints;
  // The whole table: 4-byte entries, page_size / 4 per map page.
  const uint64_t per_page = cfg_.geo.page_size / 4;
  stats_.checkpoint_writes += (logical_pages_ + per_page - 1) / per_page;
}

Ftl::Rebuild Ftl::rebuild_check() const {
  Rebuild r;
  std::vector<Ppn> l2p = have_cp_ ? cp_ppn_ : std::vector<Ppn>(logical_pages_, kInvalidPpn);
  std::vector<uint64_t> tag = have_cp_ ? cp_tag_ : std::vector<uint64_t>(logical_pages_, 0);
  // A GC copy keeps its version tag; while the victim isn't erased yet (paced
  // GC) both copies are on flash, and the one in the later block is live.
  std::vector<uint64_t> seq(logical_pages_, 0);
  for (BlockId b = 0; b < nand_.num_blocks(); ++b) {
    if (have_cp_ && block_last_prog_[b] <= cp_prog_) continue;  // unchanged since
    if (nand_.block(b).state == BlockState::kFree) continue;
    ++r.blocks;
    const Ppn first = static_cast<Ppn>(b) * nand_.ppb();
    for (uint32_t i = 0; i < nand_.ppb(); ++i) {
      const Ppn p = first + i;
      if (nand_.page_state(p) == PageState::kFree) break;  // write pointer
      ++r.pages;
      const Lpn lpn = nand_.page_lpn(p);
      const uint64_t t = nand_.page_tag(p);
      if (l2p[lpn] == kInvalidPpn || t > tag[lpn] ||
          (t == tag[lpn] && block_last_prog_[b] >= seq[lpn])) {
        l2p[lpn] = p;
        tag[lpn] = t;
        seq[lpn] = block_last_prog_[b];
      }
    }
  }
  for (Lpn l = 0; l < logical_pages_; ++l) {
    if (l2p[l] == l2p_[l]) continue;
    if (l2p_[l] == kInvalidPpn) {
      ++r.resurrected;
    } else {
      ++r.mismatches;
    }
  }
  return r;
}

uint64_t Ftl::programmed_pages() const {
  uint64_t n = 0;
  const uint64_t total = static_cast<uint64_t>(nand_.num_blocks()) * nand_.ppb();
  for (uint64_t p = 0; p < total; ++p) n += nand_.page_state(static_cast<Ppn>(p)) != PageState::kFree;
  return n;
}

void Ftl::read(Lpn lpn) {
  if (lpn >= logical_pages_) throw std::out_of_range("read lpn out of range");
  ++stats_.host_reads;
  map_access(lpn, false);
  Ppn ppn = l2p_[lpn];
  if (ppn == kInvalidPpn) ++stats_.unmapped_reads;
  if (!cfg_.verify) return;

  uint64_t expect = shadow_[lpn];
  if (ppn == kInvalidPpn) {
    if (expect != 0) throw std::logic_error("shadow: mapped data lost, lpn " + std::to_string(lpn));
    return;
  }
  if (nand_.page_state(ppn) != PageState::kValid || nand_.page_lpn(ppn) != lpn ||
      nand_.page_tag(ppn) != expect) {
    throw std::logic_error("shadow: stale or foreign data, lpn " + std::to_string(lpn));
  }
}

void Ftl::trim(Lpn lpn) {
  if (lpn >= logical_pages_) throw std::out_of_range("trim lpn out of range");
  ++stats_.host_trims;
  map_access(lpn, true);
  unmap(lpn);
  if (cfg_.verify) shadow_[lpn] = 0;
}

void Ftl::unmap(Lpn lpn) {
  Ppn old = l2p_[lpn];
  if (old == kInvalidPpn) return;
  nand_.invalidate(old);
  l2p_[lpn] = kInvalidPpn;
}

Ppn Ftl::append(int stream, Lpn lpn, uint64_t tag, Time data_time) {
  // open_ only ever holds blocks with free pages; full blocks are dropped
  // right after their last program, so a GC victim can never be in open_.
  BlockId& b = open_[stream];
  // GC may run here and may itself open a block for this same stream.
  if (b == kInvalidBlock && !in_gc_) ensure_free_space();
  if (b == kInvalidBlock) {
    b = allocate_block(stream == wl_stream_);
    BlockInfo& blk = nand_.block_mut(b);
    blk.stream = stream;
    blk.open_time = now_;
    blk.open_seq = ++open_seq_;
  }
  ++stats_.nand_writes;
  Ppn ppn = nand_.program(b, lpn, tag, data_time);
  block_last_prog_[b] = ++prog_seq_;
  if (nand_.block(b).state == BlockState::kClosed) b = kInvalidBlock;
  return ppn;
}

BlockId Ftl::allocate_block(bool most_worn) {
  if (free_.empty()) throw std::runtime_error("out of free blocks");
  auto it = free_.begin();
  if (most_worn) {
    // Cold data parks on the most-worn block so it rests there.
    it = std::max_element(free_.begin(), free_.end(), [&](BlockId a, BlockId b) {
      return nand_.block(a).erase_count < nand_.block(b).erase_count;
    });
  } else if (cfg_.wear_aware_alloc) {
    it = std::min_element(free_.begin(), free_.end(), [&](BlockId a, BlockId b) {
      return nand_.block(a).erase_count < nand_.block(b).erase_count;
    });
  }
  BlockId b = *it;
  free_.erase(it);
  return b;
}

void Ftl::ensure_free_space() {
  in_gc_ = true;
  // A fully valid victim (possible with FIFO) frees nothing but is legal;
  // only give up if GC keeps making no progress.
  uint32_t no_progress = 0;
  while (free_.size() < gc_threshold_) {
    size_t before = free_.size();
    collect_one();
    no_progress = free_.size() > before ? 0 : no_progress + 1;
    if (no_progress > nand_.num_blocks()) {
      throw std::runtime_error("GC makes no progress (device full?)");
    }
    if (cfg_.static_wl_delta && ++gc_since_wl_check_ >= cfg_.static_wl_period) {
      gc_since_wl_check_ = 0;
      maybe_static_wear_level();
    }
  }
  in_gc_ = false;
}

void Ftl::collect_one() {
  BlockId victim = victim_->select(nand_, now_);
  if (victim == kInvalidBlock) throw std::runtime_error("GC found no closed block");
  const BlockInfo& vblk = nand_.block(victim);
  placement_->on_victim(vblk.stream, vblk.open_time, now_);
  stats_.gc_copies += relocate_and_erase(victim);
  ++stats_.gc_count;
}

void Ftl::pace_gc() {
  if (pace_victim_ == kInvalidBlock) {
    if (free_.size() >= gc_threshold_ + cfg_.gc_soft_extra) return;
    pace_victim_ = victim_->select(nand_, now_);
    if (pace_victim_ == kInvalidBlock) return;
    const BlockInfo& v = nand_.block(pace_victim_);
    placement_->on_victim(v.stream, v.open_time, now_);
    ++stats_.gc_count;
    pace_pos_ = 0;
    // Reclaiming the victim gives (ppb - valid) pages for host writes, so
    // valid / (ppb - valid) copies per host write keep pace; take twice that.
    const uint32_t ppb = nand_.ppb(), valid = v.valid;
    pace_budget_ = valid >= ppb ? ppb : 2 * ((valid + (ppb - valid) - 1) / (ppb - valid)) + 1;
  }
  in_gc_ = true;
  const BlockId victim = pace_victim_;
  const int victim_stream = nand_.block(victim).stream;
  const Ppn first = victim * nand_.ppb();
  uint32_t moved = 0;
  for (; pace_pos_ < nand_.ppb() && moved < pace_budget_; ++pace_pos_) {
    const Ppn src = first + pace_pos_;
    if (nand_.page_state(src) != PageState::kValid) continue;
    const Lpn lpn = nand_.page_lpn(src);
    const Time data_time = nand_.page_data_time(src);
    const int stream = placement_->gc_stream(lpn, now_, data_time, victim_stream);
    const Ppn dst = append(stream, lpn, nand_.page_tag(src), data_time);
    nand_.invalidate(src);
    l2p_[lpn] = dst;
    map_access(lpn, true);
    ++moved;
  }
  stats_.gc_copies += moved;
  if (pace_pos_ >= nand_.ppb()) {
    nand_.erase(victim);
    free_.push_back(victim);
    ++stats_.erases;
    pace_victim_ = kInvalidBlock;
  }
  in_gc_ = false;
}

void Ftl::maybe_static_wear_level() {
  // Cold data pins a block at a low erase count; when the spread exceeds
  // delta, move that data out so the young block re-enters circulation.
  uint32_t max_ec = 0, min_ec = UINT32_MAX;
  BlockId coldest = kInvalidBlock;
  for (BlockId b = 0; b < nand_.num_blocks(); ++b) {
    const BlockInfo& blk = nand_.block(b);
    max_ec = std::max(max_ec, blk.erase_count);
    if (blk.state == BlockState::kClosed && blk.erase_count < min_ec) {
      min_ec = blk.erase_count;
      coldest = b;
    }
  }
  if (coldest == kInvalidBlock || max_ec - min_ec <= cfg_.static_wl_delta) return;
  stats_.wl_copies += relocate_and_erase(coldest, wl_stream_);
  ++stats_.wl_count;
}

uint32_t Ftl::relocate_and_erase(BlockId victim, int forced_stream) {
  if (victim == pace_victim_) pace_victim_ = kInvalidBlock;  // a burst GC finishes it
  const int victim_stream = nand_.block(victim).stream;
  const Ppn first = victim * nand_.ppb();
  uint32_t copies = 0;
  for (uint32_t i = 0; i < nand_.ppb(); ++i) {
    Ppn src = first + i;
    if (nand_.page_state(src) != PageState::kValid) continue;
    Lpn lpn = nand_.page_lpn(src);
    Time data_time = nand_.page_data_time(src);
    int stream = forced_stream >= 0
                     ? forced_stream
                     : placement_->gc_stream(lpn, now_, data_time, victim_stream);
    Ppn dst = append(stream, lpn, nand_.page_tag(src), data_time);
    nand_.invalidate(src);
    l2p_[lpn] = dst;
    map_access(lpn, true);  // the moved page's entry changes too
    ++copies;
  }
  nand_.erase(victim);
  free_.push_back(victim);
  ++stats_.erases;
  return copies;
}

double Ftl::effective_alpha() const {
  // Free count oscillates between threshold-1 and threshold.
  double unusable_blocks = (gc_threshold_ + cfg_.gc_soft_extra - 0.5) + 0.5 * open_.size();
  double usable = static_cast<double>(nand_.geo().physical_pages()) -
                  unusable_blocks * nand_.ppb();
  return usable / logical_pages_;
}

void Ftl::check_invariants() const {
  auto fail = [](const std::string& msg) { throw std::logic_error("invariant: " + msg); };
  std::vector<uint32_t> valid(nand_.num_blocks(), 0);
  uint64_t mapped = 0;
  for (Lpn lpn = 0; lpn < logical_pages_; ++lpn) {
    Ppn ppn = l2p_[lpn];
    if (ppn == kInvalidPpn) continue;
    ++mapped;
    if (nand_.page_state(ppn) != PageState::kValid) fail("L2P points to non-valid page");
    if (nand_.page_lpn(ppn) != lpn) fail("OOB lpn mismatch");
    ++valid[nand_.block_of(ppn)];
  }
  uint64_t total_valid = 0;
  for (BlockId b = 0; b < nand_.num_blocks(); ++b) {
    const BlockInfo& blk = nand_.block(b);
    if (blk.valid != valid[b]) fail("block valid count mismatch at " + std::to_string(b));
    total_valid += blk.valid;
  }
  if (total_valid != mapped) fail("valid pages != mapped LPNs");
  for (BlockId b : free_) {
    if (nand_.block(b).state != BlockState::kFree) fail("free list holds non-free block");
  }
  for (BlockId b : open_) {
    if (b != kInvalidBlock && nand_.block(b).state != BlockState::kOpen) {
      fail("open_ holds a block that is not open");
    }
  }
  size_t open_blocks = 0, free_blocks = 0;
  for (BlockId b = 0; b < nand_.num_blocks(); ++b) {
    if (nand_.block(b).state == BlockState::kOpen) ++open_blocks;
    if (nand_.block(b).state == BlockState::kFree) ++free_blocks;
  }
  size_t tracked_open = 0;
  for (BlockId b : open_) tracked_open += b != kInvalidBlock;
  if (open_blocks != tracked_open) fail("leaked open block");
  if (free_blocks != free_.size()) fail("leaked free block");
}

}  // namespace ftl
