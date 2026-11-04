#include "gc_policy.h"

#include <stdexcept>

namespace ftl {

BlockId GreedyPolicy::select(const Nand& nand, Time) const {
  BlockId best = kInvalidBlock;
  uint32_t best_valid = nand.ppb() + 1;
  for (BlockId b = 0; b < nand.num_blocks(); ++b) {
    const BlockInfo& blk = nand.block(b);
    if (blk.state != BlockState::kClosed) continue;
    if (blk.valid < best_valid) {
      best_valid = blk.valid;
      best = b;
      if (best_valid == 0) break;
    }
  }
  return best;
}

BlockId CostBenefitPolicy::select(const Nand& nand, Time now) const {
  BlockId best = kInvalidBlock;
  double best_score = -1.0;
  for (BlockId b = 0; b < nand.num_blocks(); ++b) {
    const BlockInfo& blk = nand.block(b);
    if (blk.state != BlockState::kClosed) continue;
    if (blk.valid == 0) return b;  // free reclaim, nothing beats it
    double u = static_cast<double>(blk.valid) / nand.ppb();
    double age = static_cast<double>(now - blk.max_data_time) + 1.0;
    double score = (1.0 - u) / (2.0 * u) * age;
    if (score > best_score) {
      best_score = score;
      best = b;
    }
  }
  return best;
}

BlockId FifoPolicy::select(const Nand& nand, Time) const {
  BlockId best = kInvalidBlock;
  uint64_t oldest = kNever;
  for (BlockId b = 0; b < nand.num_blocks(); ++b) {
    const BlockInfo& blk = nand.block(b);
    if (blk.state != BlockState::kClosed) continue;
    if (blk.open_seq < oldest) {
      oldest = blk.open_seq;
      best = b;
    }
  }
  return best;
}

std::unique_ptr<VictimPolicy> make_victim_policy(const std::string& name) {
  if (name == "greedy") return std::make_unique<GreedyPolicy>();
  if (name == "cost-benefit") return std::make_unique<CostBenefitPolicy>();
  if (name == "fifo") return std::make_unique<FifoPolicy>();
  throw std::invalid_argument("unknown gc policy: " + name);
}

}  // namespace ftl
