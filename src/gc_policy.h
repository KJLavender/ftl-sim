#pragma once

#include <memory>
#include <string>

#include "nand.h"

namespace ftl {

// Picks the GC victim among closed blocks. Returns kInvalidBlock if none.
class VictimPolicy {
 public:
  virtual ~VictimPolicy() = default;
  virtual std::string name() const = 0;
  virtual BlockId select(const Nand& nand, Time now) const = 0;
};

// Fewest valid pages (least copy work right now).
class GreedyPolicy : public VictimPolicy {
 public:
  std::string name() const override { return "greedy"; }
  BlockId select(const Nand& nand, Time now) const override;
};

// Rosenblum & Ousterhout: maximize (1-u)/(2u) * age, where u = valid fraction
// and age = time since the newest data in the block was written by the host.
class CostBenefitPolicy : public VictimPolicy {
 public:
  std::string name() const override { return "cost-benefit"; }
  BlockId select(const Nand& nand, Time now) const override;
};

// Oldest opened block first. Has a closed-form WAF under uniform random
// writes (Desnoyers, SYSTOR'12), so it is used to validate the simulator.
class FifoPolicy : public VictimPolicy {
 public:
  std::string name() const override { return "fifo"; }
  BlockId select(const Nand& nand, Time now) const override;
};

std::unique_ptr<VictimPolicy> make_victim_policy(const std::string& name);

}  // namespace ftl
