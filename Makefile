CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
BUILD    := build

LIB_SRCS := src/nand.cpp src/gc_policy.cpp src/placement.cpp src/ftl.cpp \
            src/workload.cpp src/sim.cpp src/predictor.cpp
LIB_OBJS := $(LIB_SRCS:src/%.cpp=$(BUILD)/%.o)

.PHONY: all test clean

all: $(BUILD)/ftlsim $(BUILD)/test_ftl

$(BUILD)/%.o: src/%.cpp src/*.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/ftlsim: $(LIB_OBJS) $(BUILD)/main.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/test_ftl: $(LIB_OBJS) tests/test_ftl.cpp
	$(CXX) $(CXXFLAGS) -Isrc $(LIB_OBJS) tests/test_ftl.cpp -o $@

$(BUILD):
	mkdir -p $(BUILD)

test: $(BUILD)/test_ftl
	./$(BUILD)/test_ftl

clean:
	rm -rf $(BUILD)
