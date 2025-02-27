CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -ffp-contract=off -Isrc
LDFLAGS  :=

SRC_DIR  := src
TEST_DIR := tests
BIN_DIR  := build

.PHONY: all clean test

all: $(BIN_DIR)/ptxlite $(BIN_DIR)/test_main

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/ptxlite: $(SRC_DIR)/main.cpp $(SRC_DIR)/*.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/main.cpp $(LDFLAGS)

$(BIN_DIR)/test_main: $(TEST_DIR)/test_main.cpp $(SRC_DIR)/*.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_DIR)/test_main.cpp $(LDFLAGS)

test: $(BIN_DIR)/test_main
	./$(BIN_DIR)/test_main

clean:
	rm -rf $(BIN_DIR)
