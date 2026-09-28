# GEIF - Geometric Extended Isolation Forest
# High-performance C17 Makefile

CC ?= gcc
AR ?= ar
PREFIX ?= /usr/local

CFLAGS ?= -std=c17 -O3 -Wall -Wextra -Wpedantic -Wstrict-prototypes -Wmissing-prototypes -D_GNU_SOURCE -MMD -MP
OPTFLAGS = -march=native -mavx2 -flto

INCLUDES = -Iinclude -Isrc/lib
JSON_CFLAGS = $(shell pkg-config --cflags json-c 2>/dev/null)
JSON_LIBS   = $(shell pkg-config --libs json-c 2>/dev/null || echo "-ljson-c")
LIBS = -lm $(JSON_LIBS)

LIB_SRCS = src/lib/error.c \
           src/lib/forest.c \
           src/lib/reservoir.c \
           src/lib/train.c \
           src/lib/evaluate.c \
           src/lib/json_io.c \
           src/lib/ensemble.c

CLI_SRCS = src/cli/main.c \
           src/cli/xmalloc.c \
           src/cli/columns.c \
           src/cli/template.c

LIB_OBJS = $(LIB_SRCS:.c=.o)
CLI_OBJS = $(CLI_SRCS:.c=.o)

BIN_DIR = bin
LIB_DIR = lib

STATIC_LIB = $(LIB_DIR)/libgeif.a
SHARED_LIB = $(LIB_DIR)/libgeif.so
CLI_BIN    = $(BIN_DIR)/geif

.PHONY: all clean test dirs

all: dirs $(STATIC_LIB) $(SHARED_LIB) $(CLI_BIN)

dirs:
	@mkdir -p $(BIN_DIR) $(LIB_DIR)

src/lib/%.o: src/lib/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDES) $(JSON_CFLAGS) -fPIC -c $< -o $@

src/cli/%.o: src/cli/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDES) $(JSON_CFLAGS) -c $< -o $@

$(STATIC_LIB): $(LIB_OBJS) | dirs
	$(AR) rcs $@ $^

$(SHARED_LIB): $(LIB_OBJS) | dirs
	$(CC) -shared $(OPTFLAGS) -o $@ $^ $(LIBS)

$(CLI_BIN): $(CLI_OBJS) $(STATIC_LIB) | dirs
	$(CC) $(CFLAGS) $(OPTFLAGS) $^ $(LIBS) -o $@

test: $(STATIC_LIB) $(CLI_BIN)
	@mkdir -p test/bin
	$(CC) $(CFLAGS) $(INCLUDES) test/test_voronoi.c $(STATIC_LIB) $(LIBS) -o test/bin/test_voronoi
	$(CC) $(CFLAGS) $(INCLUDES) test/test_stadium.c $(STATIC_LIB) $(LIBS) -o test/bin/test_stadium
	$(CC) $(CFLAGS) $(INCLUDES) test/test_negative.c $(STATIC_LIB) $(LIBS) -o test/bin/test_negative
	@echo "--- Running Voronoi Bisector Test ---"
	@./test/bin/test_voronoi
	@echo "--- Running Stadium Outer Space Test ---"
	@./test/bin/test_stadium
	@echo "--- Running Negative Values & Translation Invariance Test ---"
	@./test/bin/test_negative
	@echo "--- Running Feature 1: CSV Delimiters & Pipe Streaming Test ---"
	@./test/test_cli_csv.sh
	@echo "--- Running Feature 2: Column Selection & Filtering Test ---"
	@./test/test_cli_columns.sh
	@echo "--- Running Feature 3: Multi-Category Sub-Forest Engine Test ---"
	@./test/test_cli_categories.sh
	@echo "--- Running Feature 4: Category Filtering, Lifecycle & Templating Test ---"
	@./test/test_cli_lifecycle.sh
	@echo "--- Running Feature 5: Age & Decay Rate Processing Test ---"
	@./test/test_cli_decay.sh
	@echo "--- Running Feature 6: Custom Output Templating & Attribution Test ---"
	@./test/test_cli_attribution.sh
	@echo "All unit tests passed successfully!"

clean:
	rm -rf $(LIB_OBJS) $(CLI_OBJS) $(LIB_OBJS:.o=.d) $(CLI_OBJS:.o=.d) $(BIN_DIR) $(LIB_DIR) test/bin

-include $(LIB_OBJS:.o=.d) $(CLI_OBJS:.o=.d)
