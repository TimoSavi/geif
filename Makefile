# GEIF - Geometric Extended Isolation Forest
# High-performance C17 Makefile

CC ?= gcc
AR ?= ar
PREFIX ?= /usr/local

CFLAGS ?= -std=c17 -O3 -Wall -Wextra -Wpedantic -Wstrict-prototypes -Wmissing-prototypes -D_GNU_SOURCE
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
           src/lib/json_io.c

CLI_SRCS = src/cli/main.c

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

test: $(STATIC_LIB)
	@mkdir -p test/bin
	$(CC) $(CFLAGS) $(INCLUDES) test/test_voronoi.c $(STATIC_LIB) $(LIBS) -o test/bin/test_voronoi
	$(CC) $(CFLAGS) $(INCLUDES) test/test_stadium.c $(STATIC_LIB) $(LIBS) -o test/bin/test_stadium
	@echo "--- Running Voronoi Bisector Test ---"
	@./test/bin/test_voronoi
	@echo "--- Running Stadium Outer Space Test ---"
	@./test/bin/test_stadium
	@echo "All unit tests passed successfully!"

clean:
	rm -rf $(LIB_OBJS) $(CLI_OBJS) $(BIN_DIR) $(LIB_DIR) test/bin
