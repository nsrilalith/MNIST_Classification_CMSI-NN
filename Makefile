# Compiler and flags
CC = gcc
CFLAGS = -O3 -Wall -I./include -I./CMSIS-NN-6.0.0/Include -I./CMSIS-NN-6.0.0/Include/Internal

# Source and object files
SRC_DIR = src
OBJ_DIR = obj

# Add CMSIS-NN source directories
CMSIS_SRC_DIRS = $(wildcard CMSIS-NN-6.0.0/Source/*)
CMSIS_SRCS = $(foreach dir, $(CMSIS_SRC_DIRS), $(wildcard $(dir)/*.c))

SRCS = $(wildcard $(SRC_DIR)/*.c) $(CMSIS_SRCS)
OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.o)

# Output binary
TARGET = IMG_CNN

# Create object directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)/$(SRC_DIR)
	@for dir in $(CMSIS_SRC_DIRS); do mkdir -p $(OBJ_DIR)/$$dir; done

# Build rules
all: $(OBJ_DIR) $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "Build complete: $(TARGET)"

$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
	@echo "Clean complete"

# Include dependency files
-include $(OBJS:.o=.d)
