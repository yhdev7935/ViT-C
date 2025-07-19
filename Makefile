# Makefile for the Pure C ViT Project

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude -lm

# Source files
SRCS = $(wildcard src/*.c)

# Object files
OBJS = $(SRCS:.c=.o)

# Executable name
TARGET = vit_inference

# Default rule
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(CFLAGS)

# Rule to compile source files into object files
%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

# Rule to run the executable
run: all
	./$(TARGET)

# Rule to clean up the project directory
clean:
	rm -f src/*.o $(TARGET)

# Phony targets
.PHONY: all run clean 