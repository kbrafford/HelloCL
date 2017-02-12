
ifeq ($(OS),Windows_NT)
	HOST_OS := win32
	ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
		HOST_REGSIZE := 64bit
	endif
	ifeq ($(PROCESSOR_ARCHITECTURE),x86)
		HOST_REGSIZE := 32bit
	endif
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux2
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := darwin
	endif
	UNAME_P := $(shell uname -p)
	ifeq ($(UNAME_P),x86_64)
		HOST_REGSIZE := 64bit
	endif
	ifneq ($(filter %86,$(UNAME_P)),)
		HOST_REGSIZE := 32bit
	endif
endif

CC = gcc

CFLAGS =-std=c99 -O2
CFLAGS_darwin_32bit =
CFLAGS_darwin_64bit =
CFLAGS_linux2_32bit =
CFLAGS_linux2_64bit =
CFLAGS_win32_32bit = -I"..\tools_intel_win\include" 
CFLAGS_win32_64bit = -I"..\tools_intel_win\include"

CFLAGS +=$(CFLAGS_$(HOST_OS)_$(HOST_REGSIZE))

LFLAGS =
LFLAGS_darwin_32bit =-framework OpenCL
LFLAGS_darwin_64bit =-framework OpenCL
LFLAGS_linux2_32bit =-lOpenCL -lm -L/opt/AMDAPPSDK-2.9-1/lib/x86
LFLAGS_linux2_64bit =-lOpenCL -lm -L/opt/AMDAPPSDK-2.9-1/lib/x86_64
LFLAGS_win32_32bit =-L"..\tools_intel_win\lib\x86" -lOpenCL
LFLAGS_win32_64bit =-L"..\tools_intel_win\lib\x64" -lOpenCL

LFLAGS += $(LFLAGS_$(HOST_OS)_$(HOST_REGSIZE))

SRCS = hellocl.c

OBJS = $(SRCS:.c=.o)

# define the executable
MAIN = hellocl

all : $(MAIN)
	@echo "Program hellocl has been generated"

$(MAIN): $(OBJS)
	$(CC) -o $(MAIN) $(OBJS) $(LFLAGS)

.c.o:
	$(CC) $(CFLAGS)-c $< -o $@

clean:
ifeq ($(OS),Windows_NT)
	$(RM) *.o $(MAIN).exe
else    
	$(RM) *.o $(MAIN)
endif
