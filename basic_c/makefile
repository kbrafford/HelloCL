HOST_INFO    := $(shell ./detect_platform.py)
HOST_OS      = $(word 1,$(HOST_INFO))
HOST_REGSIZE = $(word 2,$(HOST_INFO))

CC = gcc

CFLAGS =-std=c99 -O2
CFLAGS_darwin_32bit =
CFLAGS_darwin_64bit =
CFLAGS_linux2_32bit =
CFLAGS_linux2_64bit =
CFLAGS +=$(CFLAGS_$(HOST_OS)_$(HOST_REGSIZE))

LFLAGS =
LFLAGS_darwin_32bit =-framework OpenCL
LFLAGS_darwin_64bit =-framework OpenCL
LFLAGS_linux2_32bit =-lOpenCL -lm -L/opt/AMDAPPSDK-2.9-1/lib/x86/
LFLAGS_linux2_64bit =-lOpenCL -lm -L/opt/AMDAPPSDK-2.9-1/lib/x86_64/ 
LFLAGS += $(LFLAGS_$(HOST_OS)_$(HOST_REGSIZE))

SRCS = hellocl.c

OBJS = $(SRCS:.c=.o)

# define the executable
MAIN = hellocl

all : $(MAIN)
	@echo "Program hellocl has been generated"

$(MAIN): $(OBJS)
	$(CC) -o $(MAIN) $(OBJS)$(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS)-c $< -o $@

clean:
	$(RM) *.o $(MAIN)

