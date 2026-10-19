CC = gcc
CFLAGS = -O2 -Wall -Wextra

all: vfio_dmabuf_reset vfio_bar_bench

vfio_dmabuf_reset: LDLIBS = -libverbs
vfio_dmabuf_reset: vfio_dmabuf_reset.c

vfio_bar_bench: CFLAGS += -std=gnu11
vfio_bar_bench: LDLIBS = -lpthread -lm
vfio_bar_bench: vfio_bar_bench.c

clean:
	rm -f vfio_dmabuf_reset vfio_bar_bench

.PHONY: all clean
