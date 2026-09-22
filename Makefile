CC = gcc
CFLAGS = -O2 -Wall -Wextra
LDLIBS = -libverbs

vfio_dmabuf_reset: vfio_dmabuf_reset.c

clean:
	rm -f vfio_dmabuf_reset

.PHONY: clean
