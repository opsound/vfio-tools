#define _GNU_SOURCE

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <limits.h>
#include <linux/vfio.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define DMABUF_LEN (2u * 1024u * 1024u)
#define VFIO_CDEV_PREFIX "/dev/vfio/devices/"

static void die(const char *what)
{
	fprintf(stderr, "error: %s: %s\n", what, strerror(errno));
	exit(1);
}

static void die_msg(const char *what)
{
	fprintf(stderr, "error: %s\n", what);
	exit(1);
}

static void bdf_to_cdev(const char *bdf, char *out, size_t len)
{
	char dir[PATH_MAX];
	struct dirent *e;
	DIR *d;

	snprintf(dir, sizeof(dir), "/sys/bus/pci/devices/%s/vfio-dev", bdf);
	d = opendir(dir);
	if (!d)
		die("open vfio-dev dir");
	while ((e = readdir(d))) {
		if (e->d_name[0] == '.')
			continue;
		if (strlen(e->d_name) + sizeof(VFIO_CDEV_PREFIX) > len)
			die_msg("cdev name too long");
		snprintf(out, len, "%s%s", VFIO_CDEV_PREFIX, e->d_name);
		closedir(d);
		return;
	}
	closedir(d);
	die_msg("no cdev for BDF");
}

static int open_iommufd(void)
{
	int fd = open("/dev/iommu", O_RDWR);

	if (fd == -1)
		die("open /dev/iommu");
	return fd;
}

static int export_dmabuf(int dev_fd, uint32_t region, uint64_t off, uint64_t len)
{
	struct {
		struct vfio_device_feature feature;
		struct vfio_device_feature_dma_buf dmabuf;
		struct vfio_region_dma_range range;
	} buf = {
		.feature.argsz = sizeof(buf),
		.feature.flags = VFIO_DEVICE_FEATURE_GET | VFIO_DEVICE_FEATURE_DMA_BUF,
		.dmabuf.region_index = region,
		.dmabuf.open_flags = O_RDWR | O_CLOEXEC,
		.dmabuf.nr_ranges = 1,
		.range.offset = off,
		.range.length = len,
	};
	int fd = ioctl(dev_fd, VFIO_DEVICE_FEATURE, &buf);

	if (fd == -1)
		die("VFIO_DEVICE_FEATURE_DMA_BUF export");
	return fd;
}

static struct ibv_mr *import_mr(struct ibv_pd *pd, int dmabuf_fd, size_t len)
{
	struct ibv_mr *mr;

	mr = ibv_reg_dmabuf_mr(pd, 0, len, 0, dmabuf_fd, IBV_ACCESS_LOCAL_WRITE);
	if (!mr)
		die("ibv_reg_dmabuf_mr");
	return mr;
}

static int advise_mr(struct ibv_pd *pd, struct ibv_mr *mr)
{
	struct ibv_sge sge = {
		.addr = (uintptr_t)mr->addr,
		.length = mr->length,
		.lkey = mr->lkey,
	};

	return ibv_advise_mr(pd, IBV_ADVISE_MR_ADVICE_PREFETCH,
			     IBV_ADVISE_MR_FLAG_FLUSH, &sge, 1);
}

int main(int argc, char **argv)
{
	const char *bdf;
	const char *ibdev;

	if (argc != 3) {
		fprintf(stderr, "usage: %s BDF IB_DEV\n", argv[0]);
		return 1;
	}
	bdf = argv[1];
	ibdev = argv[2];
	struct vfio_device_info info = { .argsz = sizeof(info) };
	struct vfio_region_info reg = { .argsz = sizeof(reg),
					.index = VFIO_PCI_BAR4_REGION_INDEX };
	int dev_fd, iommufd_fd, dmabuf_fd;
	struct vfio_device_bind_iommufd bind;
	struct ibv_device **dev_list;
	struct ibv_context *ctx = NULL;
	struct ibv_pd *pd;
	struct ibv_mr *mr1, *mr2;
	char dev_path[PATH_MAX], ib_name[64];
	int i, ndevs, ret;

	bdf_to_cdev(bdf, dev_path, sizeof(dev_path));
	printf("[1] open vfio device %s (%s)\n", bdf, dev_path);
	dev_fd = open(dev_path, O_RDWR);
	if (dev_fd == -1)
		die("open vfio cdev");
	iommufd_fd = open_iommufd();
	memset(&bind, 0, sizeof(bind));
	bind.argsz = sizeof(bind);
	bind.iommufd = iommufd_fd;
	if (ioctl(dev_fd, VFIO_DEVICE_BIND_IOMMUFD, &bind) == -1)
		die("VFIO_DEVICE_BIND_IOMMUFD");

	if (ioctl(dev_fd, VFIO_DEVICE_GET_INFO, &info) == -1)
		die("VFIO_DEVICE_GET_INFO");
	printf("    device flags=%s%s regions=%u\n",
	       (info.flags & VFIO_DEVICE_FLAGS_PCI) ? "PCI " : "",
	       (info.flags & VFIO_DEVICE_FLAGS_RESET) ? "RESET" : "(no-reset)",
	       info.num_regions);
	if (!(info.flags & VFIO_DEVICE_FLAGS_RESET))
		die("device reports no RESET support");
	if (ioctl(dev_fd, VFIO_DEVICE_GET_REGION_INFO, &reg) == -1)
		die("VFIO_DEVICE_GET_REGION_INFO BAR4");
	printf("    BAR4 size=%llu MB\n", (unsigned long long)reg.size / (1024 * 1024));

	printf("[2] export dma-buf (BAR4 offset 0, len %u)\n", DMABUF_LEN);
	dmabuf_fd = export_dmabuf(dev_fd, VFIO_PCI_BAR4_REGION_INDEX, 0, DMABUF_LEN);
	printf("    dma-buf fd=%d\n", dmabuf_fd);

	dev_list = ibv_get_device_list(&ndevs);
	if (!dev_list)
		die("ibv_get_device_list");
	if (!ibdev[strspn(ibdev, "0123456789")]) {
		i = atoi(ibdev);
		if (i < 0 || i >= ndevs)
			die_msg("IB_DEV index out of range");
	} else {
		for (i = 0; i < ndevs; i++) {
			if (!strcmp(ibv_get_device_name(dev_list[i]), ibdev))
				break;
		}
		if (i == ndevs)
			die_msg("no such RDMA device");
	}
	ctx = ibv_open_device(dev_list[i]);
	if (!ctx)
		die("ibv_open_device");
	snprintf(ib_name, sizeof(ib_name), "%s", ibv_get_device_name(dev_list[i]));
	ibv_free_device_list(dev_list);
	printf("[3] import dma-buf fd=%d into %s\n", dmabuf_fd, ib_name);
	pd = ibv_alloc_pd(ctx);
	if (!pd)
		die("ibv_alloc_pd");
	mr1 = import_mr(pd, dmabuf_fd, DMABUF_LEN);
	printf("    mr1 lkey=0x%x rkey=0x%x\n", mr1->lkey, mr1->rkey);

	printf("[4] VFIO_DEVICE_RESET (dma-buf stays open across this)\n");
	if (ioctl(dev_fd, VFIO_DEVICE_RESET) == -1)
		die("VFIO_DEVICE_RESET");
	printf("    reset ok\n");

	printf("[5] advise PREFETCH on mr1 (same attachment, no re-register)\n");
	ret = advise_mr(pd, mr1);
	if (ret) {
		fprintf(stderr, "error: advise mr1 after reset: %s\n", strerror(ret));
		return 1;
	}
	printf("    mr1 live again: revoked attachment re-mapped after un-revoke\n");

	printf("[6] re-import SAME dma-buf fd=%d into %s\n", dmabuf_fd, ib_name);
	mr2 = import_mr(pd, dmabuf_fd, DMABUF_LEN);
	printf("    mr2 lkey=0x%x rkey=0x%x\n", mr2->lkey, mr2->rkey);
	printf("SUCCESS: same dma-buf fd re-imported across vfio reset\n");

	printf("[7] close vfio device, advise mr1 again (must fail)\n");
	close(dev_fd);
	close(iommufd_fd);
	dev_fd = iommufd_fd = -1;
	ret = advise_mr(pd, mr1);
	if (!ret) {
		fprintf(stderr, "error: advise mr1 after close unexpectedly succeeded\n");
		return 1;
	}
	printf("    advise failed as expected: %s\n", strerror(ret));

	ibv_dereg_mr(mr2);
	ibv_dereg_mr(mr1);
	ibv_dealloc_pd(pd);
	ibv_close_device(ctx);
	close(dmabuf_fd);
	if (dev_fd != -1)
		close(dev_fd);
	if (iommufd_fd != -1)
		close(iommufd_fd);
	return 0;
}
