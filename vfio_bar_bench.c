// SPDX-License-Identifier: GPL-2.0
/*
 * vfio_bar_bench - vfio-pci BAR mmap fault-path and mmap-path benchmark.
 *
 * Measures what it costs the kernel to mmap() a BAR of a device bound to
 * vfio-pci and to fault the mapping in: single-threaded order-0 faults, huge
 * (PMD/PUD) faults, mmap()/munmap() cycles, and many threads faulting one
 * mapping (or one mapping each) concurrently.  Useful for A/B comparisons of
 * kernels that change vfio_pci_mmap_huge_fault() or the BAR mmap path.
 *
 * Modes:
 *   probe      touch one word at the first and last page of the window
 *   fault4k    mapping laid out so that no huge mapping is possible; one
 *              touch per stride (default 4K) => order-0 fault path
 *   faulthuge  mapping laid out for PUD (or PMD) mappings; one touch per 2M
 *   mmapcycle  mmap()/munmap() loop, with and without one touch
 *   mt         N pinned threads faulting disjoint ranges of one mapping
 *              (--mt-layout split: one mapping per thread)
 *
 * Touches are 32-bit loads through PROT_READ mappings by default; a stray
 * store would SIGSEGV instead of reaching the device.  --write touches with
 * 64-bit stores instead, and --prefill writes a pattern over the window once
 * before measuring (for memories with ECC that must be written before they
 * are read).  Both write to the device: only use them on a window that is
 * plain memory and safe to overwrite.  Reads can have side effects too on
 * register BARs: pick a window that is memory.
 *
 * No DMA is set up (the cdev is bound to an iommufd but never attached to an
 * IOAS; the legacy group fallback sets a TYPE1 IOMMU, which vfio requires
 * before handing out a device fd, but maps nothing), bus mastering is not
 * enabled, and config space and interrupts are not touched.  The kernel's own
 * vfio open/close path may reset the function, as for any vfio user.
 */
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include <linux/vfio.h>

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

#define TOOL_VERSION "1.0"

#define SZ_4K (UINT64_C(1) << 12)
#define SZ_2M (UINT64_C(1) << 21)
#define SZ_1G (UINT64_C(1) << 30)

#define MAX_CPUS 1024
#define MAX_TLIST 32
#define MAX_SPARSE 16

enum {
	M_PROBE = 1 << 0,
	M_FAULT4K = 1 << 1,
	M_FAULTHUGE = 1 << 2,
	M_MMAPCYCLE = 1 << 3,
	M_MT = 1 << 4,
};

struct ctx {
	/* options */
	const char *bdf;
	const char *selftest;
	unsigned int bar;
	uint64_t offset, len, stride;
	bool have_offset, have_len;
	unsigned int modes;
	int iters;			/* 0: per-mode default */
	int warmup;
	unsigned int cycles;
	bool huge_pmd;
	bool mt_split;
	bool thp_disable;
	bool force_group;
	bool json;
	int tlist[MAX_TLIST], ntlist;
	int cpus[MAX_CPUS], ncpus;
	bool cpus_given;

	/* device */
	int fd, iommufd, container, group;
	char how[1024];
	uint64_t region_off, region_size;
	uint32_t region_flags;
	int nsparse;			/* -1: no sparse-mmap capability */
	uint64_t sparse_off[MAX_SPARSE], sparse_size[MAX_SPARSE];
	uint64_t bar_phys;		/* 0 if unknown */
	int numa_node;
	char map_name[512];
	bool dmabuf_backed;
	double clock_ns;
};

static FILE *hout;			/* human output: stdout, or stderr with --json */
static volatile uint32_t g_sink;
/* --write: touch with 64-bit stores through PROT_READ|PROT_WRITE mappings. */
static bool g_write;
/* --prefill: write a pattern over the window once before measuring. */
static bool g_prefill;

/* ------------------------------------------------------------------ */
/* JSON writer (buffered; flushed to stdout at exit, also on errors)   */

static struct {
	bool on;
	FILE *f;
	char *buf;
	size_t len;
	int depth;
	char type[16];
	bool first[16];
} J;

static void j_escape(const char *s)
{
	fputc('"', J.f);
	for (; *s; s++) {
		unsigned char c = (unsigned char)*s;

		if (c == '"' || c == '\\')
			fprintf(J.f, "\\%c", c);
		else if (c < 0x20)
			fprintf(J.f, "\\u%04x", c);
		else
			fputc(c, J.f);
	}
	fputc('"', J.f);
}

static void j_item(const char *key)
{
	if (J.depth > 0) {
		if (!J.first[J.depth])
			fputc(',', J.f);
		J.first[J.depth] = false;
		fputc('\n', J.f);
		for (int i = 0; i < J.depth; i++)
			fputs("  ", J.f);
	}
	if (key && J.type[J.depth] == '{') {
		j_escape(key);
		fputs(": ", J.f);
	}
}

static void j_open(const char *key, char type)
{
	if (!J.on)
		return;
	j_item(key);
	fputc(type, J.f);
	J.depth++;
	J.type[J.depth] = type;
	J.first[J.depth] = true;
}

static void j_close(void)
{
	char t;
	bool empty;

	if (!J.on || J.depth == 0)
		return;
	t = J.type[J.depth];
	empty = J.first[J.depth];
	J.depth--;
	if (!empty) {
		fputc('\n', J.f);
		for (int i = 0; i < J.depth; i++)
			fputs("  ", J.f);
	}
	fputc(t == '{' ? '}' : ']', J.f);
}

static void j_str(const char *k, const char *v)
{
	if (!J.on)
		return;
	j_item(k);
	j_escape(v ? v : "");
}

static void j_u64(const char *k, uint64_t v)
{
	if (!J.on)
		return;
	j_item(k);
	fprintf(J.f, "%" PRIu64, v);
}

static void j_i64(const char *k, int64_t v)
{
	if (!J.on)
		return;
	j_item(k);
	fprintf(J.f, "%" PRId64, v);
}

static void j_hex(const char *k, uint64_t v)
{
	char b[32];

	snprintf(b, sizeof(b), "0x%" PRIx64, v);
	j_str(k, b);
}

static void j_num(double v)
{
	if (isfinite(v))
		fprintf(J.f, "%.3f", v);
	else
		fputs("null", J.f);
}

static void j_dbl(const char *k, double v)
{
	if (!J.on)
		return;
	j_item(k);
	j_num(v);
}

static void j_bool(const char *k, bool v)
{
	if (!J.on)
		return;
	j_item(k);
	fputs(v ? "true" : "false", J.f);
}

static void j_darr(const char *k, const double *v, int n)
{
	if (!J.on)
		return;
	j_item(k);
	fputc('[', J.f);
	for (int i = 0; i < n; i++) {
		if (i)
			fputs(", ", J.f);
		j_num(v[i]);
	}
	fputc(']', J.f);
}

static void j_iarr(const char *k, const int *v, int n)
{
	if (!J.on)
		return;
	j_item(k);
	fputc('[', J.f);
	for (int i = 0; i < n; i++)
		fprintf(J.f, "%s%d", i ? ", " : "", v[i]);
	fputc(']', J.f);
}

static void j_begin(void)
{
	J.f = open_memstream(&J.buf, &J.len);
	if (!J.f) {
		perror("open_memstream");
		exit(1);
	}
	J.on = true;
	j_open(NULL, '{');
}

static void j_finish(const char *err)
{
	if (!J.on)
		return;
	while (J.depth > 1)
		j_close();
	if (err)
		j_str("error", err);
	j_close();
	J.on = false;
	fclose(J.f);
	fwrite(J.buf, 1, J.len, stdout);
	fputc('\n', stdout);
	fflush(stdout);
	free(J.buf);
}

/* ------------------------------------------------------------------ */
/* Utilities                                                           */

static void hp(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void hp(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(hout, fmt, ap);
	va_end(ap);
	fflush(hout);
}

static void die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void die(const char *fmt, ...)
{
	char msg[2048];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(msg, sizeof(msg), fmt, ap);
	va_end(ap);
	if (hout)
		fflush(hout);
	fprintf(stderr, "vfio_bar_bench: ERROR: %s\n", msg);
	j_finish(msg);
	exit(1);
}

static void *xcalloc(size_t n, size_t sz)
{
	void *p = calloc(n ? n : 1, sz ? sz : 1);

	if (!p)
		die("out of memory (%zu x %zu bytes)", n, sz);
	return p;
}

/* Write every page so later accesses cannot page-fault (fault counts stay exact). */
static void prefault(void *p, size_t bytes)
{
	volatile char *c = p;

	for (size_t i = 0; i < bytes; i += SZ_4K)
		c[i] = 0;
	if (bytes)
		c[bytes - 1] = 0;
}

static inline uint64_t now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double clock_overhead_ns(void)
{
	const int n = 200000;
	uint64_t t0 = now_ns();

	for (int i = 0; i < n; i++)
		(void)now_ns();
	return (double)(now_ns() - t0) / n;
}

static uint64_t align_up(uint64_t v, uint64_t a)
{
	return (v + a - 1) & ~(a - 1);
}

static uint64_t align_down(uint64_t v, uint64_t a)
{
	return v & ~(a - 1);
}

/* Number of naturally aligned 'a'-sized blocks wholly inside [va, va+len). */
static uint64_t full_blocks(uint64_t va, uint64_t len, uint64_t a)
{
	uint64_t s = align_up(va, a), e = align_down(va + len, a);

	return e > s ? (e - s) / a : 0;
}

static const char *hsize(uint64_t v, char *buf, size_t len)
{
	if (v >= SZ_1G && !(v % (SZ_1G / 100)))
		snprintf(buf, len, "%.2f GiB", (double)v / SZ_1G);
	else if (v >= SZ_2M / 2)
		snprintf(buf, len, "%.2f MiB", (double)v / (1 << 20));
	else
		snprintf(buf, len, "%.1f KiB", (double)v / 1024);
	return buf;
}

static int parse_size(const char *s, uint64_t *out)
{
	unsigned long long v;
	uint64_t mul = 1;
	char *end;

	errno = 0;
	v = strtoull(s, &end, 0);
	if (errno || end == s)
		return -1;
	switch (toupper((unsigned char)*end)) {
	case 'K':
		mul = UINT64_C(1) << 10;
		break;
	case 'M':
		mul = UINT64_C(1) << 20;
		break;
	case 'G':
		mul = UINT64_C(1) << 30;
		break;
	case 'T':
		mul = UINT64_C(1) << 40;
		break;
	case '\0':
		break;
	default:
		return -1;
	}
	if (mul > 1) {
		end++;
		if (*end == 'i' || *end == 'I')
			end++;
		if (*end == 'B' || *end == 'b')
			end++;
	}
	if (*end || v > UINT64_MAX / mul)
		return -1;
	*out = v * mul;
	return 0;
}

/* "0-3,8,10-11" -> list; stops at whitespace.  Returns count or -1. */
static int parse_list(const char *s, int *out, int max)
{
	int n = 0;

	while (*s && !isspace((unsigned char)*s)) {
		char *end;
		long a, b;

		a = strtol(s, &end, 10);
		if (end == s || a < 0)
			return -1;
		b = a;
		if (*end == '-') {
			s = end + 1;
			b = strtol(s, &end, 10);
			if (end == s || b < a)
				return -1;
		}
		for (long c = a; c <= b; c++) {
			if (n >= max)
				return -1;
			out[n++] = (int)c;
		}
		s = end;
		if (*s == ',')
			s++;
		else if (*s && !isspace((unsigned char)*s))
			return -1;
	}
	return n;
}

static int read_file(const char *path, char *buf, size_t len)
{
	ssize_t n;
	int fd, e;

	fd = open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return -errno;
	n = read(fd, buf, len - 1);
	e = errno;
	close(fd);
	if (n < 0)
		return -e;
	buf[n] = 0;
	while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == ' '))
		buf[--n] = 0;
	return (int)n;
}

/* ------------------------------------------------------------------ */
/* Statistics and metric output                                        */

struct st {
	double med, p90, p10, p99, min, max, mean;
	int n;
};

static int dcmp(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;

	return (x > y) - (x < y);
}

static double pctl(const double *s, int n, double p)
{
	double r = p * (n - 1);
	int i = (int)r;

	if (i >= n - 1)
		return s[n - 1];
	return s[i] + (r - i) * (s[i + 1] - s[i]);
}

static struct st stats_of(const double *v, int n)
{
	struct st s = { .n = n };
	double *t, sum = 0;

	if (n <= 0) {
		s.med = s.p90 = s.p10 = s.p99 = s.min = s.max = s.mean = NAN;
		return s;
	}
	t = xcalloc(n, sizeof(*t));
	memcpy(t, v, n * sizeof(*t));
	qsort(t, n, sizeof(*t), dcmp);
	for (int i = 0; i < n; i++)
		sum += t[i];
	s.min = t[0];
	s.max = t[n - 1];
	s.med = pctl(t, n, 0.5);
	s.p90 = pctl(t, n, 0.9);
	s.p10 = pctl(t, n, 0.1);
	s.p99 = pctl(t, n, 0.99);
	s.mean = sum / n;
	free(t);
	return s;
}

static double mean_of(const double *v, int n)
{
	double sum = 0;

	for (int i = 0; i < n; i++)
		sum += v[i];
	return n ? sum / n : NAN;
}

/*
 * One per-round metric.  Latency-like metrics: the slow tail is p90.
 * Rate metrics (rate=true): the slow tail is p10.  JSON has both.
 */
static void metric(const char *key, const char *label, const char *unit,
		   const double *v, int n, bool rate)
{
	struct st s = stats_of(v, n);
	int dp = strcmp(unit, "us") ? 1 : 3;

	hp("  %-36s median %11.*f  %s %11.*f  [min %.*f, max %.*f] %s\n",
	   label, dp, s.med, rate ? "p10" : "p90", dp, rate ? s.p10 : s.p90,
	   dp, s.min, dp, s.max, unit);
	j_open(key, '{');
	j_str("unit", unit);
	j_dbl("median", s.med);
	j_dbl("p90", s.p90);
	j_dbl("p10", s.p10);
	j_dbl("min", s.min);
	j_dbl("max", s.max);
	j_dbl("mean", s.mean);
	j_i64("n", n);
	j_darr("rounds", v, n);
	j_close();
}

/* Pooled per-operation distribution (mmapcycle). */
static void opmetric(const char *key, const char *label, const char *unit,
		     const double *v, int n)
{
	struct st s = stats_of(v, n);
	int dp = strcmp(unit, "us") ? 1 : 3;

	hp("  %-36s p50 %9.*f  p90 %9.*f  p99 %9.*f  [min %.*f, n=%d] %s\n",
	   label, dp, s.med, dp, s.p90, dp, s.p99, dp, s.min, n, unit);
	j_open(key, '{');
	j_str("unit", unit);
	j_dbl("p50", s.med);
	j_dbl("p90", s.p90);
	j_dbl("p99", s.p99);
	j_dbl("min", s.min);
	j_dbl("max", s.max);
	j_dbl("mean", s.mean);
	j_i64("n", n);
	j_close();
}

/* ------------------------------------------------------------------ */
/* SIGBUS/SIGSEGV guard                                                */

static __thread sigjmp_buf tl_jb;
static __thread volatile sig_atomic_t tl_armed;
static __thread volatile int tl_sig;
static __thread void *volatile tl_addr;

static void on_fault(int sig, siginfo_t *si, void *uc)
{
	(void)uc;
	if (tl_armed) {
		tl_armed = 0;
		tl_sig = sig;
		tl_addr = si->si_addr;
		siglongjmp(tl_jb, 1);
	}
	/* Not inside a guarded BAR access: take the default action. */
	signal(sig, SIG_DFL);
	raise(sig);
}

static void install_fault_guard(void)
{
	struct sigaction sa;

	memset(&sa, 0, sizeof(sa));
	sa.sa_sigaction = on_fault;
	sa.sa_flags = SA_SIGINFO;
	sigemptyset(&sa.sa_mask);
	if (sigaction(SIGBUS, &sa, NULL) || sigaction(SIGSEGV, &sa, NULL))
		die("sigaction: %s", strerror(errno));
}

/*
 * Run fn(arg) with SIGBUS/SIGSEGV converted into a -1 return (details in
 * tl_sig/tl_addr).  sigsetjmp() saves the signal mask (one syscall per
 * call, outside any timed region) so the mask is restored on the jump.
 */
static int guarded(void (*fn)(void *), void *arg)
{
	void (*volatile vfn)(void *) = fn;
	void *volatile varg = arg;

	if (sigsetjmp(tl_jb, 1)) {
		tl_armed = 0;
		return -1;
	}
	tl_armed = 1;
	vfn(varg);
	tl_armed = 0;
	return 0;
}

static void die_fault(const char *what, int sig, uintptr_t addr,
		      uintptr_t va, uint64_t win_off)
{
	uint64_t off = addr >= va ? win_off + (addr - va) : 0;

	if (sig == SIGBUS)
		die("%s: unexpected SIGBUS reading VA 0x%" PRIxPTR
		    " (BAR offset 0x%" PRIx64 "): the vfio fault handler "
		    "refused the access (memory decode off, device in D3 or "
		    "reset, or the BAR mapping/DMABUF was revoked)",
		    what, addr, off);
	die("%s: unexpected SIGSEGV at VA 0x%" PRIxPTR
	    " (BAR offset 0x%" PRIx64 ") - benchmark bug?", what, addr, off);
}

/* ------------------------------------------------------------------ */
/* Touch passes                                                        */

static long thread_faults(void)
{
	struct rusage ru;

	if (getrusage(RUSAGE_THREAD, &ru))
		return -1;
	return ru.ru_minflt + ru.ru_majflt;
}

struct pass {
	uintptr_t base;			/* address of the first touch */
	size_t n;			/* number of touches */
	size_t stride;			/* bytes between touches */
	uint64_t *lat;			/* optional per-touch latency, ns */
	uint64_t t0, t1;		/* pass start/end, ns */
	long faults;			/* faults taken by this thread in the pass */
	uint32_t acc;
};

static void pass_fn(void *arg)
{
	struct pass *p = arg;
	const char *b = (const char *)p->base;
	uint32_t acc = 0;
	uint64_t t0, t1;
	long f0;

	f0 = thread_faults();
	t0 = now_ns();
	if (p->lat) {
		for (size_t i = 0; i < p->n; i++) {
			uint64_t a = now_ns();

			if (g_write)
				*(volatile uint64_t *)(b + i * p->stride) = (uintptr_t)b ^ i;
			else
				acc ^= *(const volatile uint32_t *)(b + i * p->stride);
			p->lat[i] = now_ns() - a;
		}
	} else {
		if (g_write) {
			for (size_t i = 0; i < p->n; i++)
				*(volatile uint64_t *)(b + i * p->stride) = (uintptr_t)b ^ i;
		} else {
			for (size_t i = 0; i < p->n; i++)
				acc ^= *(const volatile uint32_t *)(b + i * p->stride);
		}
	}
	t1 = now_ns();
	p->faults = thread_faults() - f0;
	p->t0 = t0;
	p->t1 = t1;
	p->acc = acc;
}

/* ------------------------------------------------------------------ */
/* VA layout and BAR mapping                                           */

/*
 * Reserve PROT_NONE VA and punch a hole [va, va+len) with va % align == mod.
 * BAR mappings go into the hole with MAP_FIXED_NOREPLACE, so they never
 * clobber anything; the reservation remnants around the hole keep other
 * mappings from landing next to it.
 */
struct layout {
	void *res;
	size_t res_len;
	uintptr_t va;
	size_t len;
};

static int layout_make(struct layout *l, size_t len, uint64_t align, uint64_t mod)
{
	const size_t guard = SZ_2M;
	size_t res_len = len + align + 2 * guard;
	uintptr_t lo, va;
	void *p;

	p = mmap(NULL, res_len, PROT_NONE,
		 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
	if (p == MAP_FAILED)
		return -errno;
	lo = (uintptr_t)p + guard;
	va = (uintptr_t)align_up(lo - mod, align) + mod;
	if (munmap((void *)va, len)) {
		int e = -errno;

		munmap(p, res_len);
		return e;
	}
	l->res = p;
	l->res_len = res_len;
	l->va = va;
	l->len = len;
	return 0;
}

static void layout_free(struct layout *l)
{
	if (l->res)
		munmap(l->res, l->res_len);
	l->res = NULL;
}

static int bar_map(const struct ctx *x, uintptr_t va, size_t len,
		   uint64_t win_off, uint64_t *ns)
{
	uint64_t t0, t1;
	void *p;

	t0 = now_ns();
	p = mmap((void *)va, len, PROT_READ | (g_write ? PROT_WRITE : 0),
		 MAP_SHARED | MAP_FIXED_NOREPLACE, x->fd, (off_t)(x->region_off + win_off));
	t1 = now_ns();
	if (p == MAP_FAILED)
		return -errno;
	if ((uintptr_t)p != va) {
		munmap(p, len);
		return -EADDRINUSE;
	}
	if (ns)
		*ns = t1 - t0;
	return 0;
}

static int bar_unmap(uintptr_t va, size_t len, uint64_t *ns)
{
	uint64_t t0, t1;
	int r;

	t0 = now_ns();
	r = munmap((void *)va, len);
	t1 = now_ns();
	if (r)
		return -errno;
	if (ns)
		*ns = t1 - t0;
	return 0;
}

static uint64_t pa_of(const struct ctx *x, uint64_t off)
{
	/* bar_phys==0 (unknown): PCI BARs are naturally aligned to their size. */
	return x->bar_phys + off;
}

/* VA residue mod 2M such that VA - PA == 4K (mod 2M): no PMD/PUD ever fits. */
static uint64_t mod_4k_layout(const struct ctx *x, uint64_t off)
{
	return (pa_of(x, off) + SZ_4K) & (SZ_2M - 1);
}

static void map_name(uintptr_t va, char *out, size_t len)
{
	char line[4096];
	FILE *f;

	snprintf(out, len, "?");
	f = fopen("/proc/self/maps", "re");
	if (!f)
		return;
	while (fgets(line, sizeof(line), f)) {
		unsigned long s, e;
		int pos = -1;
		char *nm;

		if (sscanf(line, "%lx-%lx %*s %*s %*s %*s %n", &s, &e, &pos) < 2 ||
		    pos < 0 || s != va)
			continue;
		nm = line + pos;
		nm[strcspn(nm, "\n")] = 0;
		snprintf(out, len, "%s", nm);
		break;
	}
	fclose(f);
}

struct smaps_info {
	bool found;
	long rss_kb, pmd_kb;
	char thp[64];
	char vmflags[256];
};

static void smaps_read(uintptr_t va, struct smaps_info *si)
{
	char line[1024];
	bool in = false;
	FILE *f;

	memset(si, 0, sizeof(*si));
	si->rss_kb = si->pmd_kb = -1;
	f = fopen("/proc/self/smaps", "re");
	if (!f)
		return;
	while (fgets(line, sizeof(line), f)) {
		unsigned long s, e;
		char *v;

		if (sscanf(line, "%lx-%lx ", &s, &e) == 2) {
			in = (s == va);
			si->found |= in;
			continue;
		}
		if (!in)
			continue;
		line[strcspn(line, "\n")] = 0;
		v = strchr(line, ':');
		if (!v)
			continue;
		*v++ = 0;
		while (*v == ' ' || *v == '\t')
			v++;
		if (!strcmp(line, "Rss"))
			si->rss_kb = strtol(v, NULL, 10);
		else if (!strcmp(line, "FilePmdMapped"))
			si->pmd_kb = strtol(v, NULL, 10);
		else if (!strcmp(line, "THPeligible"))
			snprintf(si->thp, sizeof(si->thp), "%s", v);
		else if (!strcmp(line, "VmFlags"))
			snprintf(si->vmflags, sizeof(si->vmflags), "%s", v);
	}
	fclose(f);
}

static const char *const vm_keys[] = {
	"pgfault", "vma_lock_success", "vma_lock_abort", "vma_lock_retry",
	"vma_lock_miss",
};
#define NVM (sizeof(vm_keys) / sizeof(vm_keys[0]))

struct vmsnap {
	unsigned long long v[NVM];
	bool ok[NVM];
};

static int vmstat_read(struct vmsnap *s)
{
	unsigned long long val;
	char key[128];
	FILE *f;

	memset(s, 0, sizeof(*s));
	f = fopen("/proc/vmstat", "re");
	if (!f)
		return -1;
	while (fscanf(f, "%127s %llu", key, &val) == 2) {
		for (size_t k = 0; k < NVM; k++) {
			if (!strcmp(key, vm_keys[k])) {
				s->v[k] = val;
				s->ok[k] = true;
			}
		}
	}
	fclose(f);
	return 0;
}

/* ------------------------------------------------------------------ */
/* CPU selection and pinning                                           */

static int pin_self(int cpu)
{
	cpu_set_t s;

	CPU_ZERO(&s);
	CPU_SET(cpu, &s);
	return pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

static bool is_primary_thread(int cpu)
{
	char path[128], buf[512];
	int sib[256], n, mn;

	snprintf(path, sizeof(path),
		 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
	if (read_file(path, buf, sizeof(buf)) <= 0)
		return true;
	n = parse_list(buf, sib, 256);
	if (n <= 0)
		return true;
	mn = sib[0];
	for (int i = 1; i < n; i++)
		if (sib[i] < mn)
			mn = sib[i];
	return cpu == mn;
}

struct cand {
	int cpu, key;
};

static int cand_cmp(const void *a, const void *b)
{
	const struct cand *x = a, *y = b;

	if (x->key != y->key)
		return x->key - y->key;
	return x->cpu - y->cpu;
}

/*
 * Default CPU order: device-local first-SMT-sibling CPUs, then remote
 * first siblings, then local second siblings, then remote ones; CPU 0
 * last within its class.
 */
static void auto_cpus(struct ctx *x)
{
	static bool local[MAX_CPUS];
	static struct cand c[MAX_CPUS];
	static int lc[MAX_CPUS];
	char path[PATH_MAX], buf[8192];
	cpu_set_t allowed;
	int nl = -1, nc = 0;

	if (sched_getaffinity(0, sizeof(allowed), &allowed))
		die("sched_getaffinity: %s", strerror(errno));
	if (x->bdf) {
		snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/local_cpulist", x->bdf);
		if (read_file(path, buf, sizeof(buf)) > 0)
			nl = parse_list(buf, lc, MAX_CPUS);
	}
	for (int i = 0; i < nl; i++)
		if (lc[i] < MAX_CPUS)
			local[lc[i]] = true;
	for (int cpu = 0; cpu < MAX_CPUS && cpu < CPU_SETSIZE; cpu++) {
		bool loc, prim;

		if (!CPU_ISSET(cpu, &allowed))
			continue;
		loc = nl <= 0 || local[cpu];
		prim = is_primary_thread(cpu);
		c[nc].cpu = cpu;
		c[nc].key = (((prim ? 0 : 2) + (loc ? 0 : 1)) * 2) + (cpu == 0);
		nc++;
	}
	qsort(c, nc, sizeof(c[0]), cand_cmp);
	for (int i = 0; i < nc; i++)
		x->cpus[i] = c[i].cpu;
	x->ncpus = nc;
	if (!nc)
		die("no usable CPUs in the affinity mask");
}

/* ------------------------------------------------------------------ */
/* Device open                                                         */

static int vfio_open_cdev(struct ctx *x, char *err, size_t errlen)
{
	struct vfio_device_bind_iommufd bind;
	char path[PATH_MAX], name[256] = "";
	struct dirent *de;
	int fd, ifd;
	DIR *d;

	snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vfio-dev", x->bdf);
	d = opendir(path);
	if (!d) {
		snprintf(err, errlen, "%.300s: %s", path, strerror(errno));
		return -1;
	}
	while ((de = readdir(d))) {
		if (!strncmp(de->d_name, "vfio", 4)) {
			snprintf(name, sizeof(name), "%s", de->d_name);
			break;
		}
	}
	closedir(d);
	if (!name[0]) {
		snprintf(err, errlen, "no vfioN entry in %.300s", path);
		return -1;
	}
	snprintf(path, sizeof(path), "/dev/vfio/devices/%s", name);
	fd = open(path, O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		snprintf(err, errlen, "open %.300s: %s", path, strerror(errno));
		return -1;
	}
	ifd = open("/dev/iommu", O_RDWR | O_CLOEXEC);
	if (ifd < 0) {
		snprintf(err, errlen, "open /dev/iommu: %s", strerror(errno));
		close(fd);
		return -1;
	}
	memset(&bind, 0, sizeof(bind));
	bind.argsz = sizeof(bind);
	bind.iommufd = ifd;
	if (ioctl(fd, VFIO_DEVICE_BIND_IOMMUFD, &bind)) {
		snprintf(err, errlen, "VFIO_DEVICE_BIND_IOMMUFD on %.300s: %s",
			 path, strerror(errno));
		close(ifd);
		close(fd);
		return -1;
	}
	x->fd = fd;
	x->iommufd = ifd;
	snprintf(x->how, sizeof(x->how),
		 "cdev %.300s bound to /dev/iommu (devid %u), no IOAS attached (DMA stays blocked)",
		 path, bind.out_devid);
	return 0;
}

static int vfio_open_group(struct ctx *x, char *err, size_t errlen)
{
	struct vfio_group_status st;
	char path[PATH_MAX], link[PATH_MAX];
	int container, group, fd;
	const char *g;
	ssize_t n;

	snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/iommu_group", x->bdf);
	n = readlink(path, link, sizeof(link) - 1);
	if (n < 0) {
		snprintf(err, errlen, "readlink %.300s: %s", path, strerror(errno));
		return -1;
	}
	link[n] = 0;
	g = strrchr(link, '/');
	g = g ? g + 1 : link;

	container = open("/dev/vfio/vfio", O_RDWR | O_CLOEXEC);
	if (container < 0) {
		snprintf(err, errlen, "open /dev/vfio/vfio: %s", strerror(errno));
		return -1;
	}
	if (ioctl(container, VFIO_GET_API_VERSION) != VFIO_API_VERSION ||
	    ioctl(container, VFIO_CHECK_EXTENSION, VFIO_TYPE1v2_IOMMU) <= 0) {
		snprintf(err, errlen, "container lacks VFIO API %d / TYPE1v2",
			 VFIO_API_VERSION);
		close(container);
		return -1;
	}
	snprintf(path, sizeof(path), "/dev/vfio/%.64s", g);
	group = open(path, O_RDWR | O_CLOEXEC);
	if (group < 0) {
		snprintf(err, errlen, "open %.300s: %s", path, strerror(errno));
		close(container);
		return -1;
	}
	memset(&st, 0, sizeof(st));
	st.argsz = sizeof(st);
	if (ioctl(group, VFIO_GROUP_GET_STATUS, &st) ||
	    !(st.flags & VFIO_GROUP_FLAGS_VIABLE)) {
		snprintf(err, errlen,
			 "group %.300s not viable (all devices in the IOMMU group must be bound to vfio-pci or unbound)",
			 path);
		goto fail;
	}
	if (ioctl(group, VFIO_GROUP_SET_CONTAINER, &container)) {
		snprintf(err, errlen, "VFIO_GROUP_SET_CONTAINER: %s", strerror(errno));
		goto fail;
	}
	/* Required before VFIO_GROUP_GET_DEVICE_FD; no VFIO_IOMMU_MAP_DMA is ever issued. */
	if (ioctl(container, VFIO_SET_IOMMU, VFIO_TYPE1v2_IOMMU)) {
		snprintf(err, errlen, "VFIO_SET_IOMMU: %s", strerror(errno));
		goto fail;
	}
	fd = ioctl(group, VFIO_GROUP_GET_DEVICE_FD, x->bdf);
	if (fd < 0) {
		snprintf(err, errlen, "VFIO_GROUP_GET_DEVICE_FD: %s", strerror(errno));
		goto fail;
	}
	x->fd = fd;
	x->container = container;
	x->group = group;
	snprintf(x->how, sizeof(x->how),
		 "group %.300s in a /dev/vfio/vfio container (TYPE1v2, no DMA mappings)", path);
	return 0;
fail:
	close(group);
	close(container);
	return -1;
}

static void open_device(struct ctx *x)
{
	char e1[1024] = "", e2[1024] = "";
	struct stat sb;

	if (x->selftest) {
		x->fd = open(x->selftest, O_RDWR | O_CLOEXEC);
		if (x->fd < 0 || fstat(x->fd, &sb))
			die("selftest: %s: %s", x->selftest, strerror(errno));
		x->region_off = 0;
		x->region_size = (uint64_t)sb.st_size;
		x->region_flags = VFIO_REGION_INFO_FLAG_READ | VFIO_REGION_INFO_FLAG_MMAP;
		x->nsparse = -1;
		snprintf(x->how, sizeof(x->how),
			 "SELFTEST file %s (not a VFIO device; numbers are meaningless)",
			 x->selftest);
		return;
	}
	if (!x->force_group && !vfio_open_cdev(x, e1, sizeof(e1)))
		return;
	if (!vfio_open_group(x, e2, sizeof(e2))) {
		if (e1[0]) {
			size_t l = strlen(x->how);

			snprintf(x->how + l, sizeof(x->how) - l,
				 " [cdev path failed: %.300s]", e1);
		}
		return;
	}
	die("cannot open %s: cdev: %s; group: %s", x->bdf,
	    e1[0] ? e1 : "(skipped)", e2);
}

static void region_info(struct ctx *x)
{
	struct vfio_device_info di;
	struct vfio_region_info *ri;
	size_t sz = sizeof(*ri);
	char path[PATH_MAX], buf[8192];

	if (x->selftest)
		return;

	memset(&di, 0, sizeof(di));
	di.argsz = sizeof(di);
	if (ioctl(x->fd, VFIO_DEVICE_GET_INFO, &di))
		die("VFIO_DEVICE_GET_INFO: %s", strerror(errno));
	if (!(di.flags & VFIO_DEVICE_FLAGS_PCI))
		die("%s is not a vfio-pci device", x->bdf);
	if (x->bar >= VFIO_PCI_ROM_REGION_INDEX || x->bar >= di.num_regions)
		die("BAR%u is not a valid BAR region", x->bar);

	ri = xcalloc(1, sz);
	ri->argsz = sz;
	ri->index = x->bar;
	if (ioctl(x->fd, VFIO_DEVICE_GET_REGION_INFO, ri))
		die("VFIO_DEVICE_GET_REGION_INFO(%u): %s", x->bar, strerror(errno));
	if (ri->argsz > sz) {
		sz = ri->argsz;
		free(ri);
		ri = xcalloc(1, sz);
		ri->argsz = sz;
		ri->index = x->bar;
		if (ioctl(x->fd, VFIO_DEVICE_GET_REGION_INFO, ri))
			die("VFIO_DEVICE_GET_REGION_INFO(%u): %s", x->bar, strerror(errno));
	}
	x->region_off = ri->offset;
	x->region_size = ri->size;
	x->region_flags = ri->flags;
	x->nsparse = -1;
	if ((ri->flags & VFIO_REGION_INFO_FLAG_CAPS) && ri->cap_offset) {
		uint32_t off = ri->cap_offset;

		for (int guard = 0; off && guard < 64; guard++) {
			struct vfio_info_cap_header *h;

			if (off + sizeof(*h) > sz)
				break;
			h = (void *)((char *)ri + off);
			if (h->id == VFIO_REGION_INFO_CAP_SPARSE_MMAP) {
				struct vfio_region_info_cap_sparse_mmap *sp = (void *)h;

				x->nsparse = 0;
				if (off + sizeof(*sp) +
				    (uint64_t)sp->nr_areas * sizeof(sp->areas[0]) <= sz) {
					for (uint32_t i = 0; i < sp->nr_areas &&
					     x->nsparse < MAX_SPARSE; i++) {
						x->sparse_off[x->nsparse] = sp->areas[i].offset;
						x->sparse_size[x->nsparse] = sp->areas[i].size;
						x->nsparse++;
					}
				}
			}
			off = h->next;
		}
	}
	free(ri);

	snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource", x->bdf);
	if (read_file(path, buf, sizeof(buf)) > 0) {
		char *line = buf;

		for (unsigned int i = 0; i < x->bar && line; i++) {
			line = strchr(line, '\n');
			if (line)
				line++;
		}
		if (line) {
			unsigned long long s = 0, e = 0, fl = 0;

			if (sscanf(line, "%llx %llx %llx", &s, &e, &fl) == 3 && s)
				x->bar_phys = s;
		}
	}
	snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", x->bdf);
	x->numa_node = -1;
	if (read_file(path, buf, sizeof(buf)) > 0)
		x->numa_node = atoi(buf);
}

static void validate_window(struct ctx *x)
{
	if (!(x->region_flags & VFIO_REGION_INFO_FLAG_READ) ||
	    !(x->region_flags & VFIO_REGION_INFO_FLAG_MMAP))
		die("BAR%u region is not readable+mmappable (flags 0x%x)",
		    x->bar, x->region_flags);
	if (!x->region_size)
		die("BAR%u is not implemented (size 0)", x->bar);
	if (!x->have_len) {
		uint64_t avail = x->offset < x->region_size ? x->region_size - x->offset : 0;

		/* Default window: from --offset (default 0), at most 1 GiB. */
		x->len = avail < SZ_1G ? avail & ~(uint64_t)(SZ_4K - 1) : SZ_1G;
		for (int i = 0; i < x->nsparse; i++) {
			uint64_t s0 = x->sparse_off[i], s1 = s0 + x->sparse_size[i];

			if (x->offset >= s0 && x->offset < s1 && x->offset + x->len > s1)
				x->len = (s1 - x->offset) & ~(uint64_t)(SZ_4K - 1);
		}
	}
	if (x->offset % SZ_4K || x->len % SZ_4K || !x->len)
		die("--offset/--len must be non-zero multiples of 4 KiB");
	if (x->offset > x->region_size || x->len > x->region_size - x->offset)
		die("window 0x%" PRIx64 "+0x%" PRIx64 " exceeds BAR%u size 0x%" PRIx64,
		    x->offset, x->len, x->bar, x->region_size);
	if (x->nsparse >= 0) {
		bool ok = false;

		for (int i = 0; i < x->nsparse; i++)
			if (x->offset >= x->sparse_off[i] &&
			    x->offset + x->len <= x->sparse_off[i] + x->sparse_size[i])
				ok = true;
		if (!ok)
			die("window is not inside one sparse-mmap area of BAR%u", x->bar);
	}
	if (!x->stride || x->stride % SZ_4K)
		die("--stride must be a non-zero multiple of 4 KiB");
}

/*
 * Write a pattern over the whole window with 64-bit stores, then read back a
 * sample.  Memories with ECC may report errors when never-written locations
 * are read, so this lets read-mode runs touch only initialised memory.
 */
static void prefill_window(struct ctx *x)
{
	const uint64_t pat = 0x5a5aa5a5c3c33c3cULL, n = x->len / 8;
	volatile uint64_t *p;
	uint64_t t0, t1, bad = 0;
	void *m;

	m = mmap(NULL, x->len, PROT_READ | PROT_WRITE, MAP_SHARED, x->fd,
		 (off_t)(x->region_off + x->offset));
	if (m == MAP_FAILED)
		die("prefill: mmap of window failed: %s", strerror(errno));
	p = m;
	t0 = now_ns();
	for (uint64_t i = 0; i < n; i++)
		p[i] = (x->offset + i * 8) ^ pat;
	__sync_synchronize();
	t1 = now_ns();
	for (uint64_t k = 0; k < 4096; k++) {
		uint64_t i = k == 4095 ? n - 1 : (k * (n / 4096) + k * 7) % n;

		if (p[i] != ((x->offset + i * 8) ^ pat))
			bad++;
	}
	munmap(m, x->len);
	hp("prefill : wrote 0x%" PRIx64 " bytes in %.2f s (%.2f GB/s), readback %s\n",
	   x->len, (t1 - t0) / 1e9, x->len / ((t1 - t0) / 1e9) / 1e9,
	   bad ? "MISMATCH" : "ok");
	if (bad)
		die("prefill: %" PRIu64 " of 4096 sampled words read back wrong", bad);
}

/* Map one page at the window start without touching it; read its name. */
static void detect_backing(struct ctx *x)
{
	void *p;

	p = mmap(NULL, SZ_4K, PROT_READ, MAP_SHARED, x->fd,
		 (off_t)(x->region_off + x->offset));
	if (p == MAP_FAILED)
		die("mmap of BAR%u offset 0x%" PRIx64 " failed: %s",
		    x->bar, x->offset, strerror(errno));
	map_name((uintptr_t)p, x->map_name, sizeof(x->map_name));
	x->dmabuf_backed = strstr(x->map_name, "dmabuf") != NULL;
	munmap(p, SZ_4K);
}

/* ------------------------------------------------------------------ */
/* Mode: probe                                                         */

struct probe {
	uintptr_t p;
	uint32_t v;
	uint64_t ns;
	long faults;
};

static void probe_fn(void *arg)
{
	struct probe *pr = arg;
	long f0 = thread_faults();
	uint64_t t0 = now_ns();

	pr->v = *(const volatile uint32_t *)pr->p;
	pr->ns = now_ns() - t0;
	pr->faults = thread_faults() - f0;
}

static void run_probe(struct ctx *x)
{
	uint64_t offs[2] = { x->offset, x->offset + x->len - SZ_4K };
	int n = x->len > SZ_4K ? 2 : 1;

	hp("\n[probe] one 32-bit read at the first and last page of the window\n");
	j_open(NULL, '{');
	j_str("mode", "probe");
	j_open("reads", '[');
	for (int i = 0; i < n; i++) {
		struct probe pr = { 0 };
		void *p;

		p = mmap(NULL, SZ_4K, PROT_READ, MAP_SHARED, x->fd,
			 (off_t)(x->region_off + offs[i]));
		if (p == MAP_FAILED)
			die("probe: mmap BAR%u+0x%" PRIx64 ": %s", x->bar, offs[i],
			    strerror(errno));
		pr.p = (uintptr_t)p;
		if (guarded(probe_fn, &pr))
			die_fault("probe", tl_sig, (uintptr_t)tl_addr, pr.p, offs[i]);
		munmap(p, SZ_4K);
		hp("  BAR%u+0x%-14" PRIx64 " = 0x%08" PRIx32 "   (%.0f ns incl. %ld fault%s)%s\n",
		   x->bar, offs[i], pr.v, (double)pr.ns, pr.faults,
		   pr.faults == 1 ? "" : "s",
		   pr.v == 0xffffffffu ?
		   "  <-- all-ones: unbacked range or failed read? choose another window" : "");
		j_open(NULL, '{');
		j_hex("bar_offset", offs[i]);
		j_hex("value", pr.v);
		j_dbl("ns", (double)pr.ns);
		j_i64("faults", pr.faults);
		j_close();
	}
	j_close();
	j_close();
}

/* ------------------------------------------------------------------ */
/* Mode: fault4k                                                       */

static void run_fault4k(struct ctx *x)
{
	const int R = x->iters ? x->iters : 10, W = x->warmup;
	const size_t n = x->len / x->stride;
	double *ns_f, *rate, *ns_w, *ns_net, *mm, *um;
	struct vmsnap v0, v1;
	bool have_vm = false;
	long fmin = LONG_MAX, fmax = LONG_MIN, wmax = 0;
	struct layout L = { 0 };
	char b1[32], b2[32];
	int rc;

	if (!n)
		die("fault4k: --len smaller than --stride");
	ns_f = xcalloc(R, sizeof(double));
	rate = xcalloc(R, sizeof(double));
	ns_w = xcalloc(R, sizeof(double));
	ns_net = xcalloc(R, sizeof(double));
	mm = xcalloc(R, sizeof(double));
	um = xcalloc(R, sizeof(double));

	rc = layout_make(&L, x->len, SZ_2M, mod_4k_layout(x, x->offset));
	if (rc)
		die("fault4k: VA reservation failed: %s", strerror(-rc));

	hp("\n[fault4k] window %s, stride %s, %zu touches/round, %d rounds (+%d warm-up), CPU %d\n",
	   hsize(x->len, b1, sizeof(b1)), hsize(x->stride, b2, sizeof(b2)), n, R, W, x->cpus[0]);
	hp("  VA 0x%" PRIxPTR ": VA%%2M=0x%" PRIx64 " vs PA%%2M=0x%" PRIx64
	   " => PMD/PUD mappings impossible, every touch is an order-0 fault\n",
	   L.va, (uint64_t)L.va & (SZ_2M - 1), pa_of(x, x->offset) & (SZ_2M - 1));

	for (int r = -W; r < R; r++) {
		struct pass f = { .base = L.va, .n = n, .stride = x->stride };
		struct pass w = f;
		uint64_t tm = 0, tu = 0;

		rc = bar_map(x, L.va, x->len, x->offset, &tm);
		if (rc)
			die("fault4k: mmap of window failed: %s", strerror(-rc));
		if (r == 0)
			have_vm = !vmstat_read(&v0);
		if (guarded(pass_fn, &f))
			die_fault("fault4k", tl_sig, (uintptr_t)tl_addr, L.va, x->offset);
		if (r == 0 && have_vm)
			have_vm = !vmstat_read(&v1);
		if (guarded(pass_fn, &w))
			die_fault("fault4k", tl_sig, (uintptr_t)tl_addr, L.va, x->offset);
		rc = bar_unmap(L.va, x->len, &tu);
		if (rc)
			die("fault4k: munmap failed: %s", strerror(-rc));
		g_sink ^= f.acc ^ w.acc;
		if (r < 0)
			continue;
		ns_f[r] = (double)(f.t1 - f.t0) / n;
		rate[r] = 1e9 / ns_f[r];
		ns_w[r] = (double)(w.t1 - w.t0) / n;
		ns_net[r] = ns_f[r] - ns_w[r];
		mm[r] = tm / 1e3;
		um[r] = tu / 1e3;
		if (f.faults < fmin)
			fmin = f.faults;
		if (f.faults > fmax)
			fmax = f.faults;
		if (w.faults > wmax)
			wmax = w.faults;
	}
	layout_free(&L);

	hp("  faults per round: min %ld max %ld (expected %zu); re-touch pass faults max %ld%s\n",
	   fmin, fmax, n, wmax,
	   (fmin != (long)n || fmax != (long)n || wmax) ?
	   "   <-- WARNING: unexpected fault count" : "");

	j_open(NULL, '{');
	j_str("mode", "fault4k");
	j_u64("stride", x->stride);
	j_u64("touches_per_round", n);
	j_i64("rounds", R);
	j_i64("warmup", W);
	j_i64("cpu", x->cpus[0]);
	j_hex("va_mod_2m", (uint64_t)L.va & (SZ_2M - 1));
	j_hex("pa_mod_2m", pa_of(x, x->offset) & (SZ_2M - 1));
	j_bool("huge_possible", false);
	j_i64("faults_per_round_min", fmin);
	j_i64("faults_per_round_max", fmax);
	j_i64("reread_faults_max", wmax);
	metric("ns_per_fault", "ns/fault (fault pass)", "ns", ns_f, R, false);
	metric("faults_per_sec", "faults/s", "1/s", rate, R, true);
	metric("reread_ns_per_touch", "ns/touch (re-touch, no fault)", "ns", ns_w, R, false);
	metric("net_ns_per_fault", "net ns/fault (fault - re-touch)", "ns", ns_net, R, false);
	metric("mmap_us", "mmap() of window", "us", mm, R, false);
	metric("munmap_us", "munmap() of window", "us", um, R, false);
	if (have_vm) {
		hp("  /proc/vmstat delta over first measured fault pass (system-wide):");
		j_open("vmstat_delta_first_fault_pass", '{');
		for (size_t k = 0; k < NVM; k++) {
			if (!v0.ok[k] || !v1.ok[k])
				continue;
			hp(" %s=%llu", vm_keys[k], v1.v[k] - v0.v[k]);
			j_u64(vm_keys[k], v1.v[k] - v0.v[k]);
		}
		j_close();
		hp("\n");
	}
	j_close();

	free(ns_f);
	free(rate);
	free(ns_w);
	free(ns_net);
	free(mm);
	free(um);
}

/* ------------------------------------------------------------------ */
/* Mode: faulthuge                                                     */

static void run_faulthuge(struct ctx *x)
{
	const int R = x->iters ? x->iters : 30, W = x->warmup;
	const uint64_t pa = pa_of(x, x->offset);
	double *ns_t, *ns_w, *ns_f, *ns_net, *mm, *um, *tmp;
	int *f1s, *f2s, *npud, *npmd, *n4k;
	uint64_t *lat1, *lat3, max_pud, n2m;
	bool pmd_only = x->huge_pmd, pud_possible, consistent = true;
	struct smaps_info si = { 0 };
	struct layout L = { 0 };
	const char *lay;
	char b1[32];
	int rc;

	if (x->offset % SZ_2M || x->len % SZ_2M) {
		hp("\n[faulthuge] skipped: --offset and --len must be multiples of 2 MiB\n");
		j_open(NULL, '{');
		j_str("mode", "faulthuge");
		j_str("skipped", "offset/len not 2M aligned");
		j_close();
		return;
	}
	n2m = x->len / SZ_2M;

	ns_t = xcalloc(R, sizeof(double));
	ns_w = xcalloc(R, sizeof(double));
	ns_f = xcalloc(R, sizeof(double));
	ns_net = xcalloc(R, sizeof(double));
	mm = xcalloc(R, sizeof(double));
	um = xcalloc(R, sizeof(double));
	f1s = xcalloc(R, sizeof(int));
	f2s = xcalloc(R, sizeof(int));
	npud = xcalloc(R, sizeof(int));
	npmd = xcalloc(R, sizeof(int));
	n4k = xcalloc(R, sizeof(int));
	lat1 = xcalloc(n2m, sizeof(uint64_t));
	lat3 = xcalloc(n2m, sizeof(uint64_t));
	tmp = xcalloc(n2m, sizeof(double));
	prefault(lat1, n2m * sizeof(uint64_t));
	prefault(lat3, n2m * sizeof(uint64_t));
	prefault(tmp, n2m * sizeof(double));

	/*
	 * pud: VA == PA (mod 1G), so every 1G-aligned block inside the window
	 * can be a PUD mapping.  pmd: VA == PA + 2M (mod 1G): PUD impossible,
	 * every 2M block can still be a PMD mapping.
	 */
	rc = layout_make(&L, x->len, SZ_1G,
			 pmd_only ? (pa + SZ_2M) & (SZ_1G - 1) : pa & (SZ_1G - 1));
	if (rc) {
		hp("  note: 1G-aligned VA reservation failed (%s); using 2M alignment\n",
		   strerror(-rc));
		rc = layout_make(&L, x->len, SZ_2M, pa & (SZ_2M - 1));
		if (rc)
			die("faulthuge: VA reservation failed: %s", strerror(-rc));
	}
	pud_possible = !(((uint64_t)L.va - pa) & (SZ_1G - 1));
	max_pud = pud_possible ? full_blocks(L.va, x->len, SZ_1G) : 0;
	lay = pud_possible ? "pud" : "pmd";

	hp("\n[faulthuge] window %s = %" PRIu64 " x 2M blocks, layout %s, %d rounds (+%d warm-up), CPU %d%s\n",
	   hsize(x->len, b1, sizeof(b1)), n2m, lay, R, W, x->cpus[0],
	   x->thp_disable ? "  (--thp-disable: expect no huge mappings)" : "");
	hp("  VA 0x%" PRIxPTR ": (VA-PA)%%1G=0x%" PRIx64 " (VA-PA)%%2M=0x%" PRIx64
	   " => at most %" PRIu64 " PUD mappings, the remaining 2M blocks can be PMDs\n",
	   L.va, ((uint64_t)L.va - pa) & (SZ_1G - 1), ((uint64_t)L.va - pa) & (SZ_2M - 1),
	   max_pud);

	for (int r = -W; r < R; r++) {
		struct pass p1 = { .base = L.va, .n = n2m, .stride = SZ_2M, .lat = lat1 };
		struct pass p2 = { .base = L.va + SZ_2M - SZ_4K, .n = n2m, .stride = SZ_2M };
		struct pass p3 = { .base = L.va, .n = n2m, .stride = SZ_2M, .lat = lat3 };
		uint64_t tm = 0, tu = 0, sum1 = 0, sum3 = 0;
		long F1, F2, d;

		rc = bar_map(x, L.va, x->len, x->offset, &tm);
		if (rc)
			die("faulthuge: mmap of window failed: %s", strerror(-rc));
		/* 1: first touch of every 2M block (timed per touch) */
		if (guarded(pass_fn, &p1))
			die_fault("faulthuge", tl_sig, (uintptr_t)tl_addr, L.va, x->offset);
		/* 2: last 4K page of every 2M block: faults only where 4K PTEs were used */
		if (guarded(pass_fn, &p2))
			die_fault("faulthuge", tl_sig, (uintptr_t)tl_addr, L.va, x->offset);
		/* 3: re-read of pass 1 addresses: no faults; MMIO read + TLB cost */
		if (guarded(pass_fn, &p3))
			die_fault("faulthuge", tl_sig, (uintptr_t)tl_addr, L.va, x->offset);
		if (r == 0)
			smaps_read(L.va, &si);
		rc = bar_unmap(L.va, x->len, &tu);
		if (rc)
			die("faulthuge: munmap failed: %s", strerror(-rc));
		g_sink ^= p1.acc ^ p2.acc ^ p3.acc;
		if (r < 0)
			continue;

		for (uint64_t i = 0; i < n2m; i++) {
			sum1 += lat1[i];
			sum3 += lat3[i];
		}
		ns_t[r] = (double)sum1 / n2m;
		for (uint64_t i = 0; i < n2m; i++)
			tmp[i] = (double)lat3[i];
		qsort(tmp, n2m, sizeof(double), dcmp);
		ns_w[r] = pctl(tmp, (int)n2m, 0.5);

		/* The F1 touches that faulted are the F1 slowest ones. */
		F1 = p1.faults;
		F2 = p2.faults;
		ns_f[r] = NAN;
		if (F1 > 0 && (uint64_t)F1 <= n2m) {
			double s = 0;

			for (uint64_t i = 0; i < n2m; i++)
				tmp[i] = (double)lat1[i];
			qsort(tmp, n2m, sizeof(double), dcmp);
			for (uint64_t i = n2m - F1; i < n2m; i++)
				s += tmp[i];
			ns_f[r] = s / F1;
		}
		ns_net[r] = ns_f[r] - ns_w[r];
		mm[r] = tm / 1e3;
		um[r] = tu / 1e3;

		/*
		 * blocks = 512*P + M + K, pass-1 faults F1 = P + M + K,
		 * pass-2 faults F2 = K  =>  P = (blocks - F1) / 511.
		 */
		f1s[r] = (int)F1;
		f2s[r] = (int)F2;
		d = (long)n2m - F1;
		if (d >= 0 && d % 511 == 0 && F1 - d / 511 - F2 >= 0) {
			npud[r] = (int)(d / 511);
			n4k[r] = (int)F2;
			npmd[r] = (int)(F1 - d / 511 - F2);
		} else {
			npud[r] = npmd[r] = n4k[r] = -1;
		}
		if (npud[r] != npud[0] || npmd[r] != npmd[0] || n4k[r] != n4k[0])
			consistent = false;
	}
	layout_free(&L);

	if (npud[0] >= 0)
		hp("  inferred mappings per round (from per-thread fault counts): %d PUD (1G), %d PMD (2M), %d blocks of 4K PTEs%s\n",
		   npud[0], npmd[0], n4k[0], consistent ? "" : "  [varies across rounds, see JSON]");
	else
		hp("  could not infer mapping sizes (pass faults F1=%d F2=%d for %" PRIu64 " blocks)\n",
		   f1s[0], f2s[0], n2m);
	hp("  smaps (first round): VmFlags [%s] THPeligible %s Rss %ld kB FilePmdMapped %ld kB\n"
	   "    (the smaps page walk skips VM_PFNMAP VMAs, so Rss/PmdMapped say nothing here)\n",
	   si.vmflags, si.thp[0] ? si.thp : "?", si.rss_kb, si.pmd_kb);

	j_open(NULL, '{');
	j_str("mode", "faulthuge");
	j_str("layout", lay);
	j_bool("thp_disable", x->thp_disable);
	j_u64("blocks_2m", n2m);
	j_u64("max_pud", max_pud);
	j_i64("rounds", R);
	j_i64("warmup", W);
	j_i64("cpu", x->cpus[0]);
	j_hex("va_minus_pa_mod_1g", ((uint64_t)L.va - pa) & (SZ_1G - 1));
	j_open("inferred", '{');
	j_i64("pud", npud[0]);
	j_i64("pmd", npmd[0]);
	j_i64("blocks_4k", n4k[0]);
	j_bool("consistent", consistent);
	j_iarr("pass1_faults", f1s, R);
	j_iarr("pass2_faults", f2s, R);
	j_close();
	j_open("smaps", '{');
	j_bool("found", si.found);
	j_str("vmflags", si.vmflags);
	j_str("thp_eligible", si.thp);
	j_i64("rss_kb", si.rss_kb);
	j_i64("file_pmd_mapped_kb", si.pmd_kb);
	j_close();
	metric("ns_per_touch", "ns/touch (1 touch per 2M, first pass)", "ns", ns_t, R, false);
	metric("reread_ns_per_touch", "ns/touch (re-touch, no fault, p50)", "ns", ns_w, R, false);
	metric("ns_per_fault", "ns/huge fault (slowest-F1 touches)", "ns", ns_f, R, false);
	metric("net_ns_per_fault", "net ns/huge fault (minus re-touch)", "ns", ns_net, R, false);
	metric("mmap_us", "mmap() of window", "us", mm, R, false);
	metric("munmap_us", "munmap() of window", "us", um, R, false);
	j_close();

	free(ns_t);
	free(ns_w);
	free(ns_f);
	free(ns_net);
	free(mm);
	free(um);
	free(f1s);
	free(f2s);
	free(npud);
	free(npmd);
	free(n4k);
	free(lat1);
	free(lat3);
	free(tmp);
}

/* ------------------------------------------------------------------ */
/* Mode: mmapcycle                                                     */

struct cyc {
	int fd;
	off_t off;
	uintptr_t va;
	size_t size;
	bool touch;
	unsigned int cycles;
	double *mm, *tch, *rr, *um;	/* per-op ns */
	unsigned int done;
	int err;
	long faults;
	uint32_t acc;
};

static void cycle_fn(void *arg)
{
	struct cyc *c = arg;
	uint32_t acc = 0;
	long f0 = thread_faults();

	for (unsigned int i = 0; i < c->cycles; i++) {
		uint64_t t0, t1, t2, t3, t4;
		void *p;

		t0 = now_ns();
		p = mmap((void *)c->va, c->size, PROT_READ | (g_write ? PROT_WRITE : 0),
			 MAP_SHARED | MAP_FIXED_NOREPLACE, c->fd, c->off);
		t1 = now_ns();
		if (p == MAP_FAILED) {
			c->err = errno;
			return;
		}
		if ((uintptr_t)p != c->va) {
			munmap(p, c->size);
			c->err = EADDRINUSE;
			return;
		}
		t2 = t3 = t1;
		if (c->touch) {
			if (g_write)
				*(volatile uint64_t *)p = i;	/* fault + write */
			else
				acc ^= *(const volatile uint32_t *)p;	/* fault + read */
			t2 = now_ns();
			if (g_write)
				*(volatile uint64_t *)p = i;	/* write only */
			else
				acc ^= *(const volatile uint32_t *)p;	/* read only */
			t3 = now_ns();
		}
		if (munmap(p, c->size)) {
			c->err = errno;
			return;
		}
		t4 = now_ns();
		c->mm[i] = (double)(t1 - t0);
		c->tch[i] = (double)(t2 - t1);
		c->rr[i] = (double)(t3 - t2);
		c->um[i] = (double)(t4 - t3);
		c->done = i + 1;
	}
	c->faults = thread_faults() - f0;
	c->acc = acc;
}

static void run_mmapcycle(struct ctx *x)
{
	const int R = x->iters ? x->iters : 10, W = x->warmup;
	const unsigned int C = x->cycles;
	const uint64_t sizes[2] = { SZ_4K, SZ_2M };
	const int nsz = x->len >= SZ_2M ? 2 : 1;
	const size_t total = (size_t)R * C;

	hp("\n[mmapcycle] mmap()/munmap() of a 4K%s mapping at BAR offset 0x%" PRIx64
	   ", %u cycles x %d rounds (+%d warm-up), CPU %d\n",
	   nsz > 1 ? " and a 2M" : "", x->offset, C, R, W, x->cpus[0]);
	hp("  VA placed so that no huge mapping is possible; per-op latencies include ~1 clock_gettime\n");

	for (int si = 0; si < nsz; si++) {
		for (int touch = 0; touch <= 1; touch++) {
			double *mm, *tc, *rr, *um, *smm, *stc, *srr, *sum;
			double *rmm, *rtc, *rrr, *rum;
			long fmin = LONG_MAX, fmax = LONG_MIN;
			struct layout L = { 0 };
			char variant[32], b1[32];
			int rc;

			snprintf(variant, sizeof(variant), "%s-%s",
				 si ? "2M" : "4K", touch ? "touch" : "notouch");
			mm = xcalloc(total, sizeof(double));
			tc = xcalloc(total, sizeof(double));
			rr = xcalloc(total, sizeof(double));
			um = xcalloc(total, sizeof(double));
			smm = xcalloc(C, sizeof(double));
			stc = xcalloc(C, sizeof(double));
			srr = xcalloc(C, sizeof(double));
			sum = xcalloc(C, sizeof(double));
			rmm = xcalloc(R, sizeof(double));
			rtc = xcalloc(R, sizeof(double));
			rrr = xcalloc(R, sizeof(double));
			rum = xcalloc(R, sizeof(double));
			prefault(mm, total * sizeof(double));
			prefault(tc, total * sizeof(double));
			prefault(rr, total * sizeof(double));
			prefault(um, total * sizeof(double));
			prefault(smm, C * sizeof(double));
			prefault(stc, C * sizeof(double));
			prefault(srr, C * sizeof(double));
			prefault(sum, C * sizeof(double));

			rc = layout_make(&L, sizes[si], SZ_2M, mod_4k_layout(x, x->offset));
			if (rc)
				die("mmapcycle: VA reservation failed: %s", strerror(-rc));

			for (int r = -W; r < R; r++) {
				size_t o = r < 0 ? 0 : (size_t)r * C;
				struct cyc c = {
					.fd = x->fd,
					.off = (off_t)(x->region_off + x->offset),
					.va = L.va,
					.size = sizes[si],
					.touch = touch,
					.cycles = C,
					.mm = r < 0 ? smm : mm + o,
					.tch = r < 0 ? stc : tc + o,
					.rr = r < 0 ? srr : rr + o,
					.um = r < 0 ? sum : um + o,
				};

				if (guarded(cycle_fn, &c)) {
					munmap((void *)L.va, sizes[si]);
					die_fault("mmapcycle", tl_sig, (uintptr_t)tl_addr,
						  L.va, x->offset);
				}
				if (c.err)
					die("mmapcycle %s: mmap/munmap failed after %u cycles: %s",
					    variant, c.done, strerror(c.err));
				g_sink ^= c.acc;
				if (r < 0)
					continue;
				rmm[r] = mean_of(c.mm, C) / 1e3;
				rtc[r] = mean_of(c.tch, C);
				rrr[r] = mean_of(c.rr, C);
				rum[r] = mean_of(c.um, C) / 1e3;
				if (c.faults < fmin)
					fmin = c.faults;
				if (c.faults > fmax)
					fmax = c.faults;
			}
			layout_free(&L);

			hp(" %s mapping, %s:\n", hsize(sizes[si], b1, sizeof(b1)),
			   touch ? "mmap + first-touch + re-touch + munmap" : "mmap + munmap, no touch");
			j_open(NULL, '{');
			j_str("mode", "mmapcycle");
			j_str("variant", variant);
			j_u64("map_size", sizes[si]);
			j_bool("touch", touch);
			j_u64("cycles_per_round", C);
			j_i64("rounds", R);
			j_i64("warmup", W);
			j_i64("cpu", x->cpus[0]);
			j_i64("faults_per_round_min", fmin);
			j_i64("faults_per_round_max", fmax);
			metric("mmap_us", "mmap() mean per round", "us", rmm, R, false);
			metric("munmap_us", "munmap() mean per round", "us", rum, R, false);
			for (size_t i = 0; i < total; i++) {
				mm[i] /= 1e3;
				um[i] /= 1e3;
			}
			opmetric("mmap_us_per_op", "mmap() per op", "us", mm, (int)total);
			opmetric("munmap_us_per_op", "munmap() per op", "us", um, (int)total);
			if (touch) {
				metric("touch_ns", "first touch (fault+access), mean/round", "ns", rtc, R, false);
				metric("reread_ns", "re-touch (no fault), mean/round", "ns", rrr, R, false);
				opmetric("touch_ns_per_op", "first touch per op", "ns", tc, (int)total);
				opmetric("reread_ns_per_op", "re-touch per op", "ns", rr, (int)total);
				hp("  faults per round: min %ld max %ld (expected %u)%s\n",
				   fmin, fmax, C,
				   (fmin != (long)C || fmax != (long)C) ?
				   "   <-- WARNING: unexpected fault count" : "");
			}
			j_close();

			free(mm);
			free(tc);
			free(rr);
			free(um);
			free(smm);
			free(stc);
			free(srr);
			free(sum);
			free(rmm);
			free(rtc);
			free(rrr);
			free(rum);
		}
	}
}

/* ------------------------------------------------------------------ */
/* Mode: mt                                                            */

struct mt;

struct worker {
	pthread_t th;
	struct mt *m;
	int idx, cpu, pin_err;
	size_t s0, s1;			/* touch indices [s0, s1) */
	uint64_t f0, f1, w0, w1;
	long faults, wfaults;
	int err, sig;
	uintptr_t addr;
	uint32_t acc;
};

struct mt {
	struct ctx *x;
	int n;
	uintptr_t va;
	pthread_barrier_t b0, b1, b2;
	int quit;
	int fault_seen;
	unsigned long arrive;		/* spin rendezvous counter, never reset */
	struct worker *w;
};

static inline void cpu_relax(void)
{
#if defined(__x86_64__) || defined(__i386__)
	__builtin_ia32_pause();
#elif defined(__aarch64__)
	__asm__ __volatile__("yield");
#endif
}

/*
 * Workers-only spin rendezvous right before a timed pass: a futex barrier
 * wakes threads tens of microseconds apart, which would inflate the
 * aggregate wall time.  Yields now and then in case CPUs are oversubscribed.
 */
static void spin_rendezvous(struct mt *m, unsigned long *gen)
{
	unsigned long target = ++(*gen) * (unsigned long)m->n;

	__atomic_add_fetch(&m->arrive, 1, __ATOMIC_ACQ_REL);
	for (unsigned int i = 1; __atomic_load_n(&m->arrive, __ATOMIC_ACQUIRE) < target; i++) {
		cpu_relax();
		if (!(i & 1023))
			sched_yield();
	}
}

static void *mt_worker(void *arg)
{
	struct worker *w = arg;
	struct mt *m = w->m;
	const size_t stride = m->x->stride;
	unsigned long gen = 0;

	w->pin_err = pin_self(w->cpu);
	for (;;) {
		struct pass p, q;

		pthread_barrier_wait(&m->b0);
		if (__atomic_load_n(&m->quit, __ATOMIC_ACQUIRE))
			break;
		memset(&p, 0, sizeof(p));
		p.base = m->va + w->s0 * stride;
		p.n = w->s1 - w->s0;
		p.stride = stride;
		q = p;
		w->err = 0;
		spin_rendezvous(m, &gen);
		if (guarded(pass_fn, &p)) {
			w->err = 1;
			w->sig = tl_sig;
			w->addr = (uintptr_t)tl_addr;
			__atomic_store_n(&m->fault_seen, 1, __ATOMIC_RELEASE);
		} else {
			w->f0 = p.t0;
			w->f1 = p.t1;
			w->faults = p.faults;
			w->acc ^= p.acc;
		}
		pthread_barrier_wait(&m->b1);
		/* Every worker sees the same fault_seen here, so all or none rendezvous. */
		if (!__atomic_load_n(&m->fault_seen, __ATOMIC_ACQUIRE)) {
			spin_rendezvous(m, &gen);
			if (guarded(pass_fn, &q)) {
				w->err = 1;
				w->sig = tl_sig;
				w->addr = (uintptr_t)tl_addr;
			} else {
				w->w0 = q.t0;
				w->w1 = q.t1;
				w->wfaults = q.faults;
				w->acc ^= q.acc;
			}
		}
		pthread_barrier_wait(&m->b2);
	}
	return NULL;
}

/* Returns the median aggregate fault rate (faults/s). */
static double run_mt(struct ctx *x, int n, int base_n, double base_rate)
{
	const int R = x->iters ? x->iters : 10, W = x->warmup;
	const size_t S = x->len / x->stride;
	const size_t stride = x->stride;
	double *agg_rate, *agg_ns, *thr_ns, *thr_w, *thr_net, *mm, *um, med;
	long fexp = (long)S, fmin = LONG_MAX, fmax = LONG_MIN, wmax = 0;
	struct layout L = { 0 };
	struct mt m;
	int pin_fail = 0, rc;
	char b1[32], b2[32];

	if (S < (size_t)n) {
		hp("\n[mt] %d threads skipped: only %zu touches in the window\n", n, S);
		return NAN;
	}
	agg_rate = xcalloc(R, sizeof(double));
	agg_ns = xcalloc(R, sizeof(double));
	thr_ns = xcalloc(R, sizeof(double));
	thr_w = xcalloc(R, sizeof(double));
	thr_net = xcalloc(R, sizeof(double));
	mm = xcalloc(R, sizeof(double));
	um = xcalloc(R, sizeof(double));

	memset(&m, 0, sizeof(m));
	m.x = x;
	m.n = n;
	m.w = xcalloc(n, sizeof(*m.w));
	if (pthread_barrier_init(&m.b0, NULL, n + 1) ||
	    pthread_barrier_init(&m.b1, NULL, n + 1) ||
	    pthread_barrier_init(&m.b2, NULL, n + 1))
		die("pthread_barrier_init failed");
	/* Threads (and their mmap()ed stacks) exist before the VA hole is made. */
	for (int i = 0; i < n; i++) {
		struct worker *w = &m.w[i];

		w->m = &m;
		w->idx = i;
		w->cpu = x->cpus[i % x->ncpus];
		w->s0 = S * i / n;
		w->s1 = S * (i + 1) / n;
		rc = pthread_create(&w->th, NULL, mt_worker, w);
		if (rc)
			die("pthread_create: %s", strerror(rc));
	}
	if (n < x->ncpus)
		pin_self(x->cpus[n]);

	rc = layout_make(&L, x->len, SZ_2M, mod_4k_layout(x, x->offset));
	if (rc)
		die("mt: VA reservation failed: %s", strerror(-rc));
	m.va = L.va;

	hp("\n[mt] %d thread%s, %s layout, window %s, stride %s, %zu touches/round, %d rounds (+%d warm-up)\n",
	   n, n == 1 ? "" : "s", x->mt_split ? "split (one mapping per thread)" :
	   "shared (one mapping)", hsize(x->len, b1, sizeof(b1)),
	   hsize(stride, b2, sizeof(b2)), S, R, W);

	for (int r = -W; r < R; r++) {
		uint64_t tm = 0, tu = 0, t, mn = UINT64_MAX, mx = 0;
		double s_t = 0, s_w = 0, s_n = 0;
		long ftot = 0, wtot = 0;

		if (!x->mt_split) {
			rc = bar_map(x, L.va, x->len, x->offset, &tm);
			if (rc)
				die("mt: mmap of window failed: %s", strerror(-rc));
		} else {
			for (int i = 0; i < n; i++) {
				struct worker *w = &m.w[i];

				rc = bar_map(x, L.va + w->s0 * stride,
					     (w->s1 - w->s0) * stride,
					     x->offset + w->s0 * stride, &t);
				if (rc)
					die("mt: mmap of thread %d slice failed: %s", i, strerror(-rc));
				tm += t;
			}
		}
		__atomic_store_n(&m.fault_seen, 0, __ATOMIC_RELAXED);
		pthread_barrier_wait(&m.b0);	/* fault pass runs */
		pthread_barrier_wait(&m.b1);	/* re-read pass runs */
		pthread_barrier_wait(&m.b2);
		if (!x->mt_split) {
			rc = bar_unmap(L.va, x->len, &tu);
			if (rc)
				die("mt: munmap failed: %s", strerror(-rc));
		} else {
			for (int i = 0; i < n; i++) {
				struct worker *w = &m.w[i];

				rc = bar_unmap(L.va + w->s0 * stride,
					       (w->s1 - w->s0) * stride, &t);
				if (rc)
					die("mt: munmap failed: %s", strerror(-rc));
				tu += t;
			}
		}
		for (int i = 0; i < n; i++) {
			struct worker *w = &m.w[i];

			if (w->err)
				die_fault("mt", w->sig, w->addr, L.va, x->offset);
		}
		if (r < 0)
			continue;
		for (int i = 0; i < n; i++) {
			struct worker *w = &m.w[i];
			double cnt = (double)(w->s1 - w->s0);
			double tf = (double)(w->f1 - w->f0) / cnt;
			double tw = (double)(w->w1 - w->w0) / cnt;

			if (w->f0 < mn)
				mn = w->f0;
			if (w->f1 > mx)
				mx = w->f1;
			s_t += tf;
			s_w += tw;
			s_n += tf - tw;
			ftot += w->faults;
			wtot += w->wfaults;
		}
		agg_ns[r] = (double)(mx - mn) / S;
		agg_rate[r] = 1e9 / agg_ns[r];
		thr_ns[r] = s_t / n;
		thr_w[r] = s_w / n;
		thr_net[r] = s_n / n;
		mm[r] = tm / 1e3;
		um[r] = tu / 1e3;
		if (ftot < fmin)
			fmin = ftot;
		if (ftot > fmax)
			fmax = ftot;
		if (wtot > wmax)
			wmax = wtot;
	}

	__atomic_store_n(&m.quit, 1, __ATOMIC_RELEASE);
	pthread_barrier_wait(&m.b0);
	for (int i = 0; i < n; i++) {
		pthread_join(m.w[i].th, NULL);
		g_sink ^= m.w[i].acc;
		if (m.w[i].pin_err)
			pin_fail++;
	}
	pthread_barrier_destroy(&m.b0);
	pthread_barrier_destroy(&m.b1);
	pthread_barrier_destroy(&m.b2);
	layout_free(&L);
	pin_self(x->cpus[0]);

	hp("  CPUs:");
	for (int i = 0; i < n; i++)
		hp("%s%d", i ? "," : " ", m.w[i].cpu);
	hp("%s\n", pin_fail ? "   <-- WARNING: some threads could not be pinned" : "");
	hp("  faults per round: min %ld max %ld (expected %ld); re-touch faults max %ld%s\n",
	   fmin, fmax, fexp, wmax,
	   (fmin != fexp || fmax != fexp || wmax) ? "   <-- WARNING: unexpected fault count" : "");

	j_open(NULL, '{');
	j_str("mode", "mt");
	j_i64("threads", n);
	j_str("layout", x->mt_split ? "split" : "shared");
	j_u64("stride", stride);
	j_u64("touches_per_round", S);
	j_i64("rounds", R);
	j_i64("warmup", W);
	j_open("cpus", '[');
	for (int i = 0; i < n; i++)
		j_i64(NULL, m.w[i].cpu);
	j_close();
	j_i64("pin_failures", pin_fail);
	j_i64("faults_per_round_min", fmin);
	j_i64("faults_per_round_max", fmax);
	j_i64("reread_faults_max", wmax);
	metric("agg_faults_per_sec", "aggregate faults/s (all threads)", "1/s", agg_rate, R, true);
	metric("agg_wall_ns_per_fault", "aggregate wall ns/fault", "ns", agg_ns, R, false);
	metric("thread_ns_per_fault", "per-thread ns/fault (mean of threads)", "ns", thr_ns, R, false);
	metric("thread_reread_ns_per_touch", "per-thread ns/touch re-touch", "ns", thr_w, R, false);
	metric("thread_net_ns_per_fault", "per-thread net ns/fault", "ns", thr_net, R, false);
	metric("mmap_us", x->mt_split ? "mmap() total of slices" : "mmap() of window", "us", mm, R, false);
	metric("munmap_us", x->mt_split ? "munmap() total of slices" : "munmap() of window", "us", um, R, false);
	med = stats_of(agg_rate, R).med;
	if (base_rate > 0 && base_n > 0 && n != base_n) {
		hp("  scaling: %.2fx the %d-thread aggregate rate (ideal %.2fx)\n",
		   med / base_rate, base_n, (double)n / base_n);
		j_dbl("speedup_vs_first", med / base_rate);
		j_i64("speedup_base_threads", base_n);
	}
	j_close();

	free(m.w);
	free(agg_rate);
	free(agg_ns);
	free(thr_ns);
	free(thr_w);
	free(thr_net);
	free(mm);
	free(um);
	return med;
}

/* ------------------------------------------------------------------ */
/* Header                                                              */

static void print_env(struct ctx *x)
{
	char cmdline[4096] = "", thp[256] = "", b1[32], b2[32];
	struct utsname u;

	if (uname(&u))
		memset(&u, 0, sizeof(u));
	read_file("/proc/cmdline", cmdline, sizeof(cmdline));
	read_file("/sys/kernel/mm/transparent_hugepage/enabled", thp, sizeof(thp));

	hp("vfio_bar_bench %s - vfio-pci BAR fault/mmap benchmark\n", TOOL_VERSION);
	hp("  kernel  : %s %s %s %s\n", u.sysname, u.release, u.version, u.machine);
	hp("  host    : %s\n", u.nodename);
	hp("  cmdline : %s\n", cmdline);
	hp("  device  : %s BAR%u via %s\n", x->bdf ? x->bdf : "-", x->bar, x->how);
	hp("  region  : size 0x%" PRIx64 " (%s), vfio offset 0x%" PRIx64 ", flags%s%s%s%s",
	   x->region_size, hsize(x->region_size, b1, sizeof(b1)), x->region_off,
	   x->region_flags & VFIO_REGION_INFO_FLAG_READ ? " READ" : "",
	   x->region_flags & VFIO_REGION_INFO_FLAG_WRITE ? " WRITE" : "",
	   x->region_flags & VFIO_REGION_INFO_FLAG_MMAP ? " MMAP" : "",
	   x->region_flags & VFIO_REGION_INFO_FLAG_CAPS ? " CAPS" : "");
	if (x->bar_phys)
		hp(", phys 0x%" PRIx64, x->bar_phys);
	if (x->nsparse >= 0)
		hp(", %d sparse-mmap areas", x->nsparse);
	hp("\n");
	hp("  window  : BAR offset 0x%" PRIx64 " len 0x%" PRIx64 " (%s)\n",
	   x->offset, x->len, hsize(x->len, b2, sizeof(b2)));
	hp("  backing : /proc/self/maps name \"%s\" => %s\n", x->map_name,
	   x->selftest ? "selftest file" :
	   x->dmabuf_backed ? "DMABUF-backed BAR mmap" : "vfio device file");
	hp("  access  : %s\n", g_write ? "WRITE: 64-bit stores (--write)" : "read: 32-bit loads");
	hp("  cpus    : ");
	for (int i = 0; i < x->ncpus && i < 64; i++)
		hp("%s%d", i ? "," : "", x->cpus[i]);
	hp("%s (device NUMA node %d)%s\n", x->ncpus > 64 ? ",..." : "", x->numa_node,
	   x->cpus_given ? " [--cpus]" : "");
	hp("  clock   : clock_gettime(CLOCK_MONOTONIC) %.1f ns/call%s\n", x->clock_ns,
	   x->clock_ns > 100 ? "  <-- WARNING: slow clock (no vDSO?); per-touch numbers inflated" : "");
	hp("  thp     : sysfs enabled \"%s\" (PFNMAP huge_fault ignores it), PR_SET_THP_DISABLE %s\n",
	   thp, x->thp_disable ? "on: no huge_fault calls" : "off");

	j_str("tool", "vfio_bar_bench");
	j_str("tool_version", TOOL_VERSION);
	j_open("uname", '{');
	j_str("sysname", u.sysname);
	j_str("nodename", u.nodename);
	j_str("release", u.release);
	j_str("access", g_write ? "write" : "read");
	j_str("version", u.version);
	j_str("machine", u.machine);
	j_close();
	j_str("cmdline", cmdline);
	j_str("thp_enabled", thp);
	j_open("device", '{');
	j_str("bdf", x->bdf ? x->bdf : "");
	j_i64("bar", x->bar);
	j_str("open", x->how);
	j_hex("region_size", x->region_size);
	j_hex("region_offset", x->region_off);
	j_hex("region_flags", x->region_flags);
	j_hex("bar_phys", x->bar_phys);
	j_i64("numa_node", x->numa_node);
	j_str("map_name", x->map_name);
	j_bool("dmabuf_backed", x->dmabuf_backed);
	j_close();
	j_open("window", '{');
	j_hex("offset", x->offset);
	j_hex("len", x->len);
	j_close();
	j_open("options", '{');
	j_i64("iters", x->iters);
	j_i64("warmup", x->warmup);
	j_u64("stride", x->stride);
	j_u64("cycles", x->cycles);
	j_str("huge", x->huge_pmd ? "pmd" : "pud");
	j_str("mt_layout", x->mt_split ? "split" : "shared");
	j_bool("thp_disable", x->thp_disable);
	j_iarr("threads", x->tlist, x->ntlist);
	j_close();
	j_iarr("cpus", x->cpus, x->ncpus);
	j_dbl("clock_ns_per_call", x->clock_ns);
}

/* ------------------------------------------------------------------ */
/* main                                                                */

static void usage(FILE *f)
{
	fprintf(f,
"usage: vfio_bar_bench --bdf DDDD:BB:DD.F --bar N [--offset OFF] [--len LEN] [options]\n"
"\n"
"vfio-pci BAR mmap fault/mmap benchmark.  OFF and LEN select the BAR window to\n"
"touch (bytes; 0x.. and K/M/G/T suffixes accepted).  The window should be device\n"
"memory: reads of registers can have side effects.\n"
"\n"
"  --bdf BDF          PCI device already bound to vfio-pci (required)\n"
"  --bar N            BAR index 0-5 (required)\n"
"  --offset OFF       window offset into the BAR (default 0, 4K aligned)\n"
"  --len LEN          window length (default: rest of the BAR, at most 1G;\n"
"                     multiple of 4K; huge-fault modes need 2M/1G-aligned room)\n"
"  --mode LIST        comma list: probe fault4k faulthuge mmapcycle mt all\n"
"                     (default: all = fault4k,faulthuge,mmapcycle,mt)\n"
"  --iters N          measured rounds per mode (defaults: fault4k 10,\n"
"                     faulthuge 30, mmapcycle 10, mt 10)\n"
"  --warmup N         discarded warm-up rounds per mode (default 1)\n"
"  --threads LIST     mt thread counts, e.g. 1,2,4,8 (default 1,2,4,8,16,32\n"
"                     capped at the number of usable CPUs)\n"
"  --cpus LIST        CPUs to pin to, in order (default: device-local CPUs,\n"
"                     one SMT sibling per core first)\n"
"  --stride BYTES     fault4k/mt distance between touches (default 4K; 2M\n"
"                     makes every touch the first in a fresh 2M block)\n"
"  --huge pud|pmd     faulthuge layout: VA==PA mod 1G (pud, default) or\n"
"                     VA==PA mod 2M but not 1G (pmd: PUD impossible)\n"
"  --cycles N         mmapcycle cycles per round (default 1000)\n"
"  --mt-layout L      shared (default; one mapping) or split (one mapping\n"
"                     per thread)\n"
"  --thp-disable      prctl(PR_SET_THP_DISABLE): no huge_fault calls at all\n"
"  --write            touch with 64-bit stores (PROT_READ|PROT_WRITE) instead of\n"
"                     loads.  WRITES TO THE DEVICE: the window must be memory\n"
"                     that is safe to overwrite.  Needs explicit --offset/--len.\n"
"  --prefill          before measuring, write a pattern over the window and read\n"
"                     back a sample (for ECC memories that must be written before\n"
"                     they are read).  Same caveats as --write.\n"
"  --group            use the group/container path instead of cdev+iommufd\n"
"  --json             JSON on stdout; human-readable output goes to stderr\n"
"  --selftest FILE    plumbing self-test: map FILE instead of a VFIO BAR\n"
"                     (no device access; numbers are meaningless)\n"
"  -h, --help\n");
}

static unsigned int parse_modes(const char *s)
{
	char buf[256], *tok, *save = NULL;
	unsigned int m = 0;

	snprintf(buf, sizeof(buf), "%s", s);
	for (tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
		if (!strcmp(tok, "probe"))
			m |= M_PROBE;
		else if (!strcmp(tok, "fault4k"))
			m |= M_FAULT4K;
		else if (!strcmp(tok, "faulthuge"))
			m |= M_FAULTHUGE;
		else if (!strcmp(tok, "mmapcycle"))
			m |= M_MMAPCYCLE;
		else if (!strcmp(tok, "mt"))
			m |= M_MT;
		else if (!strcmp(tok, "all"))
			m |= M_FAULT4K | M_FAULTHUGE | M_MMAPCYCLE | M_MT;
		else
			return 0;
	}
	return m;
}

static int parse_int(const char *s, int lo, int hi, const char *what)
{
	char *end;
	long v;

	errno = 0;
	v = strtol(s, &end, 0);
	if (errno || end == s || *end || v < lo || v > hi) {
		fprintf(stderr, "invalid %s: %s\n", what, s);
		exit(2);
	}
	return (int)v;
}

int main(int argc, char **argv)
{
	static const struct option opts[] = {
		{ "bdf", required_argument, NULL, 'b' },
		{ "bar", required_argument, NULL, 'B' },
		{ "offset", required_argument, NULL, 'o' },
		{ "len", required_argument, NULL, 'l' },
		{ "mode", required_argument, NULL, 'm' },
		{ "iters", required_argument, NULL, 'i' },
		{ "warmup", required_argument, NULL, 'w' },
		{ "threads", required_argument, NULL, 't' },
		{ "cpus", required_argument, NULL, 'c' },
		{ "stride", required_argument, NULL, 's' },
		{ "huge", required_argument, NULL, 'H' },
		{ "cycles", required_argument, NULL, 'C' },
		{ "mt-layout", required_argument, NULL, 'L' },
		{ "thp-disable", no_argument, NULL, 'T' },
		{ "write", no_argument, NULL, 'W' },
		{ "prefill", no_argument, NULL, 'P' },
		{ "group", no_argument, NULL, 'g' },
		{ "json", no_argument, NULL, 'j' },
		{ "selftest", required_argument, NULL, 'S' },
		{ "help", no_argument, NULL, 'h' },
		{ NULL, 0, NULL, 0 },
	};
	static struct ctx X;
	struct ctx *x = &X;
	double base_rate = 0;
	int base_n = 0, opt;

	x->bar = UINT_MAX;
	x->stride = SZ_4K;
	x->warmup = 1;
	x->cycles = 1000;
	x->modes = M_FAULT4K | M_FAULTHUGE | M_MMAPCYCLE | M_MT;
	x->fd = x->iommufd = x->container = x->group = -1;
	x->numa_node = -1;

	while ((opt = getopt_long(argc, argv, "h", opts, NULL)) != -1) {
		switch (opt) {
		case 'b':
			x->bdf = optarg;
			break;
		case 'B':
			x->bar = (unsigned int)parse_int(optarg, 0, 5, "--bar");
			break;
		case 'o':
			if (parse_size(optarg, &x->offset)) {
				fprintf(stderr, "invalid --offset: %s\n", optarg);
				return 2;
			}
			x->have_offset = true;
			break;
		case 'l':
			if (parse_size(optarg, &x->len)) {
				fprintf(stderr, "invalid --len: %s\n", optarg);
				return 2;
			}
			x->have_len = true;
			break;
		case 'm':
			x->modes = parse_modes(optarg);
			if (!x->modes) {
				fprintf(stderr, "invalid --mode: %s\n", optarg);
				return 2;
			}
			break;
		case 'i':
			x->iters = parse_int(optarg, 1, 1000000, "--iters");
			break;
		case 'w':
			x->warmup = parse_int(optarg, 0, 1000, "--warmup");
			break;
		case 't':
			x->ntlist = parse_list(optarg, x->tlist, MAX_TLIST);
			if (x->ntlist <= 0) {
				fprintf(stderr, "invalid --threads: %s\n", optarg);
				return 2;
			}
			for (int i = 0; i < x->ntlist; i++)
				if (x->tlist[i] < 1 || x->tlist[i] > MAX_CPUS) {
					fprintf(stderr, "invalid --threads: %s\n", optarg);
					return 2;
				}
			break;
		case 'c':
			x->ncpus = parse_list(optarg, x->cpus, MAX_CPUS);
			if (x->ncpus <= 0) {
				fprintf(stderr, "invalid --cpus: %s\n", optarg);
				return 2;
			}
			x->cpus_given = true;
			break;
		case 's':
			if (parse_size(optarg, &x->stride)) {
				fprintf(stderr, "invalid --stride: %s\n", optarg);
				return 2;
			}
			break;
		case 'H':
			if (!strcmp(optarg, "pud")) {
				x->huge_pmd = false;
			} else if (!strcmp(optarg, "pmd")) {
				x->huge_pmd = true;
			} else {
				fprintf(stderr, "invalid --huge: %s\n", optarg);
				return 2;
			}
			break;
		case 'C':
			x->cycles = (unsigned int)parse_int(optarg, 1, 10000000, "--cycles");
			break;
		case 'L':
			if (!strcmp(optarg, "shared")) {
				x->mt_split = false;
			} else if (!strcmp(optarg, "split")) {
				x->mt_split = true;
			} else {
				fprintf(stderr, "invalid --mt-layout: %s\n", optarg);
				return 2;
			}
			break;
		case 'T':
			x->thp_disable = true;
			break;
		case 'W':
			g_write = true;
			break;
		case 'P':
			g_prefill = true;
			break;
		case 'g':
			x->force_group = true;
			break;
		case 'j':
			x->json = true;
			break;
		case 'S':
			x->selftest = optarg;
			break;
		case 'h':
			usage(stdout);
			return 0;
		default:
			usage(stderr);
			return 2;
		}
	}
	if (optind != argc || (!x->selftest && (!x->bdf || x->bar == UINT_MAX))) {
		fprintf(stderr, "--bdf and --bar are required\n\n");
		usage(stderr);
		return 2;
	}
	if (x->selftest && x->bar == UINT_MAX)
		x->bar = 0;
	if ((g_write || g_prefill) && (!x->have_offset || !x->have_len)) {
		fprintf(stderr, "--write and --prefill need an explicit --offset and --len\n");
		return 2;
	}

	hout = x->json ? stderr : stdout;
	if (x->json)
		j_begin();

	/* Keep malloc out of mmap() so it never lands in a reserved VA hole. */
	mallopt(M_MMAP_MAX, 0);
	install_fault_guard();
	if (x->thp_disable && prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0))
		die("prctl(PR_SET_THP_DISABLE): %s", strerror(errno));

	open_device(x);
	region_info(x);
	validate_window(x);
	if (!x->cpus_given)
		auto_cpus(x);
	if (pin_self(x->cpus[0]))
		hp("warning: could not pin to CPU %d\n", x->cpus[0]);
	if (!x->ntlist) {
		static const int def[] = { 1, 2, 4, 8, 16, 32 };

		for (size_t i = 0; i < sizeof(def) / sizeof(def[0]); i++)
			if (def[i] <= x->ncpus || !i)
				x->tlist[x->ntlist++] = def[i];
	}
	x->clock_ns = clock_overhead_ns();
	detect_backing(x);
	print_env(x);
	if (g_prefill)
		prefill_window(x);

	j_open("results", '[');
	if (x->modes & M_PROBE)
		run_probe(x);
	if (x->modes & M_FAULT4K)
		run_fault4k(x);
	if (x->modes & M_FAULTHUGE)
		run_faulthuge(x);
	if (x->modes & M_MMAPCYCLE)
		run_mmapcycle(x);
	if (x->modes & M_MT) {
		for (int i = 0; i < x->ntlist; i++) {
			double rate;

			if (x->tlist[i] > x->ncpus)
				hp("\nnote: %d threads > %d CPUs; CPUs will be shared\n",
				   x->tlist[i], x->ncpus);
			rate = run_mt(x, x->tlist[i], base_n, base_rate);
			if (i == 0 && isfinite(rate)) {
				base_rate = rate;
				base_n = x->tlist[i];
			}
		}
	}
	j_close();
	j_finish(NULL);
	hp("\ndone (sink 0x%08" PRIx32 ")\n", g_sink);
	return 0;
}
