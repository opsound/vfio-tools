# vfio-tools

Small throwaway programs and scripts for experimenting with VFIO and iommufd.

## vfio_dmabuf_reset

Demonstrates dma-buf export from vfio-pci, import into an RDMA device, and
the revoke/un-revoke lifecycle across `VFIO_DEVICE_RESET`:

1. Open a vfio-pci device via cdev + iommufd bind.
2. Export a dma-buf for a slice of BAR4 (`VFIO_DEVICE_FEATURE_DMA_BUF`).
3. Import it into an RDMA device (`ibv_reg_dmabuf_mr`).
4. `VFIO_DEVICE_RESET` with the dma-buf and MR held open.
5. Re-map the *same* revoked attachment (`ibv_advise_mr` prefetch) and
   re-import the *same* dma-buf fd — both succeed after un-revoke.
6. Close the device (permanent revoke) and show re-map now fails.

Build:

    make

Run:

    ./vfio_dmabuf_reset BDF IB_DEV

`BDF` is the PCI address of the vfio-pci device (e.g. `0000:23:00.0`).
`IB_DEV` is an RDMA device name (e.g. `mlx5_0`) or index. For P2P, pick
devices under the same root complex.

## vfio_dmabuf_reset.bt

bpftrace companion to the above: traces the in-kernel export, attach, map,
revoke, re-map, and teardown sequence while the test runs.

    sudo bpftrace vfio_dmabuf_reset.bt

then run the test in another terminal.

## vfio_bar_bench

Measures what the kernel spends on mmap()ing a vfio-pci BAR and faulting it
in, for A/B comparisons of kernels that change the vfio-pci BAR mmap or fault
path. It covers single-threaded order-0 faults, PMD/PUD huge faults,
mmap()/munmap() cycles, and N threads faulting one shared mapping (or one
mapping each) at the same time.

Build:

    make vfio_bar_bench

Run as root against any device already bound to vfio-pci, with nothing else
holding it open:

    sudo ./vfio_bar_bench --bdf BDF --bar N [--offset OFF] [--len LEN] [options]

`--offset`/`--len` select the window inside the BAR (default: offset 0, the
rest of the BAR up to 1 GiB). The window should be device memory: reading
register BARs can have side effects. Touches are 32-bit loads through a
`PROT_READ` mapping by default. `--write` touches with 64-bit stores instead,
and `--prefill` writes a pattern over the window once before measuring (for
memories with ECC that must be written before they are read). Both write to
the device and need an explicit `--offset` and `--len`. No DMA is set up and
config space and interrupts are not touched, but the kernel's vfio open/close
path may reset the function, as for any vfio user.

Modes (`--mode`, comma-separated; default `fault4k,faulthuge,mmapcycle,mt`):

| mode | what it does |
|---|---|
| `probe` | one load at the first and last page of the window |
| `fault4k` | VA = PA + 4K (mod 2M), so every touch is an order-0 fault; one touch per `--stride` (default 4K; `2M` makes every fault the first in a fresh 2M block) |
| `faulthuge` | VA = PA (mod 1G) for PUD faults, or `--huge pmd` for PMD only; one touch per 2M |
| `mmapcycle` | mmap()/munmap() of 4K and 2M mappings in a loop, with and without one touch |
| `mt` | `--threads 1,2,4,...` pinned threads fault disjoint slices of one mapping (`--mt-layout split`: one mapping per thread) |

Each round re-creates the mapping, runs a fault pass and then a re-touch pass
over the same pages; `net_ns_per_fault` subtracts the re-touch cost, which
removes the device access itself (an uncached load costs ~0.5-1 us; stores are
posted and cheap). `--thp-disable` sets `PR_SET_THP_DISABLE`, which suppresses
all huge-fault attempts. Threads are pinned to device-local CPUs unless
`--cpus` is given. `--json` writes machine-readable results to stdout.
`--selftest FILE` maps a regular file instead of a BAR, to check the plumbing
without a device (its numbers are meaningless).

A/B comparison of two kernels, booting each in turn:

    mkdir -p out/$(uname -r)
    sudo ./vfio_bar_bench --bdf BDF --bar N --offset OFF --len 1G \
        --mode mt --threads 1,4,16,32 --json > out/$(uname -r)/mt.json
    # ... same commands on the other kernel, then:
    ./vfio_bar_bench_compare.py out/KERNEL_A out/KERNEL_B
