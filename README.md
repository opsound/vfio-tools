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
