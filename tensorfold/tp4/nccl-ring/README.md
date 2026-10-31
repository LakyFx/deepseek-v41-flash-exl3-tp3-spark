# Necessary switchless TP4 runtime

The first live TP4 startup exposed two missing NCCL requirements: the old
listener advertises only its selected NIC's GIDs, which can face the wrong
neighbor; Tree/PAT setup also attempts uncabled diagonal links. Merely setting
`NCCL_ALGO=Ring` does not remove those startup connections.

This directory preserves FujitsuPolycom/sparkring's cumulative dual PCI-domain
patch and its upstream hash manifest. It includes the cycle and Tree/PAT
changes; do not layer another cycle patch on it. Source:
https://github.com/FujitsuPolycom/sparkring/blob/main/spark_transport/nccl/DUAL_PCI_DOMAIN.md

The separate CTS acquire fence is Stanislav Bardyuk's credited backport of
NVIDIA/nccl#2393, obtained from knapcio's recipe. It orders the RDMA-updated
CTS loads on AArch64. These are third-party fixes, not our inventions.

The builder copies the existing NCCL source/build into a new directory,
checks both patches, runs the upstream CPU routing-handle checks and builds
SM121 with our CUDA 13.0 toolchain. It records input/output hashes and never
replaces the old TP3 library. GPU/model checks remain separate.

Use the candidate with `NCCL_SWITCHLESS_RING_ONLY=1`, `NCCL_ALGO=Ring`,
`NCCL_IB_EXTENDED_IPV4_GIDS=1`, `NCCL_IB_PRESERVE_PCI_DOMAIN=1`, and existing
subnet-aware routing. Record actual connected QPs; NIC discovery alone does
not prove use of both domains. The transport patch does not change kernels,
weight values, sampling, KV policy or adaptive speculation.
