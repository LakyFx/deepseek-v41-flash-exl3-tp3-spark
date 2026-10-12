# Qualified host environment

DS41F 1.0.0 was measured on three GB10 hosts with Linux `7.0.0-1019-nvidia-64k`, 65,536-byte pages, NVIDIA driver `580.178.04`, performance CPU governors, working RoCE v2 and the signed memory-saver module with packing enabled. Secure Boot remained enabled. This is the **host OS kernel**, not the CUDA toolkit version or a new quantization.

The memory-saver source was pinned to [christopherowen/dgx-spark-memory-saver at 6f97319](https://github.com/christopherowen/dgx-spark-memory-saver/tree/6f97319e7c00e5422b0afe4cabbd348211bfcb34). The supporting recipe research snapshot was [spark-ds41f at 459a2e1](https://github.com/christopherowen/spark-ds41f/tree/459a2e1e40607fcc3de3490050a00fc18f66f62b). Benchmark provenance uses a separate older commit, deliberately pinned in [OWEN-BENCHMARK.md](OWEN-BENCHMARK.md).

The module changes host memory allocation/packing behavior. It does not change EXL3 precision, replace the KV format or guarantee faster kernels by itself. Our migration included a loader compatibility fix; that fix is included in the runtime sources. The B0 campaign baseline already uses this environment, so C4-versus-B0 gains are not attributed to the kernel upgrade.

## Verify before loading the model

On every rank, record these read-only checks together with your image digest:

```bash
uname -r
getconf PAGESIZE
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
mokutil --sb-state
lsmod
ip -br address
ibv_devices
ibv_devinfo
```

Also verify the installed module's own packing state using the pinned upstream instructions, the trusted module signer, available RAM, rank-local Engram files and each read-only body mount. A module file on disk is not proof that its packing behavior is active. A 64K page-size kernel and the ordinary 4K-page kernel are different boot choices.

Host provisioning is deliberately separate from the container launcher. Follow the pinned upstream module build/install instructions for your kernel and driver. With Secure Boot, sign the module using your own enrolled key. If the key is not yet enrolled, enrollment requires the firmware/MOK confirmation on that machine; never use somebody else's private signing key. Make packing activation persistent through the upstream-supported mechanism and check it after an operator-approved reboot before measuring.

The original installation used a signed module build with packing default-on. This repository records that requirement and source pin; it does not distribute private keys, machine-specific signed modules, bootloader state or a general-purpose unattended kernel updater. A fresh installation must qualify its own host setup. Do not assume that a differently packaged or newer module reproduces this environment simply because it loads successfully.

## Startup and rollback

Preserve the prior image, launch configuration and independent writable caches. Bring up all three ranks with the same source/profile/image, wait for rendezvous and model loading, check reported KV pool and model health, then expose the gateway. Never reopen ingress based only on one worker container being alive. If weights are shared over NFS, each worker must wait for and validate the required mount, not load an empty mountpoint.

The production installation uses operator-specific boot orchestration to enforce these dependencies. The public foreground launcher prints or runs one rank and does not install systemd units, reboot machines or promise unattended recovery on an unknown network. Keep gateway startup dependent on the complete cluster's health when integrating it into your own service manager. Restoring an image is separate from rolling back a host kernel; retain both paths until your own new-host qualification passes.

## Separate migration measurements

The retained [Hermes migration comparison](../results/20261011-host-migration/HERMES-COMPARISON-r1.json) and [Owen migration summary](../results/20261011-host-migration/OWEN-SUMMARY-r1.json) precede the B0/N/C sweep. Their scopes and incomplete phases remain in the files. Do not merge that host-migration comparison with the later same-host candidate campaign or treat its historical MAX reference as a fresh paired baseline.
