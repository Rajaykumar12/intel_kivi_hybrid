"""Preallocated packed-KV arena for one layer (Phase 1 step 10).

Replaces the per-flush ``torch.cat`` of the whole packed history (which made
total copy volume O(T^2/G) over a generation) with an append into spare
capacity. On Iris Xe the arena lives in USM host memory, so the quantize
kernels write into it directly and the CPU dequant path reads it in place:
no host<->device copies at all.

Layout (see docs/optimization_roadmap/phase1_memory_and_metadata.md §B.2):

    keys,   channel-major: packed [B,H,D,cap/4] u8,  scales/zeros [B,H,D,cap/G]
    values, token-major:   packed [B,H,cap,D/4] u8,  scales/zeros [B,H,cap,D/G]

Key rows (one per (b,h,d) channel) run along the token axis. They can be
padded to a byte multiple (``row_align``) so every channel starts on a cache
line, but that is off by default: measured on the integration test shapes,
64-byte padding of the scale/zero rows (cap/G * 2 bytes each — 12 bytes at
a 192-token capacity) inflated the resident store more than the whole spare
capacity did, and its benefit to the CPU dequant path is unmeasured. Base
allocations are 64-byte aligned regardless (alloc_usm_host). Value rows (one
per (b,h) head) are already contiguous. The kernels learn the strides from
the views they are handed.

Capacity is tracked in tokens and is always a multiple of G. ``used`` tokens
are valid; the rest is spare. Byte accounting covers the full allocation,
padding included — that memory really is resident.
"""

from typing import Callable, Optional

import torch

__all__ = ["LayerArena"]

_MIN_GROW_GROUPS = 4


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def grow_capacity(cap: int, needed: int, G: int) -> int:
    """Next capacity when ``needed`` tokens no longer fit in ``cap``.

    Grows by max(4 groups, cap/4): geometric enough that total copy volume
    over a generation stays O(T), and bounded enough that the spare tail
    (counted against compression) is at most ~25% of the store, or a few
    groups for short sequences.
    """
    target = max(needed, cap + max(_MIN_GROW_GROUPS * G, cap // 4))
    return _round_up(target, G)


class LayerArena:
    def __init__(self, B: int, H: int, D: int, G: int, cap: int,
                 meta_dtype: torch.dtype,
                 alloc: Callable[[tuple, torch.dtype], torch.Tensor],
                 row_align: int = 1):
        assert cap > 0 and cap % G == 0, (cap, G)
        self.row_align = row_align
        self.B, self.H, self.D, self.G = B, H, D, G
        self.cap = cap
        self.used = 0
        self.meta_dtype = meta_dtype
        self._alloc = alloc
        self._bases = []

        # Keys: rows along the token axis (optionally padded, see module doc).
        self.packed_k = self._rows((B, H, D), cap // 4, torch.uint8)
        self.scales_k = self._rows((B, H, D), cap // G, meta_dtype)
        self.zeros_k = self._rows((B, H, D), cap // G, meta_dtype)
        # Values: natural layout, tokens contiguous within each head.
        self.packed_v = self._plain((B, H, cap, D // 4), torch.uint8)
        self.scales_v = self._plain((B, H, cap, D // G), meta_dtype)
        self.zeros_v = self._plain((B, H, cap, D // G), meta_dtype)

    # -- allocation ---------------------------------------------------------

    def _rows(self, outer, row_elems, dtype):
        elt = torch.tensor([], dtype=dtype).element_size()
        padded = _round_up(row_elems * elt, self.row_align) // elt
        base = self._alloc(tuple(outer) + (padded,), dtype)
        self._bases.append(base)
        return base[..., :row_elems]

    def _plain(self, shape, dtype):
        base = self._alloc(tuple(shape), dtype)
        self._bases.append(base)
        return base

    @property
    def allocated_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self._bases)

    # -- views ----------------------------------------------------------------

    def key_slices(self, t0: int, t1: int):
        """(packed, scales, zeros) views covering tokens [t0, t1)."""
        G = self.G
        return (self.packed_k[..., t0 // 4:t1 // 4],
                self.scales_k[..., t0 // G:t1 // G],
                self.zeros_k[..., t0 // G:t1 // G])

    def value_slices(self, t0: int, t1: int):
        return (self.packed_v[:, :, t0:t1],
                self.scales_v[:, :, t0:t1],
                self.zeros_v[:, :, t0:t1])

    def used_keys(self):
        return self.key_slices(0, self.used)

    def used_values(self):
        return self.value_slices(0, self.used)

    def used_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for t in self.used_keys() + self.used_values())

    # -- growth ---------------------------------------------------------------

    def grown(self, new_cap: int) -> "LayerArena":
        """A larger arena holding a copy of this one's used tokens. The caller
        must ensure no kernel is still writing into this arena."""
        assert new_cap >= self.cap and new_cap % self.G == 0
        new = LayerArena(self.B, self.H, self.D, self.G, new_cap,
                         self.meta_dtype, self._alloc, self.row_align)
        if self.used:
            for dst, src in zip(new.key_slices(0, self.used) + new.value_slices(0, self.used),
                                self.used_keys() + self.used_values()):
                dst.copy_(src)
        new.used = self.used
        return new


def ensure_capacity(arena: Optional[LayerArena], needed: int, *, B: int, H: int,
                    D: int, G: int, meta_dtype: torch.dtype, reserve: int,
                    alloc) -> LayerArena:
    """Return an arena with room for ``needed`` tokens, allocating or growing
    as required. ``reserve`` (a token count, 0 if unknown) sizes the first
    allocation and any growth up front when the caller knows the final
    length, so the arena never needs to grow and ends with no spare tail."""
    floor = max(needed, reserve)
    if arena is None:
        # First allocation: exactly what this flush (or the reserve) needs.
        return LayerArena(B, H, D, G, _round_up(floor, G), meta_dtype, alloc)
    if needed <= arena.cap:
        return arena
    cap = _round_up(floor, G) if reserve >= needed else grow_capacity(arena.cap, needed, G)
    return arena.grown(cap)
