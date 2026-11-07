#pragma once
// ==========================================================================
// KIVI: output layout for the quantize kernels.
//
// The kernels read a contiguous input chunk and write packed bytes plus one
// scale/zero per group. Group `gid` (0 .. num_groups-1, in input order)
// always lands at
//
//     packed[row * out_row_stride  + g * (group_size / 4)]
//     meta  [row * meta_row_stride + g]
//     with  row = gid / groups_per_row,  g = gid % groups_per_row
//
// For a contiguous output, groups_per_row = num_groups and the strides are
// irrelevant (row is always 0) — exactly the original flat addressing.
//
// For a chunk written into a preallocated arena (Phase 1 §B.2), the output
// tensors are *views* into the arena and the ops layer derives the layout
// from their strides:
//   keys   arena [B,H,D,cap/4]  — a row is one (b,h,d) channel; the chunk's
//          groups along tokens are contiguous within it.
//   values arena [B,H,cap,D/4]  — a row is one (b,h) head; the chunk's
//          tokens (and their D/G groups) are contiguous within it.
// The chunk's starting token offset is folded into the views' data
// pointers, so it is not a field here.
//
// All index math is int64: arena offsets (row * stride) outgrow int32 long
// before a single chunk's element count does.
// ==========================================================================

#include <cstdint>

namespace kivi {

struct KiviLayout {
    int64_t groups_per_row;   // groups written contiguously per output row
    int64_t out_row_stride;   // packed bytes between consecutive rows
    int64_t meta_row_stride;  // scale/zero elements between consecutive rows

    static KiviLayout contiguous(int64_t num_groups) {
        return KiviLayout{num_groups > 0 ? num_groups : 1, 0, 0};
    }
};

}  // namespace kivi
