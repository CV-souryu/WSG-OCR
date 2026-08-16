// mega.wgsl — one workgroup per glyph runs the entire TinyCNN:
//
//   packed uint8 [24,24] glyph
//     -> normalize (byte/255, channel 0)
//     -> conv1  (3x3, SAME, stride 2, ReLU)   12x12x8
//     -> dw1    (3x3 depthwise, SAME, ReLU)   12x12x8
//     -> pw1    (1x1, stride 2, ReLU)         6x6x16
//     -> dw2    (3x3 depthwise, SAME, ReLU)   6x6x16
//     -> pw2    (1x1, stride 2, ReLU)         3x3x32
//     -> gap    (global average pool)         32
//     -> MODE_CLASSIFY:    dense + top-1/top-2 reduction -> 12-byte record
//     -> MODE_LOGITS:      dense -> full [C] logits
//     -> MODE_TOPK:        dense -> masked top-7 (ids+logits) -> 14-u32 record
//     -> MODE_GATHER:      dense -> logits for M requested ids -> [N*M] f32
//     -> MODE_TOPK_GATHER: masked top-7 + gather the template top-K ids
//                          (binding 5) -> 24-u32 record (T2 stage path)
//
// All intermediate tensors live in workgroup shared memory: the host uploads
// the packed glyph batch once, submits ONE dispatch (n workgroups), and reads
// the final results once — zero intermediate copy-backs.
//
// Per-layer f32 accumulation order replicates the verified per-layer shaders
// (conv3x3/dwconv3x3/pointwise/gap/linear_argmax), so the numeric contract
// against the numpy reference is unchanged.
//
// T2 (DP Top-K 回灌): MODE_TOPK/MODE_TOPK_GATHER read back 7 ranks instead
// of the full [C] logits — the host uses ranks 6-7 (k+1/k+2 for k=5) as the
// tie-boundary guard and falls back to a full readback when they are within
// 1e-4. MODE_GATHER computes only the requested (glyph, class) dots, so the
// hybrid fusion can fetch template-only logits without a full readback.

struct Params {
    a: vec4<u32>,  // n, num_classes, mode (0-4), M (gather count)
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<u32>;
// Mode 2/4: allowed-class bitmask (words). Mode 3: requested ids [N*M].
@group(0) @binding(4) var<storage, read> aux: array<u32>;
// Mode 4: template match records (15 u32 per glyph) for the gather ids.
@group(0) @binding(5) var<storage, read> template_results: array<u32>;

// Fused weight-buffer layout (floats):
//   conv1.w(72) conv1.b(8) dw1.w(72) dw1.b(8) pw1.w(128) pw1.b(16)
//   dw2.w(144) dw2.b(16) pw2.w(512) pw2.b(32) fc.w(C*32) fc.b(C)
const OFF_CONV1W: u32 = 0u;
const OFF_CONV1B: u32 = 72u;
const OFF_DW1W: u32 = 80u;
const OFF_DW1B: u32 = 152u;
const OFF_PW1W: u32 = 160u;
const OFF_PW1B: u32 = 288u;
const OFF_DW2W: u32 = 304u;
const OFF_DW2B: u32 = 448u;
const OFF_PW2W: u32 = 464u;
const OFF_PW2B: u32 = 976u;
const OFF_FCW: u32 = 1008u;  // fc.b at 1008 + C*32

var<workgroup> sm_c1: array<f32, 1152>;  // 12*12*8
var<workgroup> sm_d1: array<f32, 1152>;  // 12*12*8
var<workgroup> sm_p1: array<f32, 576>;   // 6*6*16
var<workgroup> sm_d2: array<f32, 576>;   // 6*6*16
var<workgroup> sm_p2: array<f32, 288>;   // 3*3*32
var<workgroup> sm_gap: array<f32, 32>;
var<workgroup> sm_score: array<f32, 64>;
var<workgroup> sm_id: array<u32, 64>;
var<workgroup> sm_topv: array<f32, 448>;  // 64 threads x 7 top-logits
var<workgroup> sm_topi: array<u32, 448>;  // 64 threads x 7 top-ids

fn glyph_byte(base_words: u32, pos: u32) -> f32 {
    let word = input[base_words + pos / 4u];
    let byte = (word >> ((pos % 4u) * 8u)) & 0xffu;
    return f32(byte) * (1.0 / 255.0);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let g = wgid.x;
    let tid = lid.x;
    let num_classes = params.a.y;
    let mode = params.a.z;
    let in_words = g * 144u;  // 24*24 bytes = 144 u32 words

    // ---- conv1: 12x12x8 from packed glyph bytes, SAME, stride 2, ReLU ----
    for (var i = tid; i < 1152u; i += 64u) {
        let c = i % 8u;
        let px = (i / 8u) % 12u;
        let py = i / 96u;
        var acc = weights[OFF_CONV1B + c];
        for (var dy = 0u; dy < 3u; dy++) {
            let iy = i32(py * 2u) + i32(dy) - 1;
            if (iy < 0 || iy >= 24) { continue; }
            for (var dx = 0u; dx < 3u; dx++) {
                let ix = i32(px * 2u) + i32(dx) - 1;
                if (ix < 0 || ix >= 24) { continue; }
                let v = glyph_byte(in_words, u32(iy) * 24u + u32(ix));
                acc += v * weights[OFF_CONV1W + c * 9u + dy * 3u + dx];
            }
        }
        sm_c1[i] = max(acc, 0.0);
    }
    workgroupBarrier();

    // ---- dw1: 12x12x8 depthwise, SAME, stride 1, ReLU ----
    for (var i = tid; i < 1152u; i += 64u) {
        let c = i % 8u;
        let px = (i / 8u) % 12u;
        let py = i / 96u;
        var acc = weights[OFF_DW1B + c];
        for (var dy = 0u; dy < 3u; dy++) {
            let iy = i32(py) + i32(dy) - 1;
            if (iy < 0 || iy >= 12) { continue; }
            for (var dx = 0u; dx < 3u; dx++) {
                let ix = i32(px) + i32(dx) - 1;
                if (ix < 0 || ix >= 12) { continue; }
                acc += sm_c1[(u32(iy) * 12u + u32(ix)) * 8u + c]
                    * weights[OFF_DW1W + c * 9u + dy * 3u + dx];
            }
        }
        sm_d1[i] = max(acc, 0.0);
    }
    workgroupBarrier();

    // ---- pw1: 1x1, stride 2 -> 6x6x16, ReLU ----
    for (var i = tid; i < 576u; i += 64u) {
        let c = i % 16u;
        let px = (i / 16u) % 6u;
        let py = i / 96u;
        let base = ((py * 2u) * 12u + px * 2u) * 8u;  // iy = py*2, ix = px*2
        var acc = weights[OFF_PW1B + c];
        for (var ic = 0u; ic < 8u; ic++) {
            acc += sm_d1[base + ic] * weights[OFF_PW1W + c * 8u + ic];
        }
        sm_p1[i] = max(acc, 0.0);
    }
    workgroupBarrier();

    // ---- dw2: 6x6x16 depthwise, SAME, stride 1, ReLU ----
    for (var i = tid; i < 576u; i += 64u) {
        let c = i % 16u;
        let px = (i / 16u) % 6u;
        let py = i / 96u;
        var acc = weights[OFF_DW2B + c];
        for (var dy = 0u; dy < 3u; dy++) {
            let iy = i32(py) + i32(dy) - 1;
            if (iy < 0 || iy >= 6) { continue; }
            for (var dx = 0u; dx < 3u; dx++) {
                let ix = i32(px) + i32(dx) - 1;
                if (ix < 0 || ix >= 6) { continue; }
                acc += sm_p1[(u32(iy) * 6u + u32(ix)) * 16u + c]
                    * weights[OFF_DW2W + c * 9u + dy * 3u + dx];
            }
        }
        sm_d2[i] = max(acc, 0.0);
    }
    workgroupBarrier();

    // ---- pw2: 1x1, stride 2 -> 3x3x32, ReLU ----
    for (var i = tid; i < 288u; i += 64u) {
        let c = i % 32u;
        let px = (i / 32u) % 3u;
        let py = i / 96u;
        let base = ((py * 2u) * 6u + px * 2u) * 16u;  // iy = py*2, ix = px*2
        var acc = weights[OFF_PW2B + c];
        for (var ic = 0u; ic < 16u; ic++) {
            acc += sm_d2[base + ic] * weights[OFF_PW2W + c * 16u + ic];
        }
        sm_p2[i] = max(acc, 0.0);
    }
    workgroupBarrier();

    // ---- gap: average over 3x3 -> 32 features ----
    if (tid < 32u) {
        var acc = 0.0;
        for (var p = 0u; p < 9u; p++) {
            acc += sm_p2[p * 32u + tid];
        }
        sm_gap[tid] = acc / 9.0;
    }
    workgroupBarrier();

    let off_fcb = OFF_FCW + num_classes * 32u;

    if (mode == 0u) {
        // ---- dense + local top-2 per thread (same rules as linear_argmax) ----
        var s1 = -3.402823466e+38;
        var i1 = 0u;
        var s2 = -3.402823466e+38;
        var i2 = 0u;
        for (var c = tid; c < num_classes; c += 64u) {
            var acc = weights[off_fcb + c];
            for (var ic = 0u; ic < 32u; ic++) {
                acc += sm_gap[ic] * weights[OFF_FCW + c * 32u + ic];
            }
            if (acc > s1) {
                s2 = s1;
                i2 = i1;
                s1 = acc;
                i1 = c;
            } else if (acc > s2) {
                s2 = acc;
                i2 = c;
            }
        }
        sm_score[tid] = s1;
        sm_id[tid] = i1;
    }
    workgroupBarrier();

    if (mode == 0u) {
        if (tid == 0u) {
            var r1 = sm_score[0u];
            var r1_id = sm_id[0u];
            var r2 = -3.402823466e+38;
            var r2_id = 0u;
            for (var i = 1u; i < 64u; i++) {
                let s = sm_score[i];
                let id = sm_id[i];
                if (s > r1) {
                    r2 = r1;
                    r2_id = r1_id;
                    r1 = s;
                    r1_id = id;
                } else if (s > r2) {
                    r2 = s;
                    r2_id = id;
                }
            }
            results[g * 3u + 0u] = r1_id;
            results[g * 3u + 1u] = bitcast<u32>(r1);
            results[g * 3u + 2u] = bitcast<u32>(r2);
        }
    } else if (mode == 1u) {
        // ---- dense: full [C] logits, one value per assigned class ----
        for (var c = tid; c < num_classes; c += 64u) {
            var acc = weights[off_fcb + c];
            for (var ic = 0u; ic < 32u; ic++) {
                acc += sm_gap[ic] * weights[OFF_FCW + c * 32u + ic];
            }
            results[g * num_classes + c] = bitcast<u32>(acc);
        }
    } else if (mode == 2u || mode == 4u) {
        // ---- dense + masked top-7 per thread (ties keep the lower id) ----
        var tv: array<f32, 7>;
        var ti: array<u32, 7>;
        var cnt = 0u;
        for (var i = 0u; i < 7u; i++) {
            tv[i] = -3.402823466e+38;
            ti[i] = 0xffffffffu;
        }
        for (var c = tid; c < num_classes; c += 64u) {
            if (((aux[c / 32u] >> (c % 32u)) & 1u) == 0u) {
                continue;
            }
            var acc = weights[off_fcb + c];
            for (var ic = 0u; ic < 32u; ic++) {
                acc += sm_gap[ic] * weights[OFF_FCW + c * 32u + ic];
            }
            if (cnt < 7u) {
                var j = cnt;
                while (j > 0u && tv[j - 1u] < acc) {
                    tv[j] = tv[j - 1u];
                    ti[j] = ti[j - 1u];
                    j -= 1u;
                }
                tv[j] = acc;
                ti[j] = c;
                cnt += 1u;
            } else if (acc > tv[6u]) {
                var j = 6u;
                while (j > 0u && tv[j - 1u] < acc) {
                    tv[j] = tv[j - 1u];
                    ti[j] = ti[j - 1u];
                    j -= 1u;
                }
                tv[j] = acc;
                ti[j] = c;
            }
        }
        for (var i = 0u; i < 7u; i++) {
            sm_topv[tid * 7u + i] = tv[i];
            sm_topi[tid * 7u + i] = ti[i];
        }
    }
    workgroupBarrier();

    if (mode == 2u || mode == 4u) {
        // ---- merge 64 x 7 local tops -> global top-7 record (14 u32) ----
        if (tid == 0u) {
            var rv: array<f32, 7>;
            var ri: array<u32, 7>;
            var rcnt = 0u;
            for (var i = 0u; i < 7u; i++) {
                rv[i] = -3.402823466e+38;
                ri[i] = 0xffffffffu;
            }
            for (var i = 0u; i < 448u; i++) {
                let v = sm_topv[i];
                let id = sm_topi[i];
                if (id == 0xffffffffu) {
                    continue;
                }
                if (rcnt < 7u) {
                    var j = rcnt;
                    while (j > 0u && rv[j - 1u] < v) {
                        rv[j] = rv[j - 1u];
                        ri[j] = ri[j - 1u];
                        j -= 1u;
                    }
                    rv[j] = v;
                    ri[j] = id;
                    rcnt += 1u;
                } else if (v > rv[6u]) {
                    var j = 6u;
                    while (j > 0u && rv[j - 1u] < v) {
                        rv[j] = rv[j - 1u];
                        ri[j] = ri[j - 1u];
                        j -= 1u;
                    }
                    rv[j] = v;
                    ri[j] = id;
                }
            }
            // Mode 2 records are 14 u32/glyph; mode 4 packs top-7 + the
            // gathered pairs into a 24-u32/glyph record, so the top-7 base
            // must use the same stride the host parses.
            let rec_stride = select(14u, 24u, mode == 4u);
            let base = g * rec_stride;
            for (var i = 0u; i < 7u; i++) {
                results[base + i] = ri[i];
                results[base + 7u + i] = bitcast<u32>(rv[i]);
            }
        }
        // ---- mode 4: gather the template top-K logits (5 slots) ----
        if (mode == 4u) {
            if (tid < 5u) {
                let tpl_id = template_results[g * 15u + tid];
                let out_base = g * 24u + 14u;
                if (tpl_id == 0xffffffffu) {
                    results[out_base + tid] = 0xffffffffu;
                    results[out_base + 5u + tid] = bitcast<u32>(-3.402823466e+38);
                } else {
                    var acc = weights[off_fcb + tpl_id];
                    for (var ic = 0u; ic < 32u; ic++) {
                        acc += sm_gap[ic] * weights[OFF_FCW + tpl_id * 32u + ic];
                    }
                    results[out_base + tid] = tpl_id;
                    results[out_base + 5u + tid] = bitcast<u32>(acc);
                }
            }
        }
    } else if (mode == 3u) {
        // ---- gather: logits for M requested ids, one value per request ----
        let M = params.a.w;
        for (var m = tid; m < M; m += 64u) {
            let c = aux[g * M + m];
            if (c == 0xffffffffu) {
                results[g * M + m] = bitcast<u32>(-3.402823466e+38);
            } else {
                var acc = weights[off_fcb + c];
                for (var ic = 0u; ic < 32u; ic++) {
                    acc += sm_gap[ic] * weights[OFF_FCW + c * 32u + ic];
                }
                results[g * M + m] = bitcast<u32>(acc);
            }
        }
    }
}
