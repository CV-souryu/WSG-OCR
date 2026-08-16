// template_match.wgsl — GPU Template V2 matching in ONE dispatch.
//
// One 64-thread workgroup per glyph replicates TemplateV2Classifier.match_batch
// semantics exactly:
//
//   1. glyph coarse features (ink, h/w/top/left/bottom/right, ink tolerance)
//   2. per-prototype coarse filter (same tolerances as the CPU cascade)
//   3. XOR + popcount exact scan over filter-passing prototypes
//   4. per-character minimum distance (chars own consecutive prototypes)
//   5. k-th character distance -> ink-band fallback scan of skipped prototypes
//      (|proto_ink - glyph_ink| <= kth), keeping the returned Top-K exact
//   6. Top-K characters ordered by (dist, char_id) ascending — identical to
//      the CPU's lexsort((char_id, dist)) over argpartition
//   7. winning prototype inside the best character: first minimal distance
//      (same tie rule as np.argmin), plus its render-size/phase/mode metadata
//
// All distances are exact integers (XOR + popcount), so the output matches
// the numpy reference exactly, not within a tolerance.
//
// Record per glyph (15 x u32):
//   [0..4]   top-K char ids (-1 = 0xFFFFFFFF padding)
//   [5..9]   top-K Hamming distances (padding = area+1)
//   [10]     winning prototype index (global)
//   [11..14] render_size, dx, dy, downsample_mode
//
// Constants: K = 5, C_MAX = 8192, P_MAX = 256 (host enforces).

const K: u32 = 5u;
const C_MAX: u32 = 7000u;
const P_MAX: u32 = 256u;

struct Params {
    a: vec4<u32>,  // n, C, p, unused
    b: vec4<u32>,  // area, fill, allowed_words, unused
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> glyph_words: array<u32>;
@group(0) @binding(2) var<storage, read> tbits: array<u32>;
@group(0) @binding(3) var<storage, read> feats: array<u32>;
@group(0) @binding(4) var<storage, read> protos_meta: array<u32>;
@group(0) @binding(5) var<storage, read> allowed: array<u32>;
@group(0) @binding(6) var<storage, read_write> results: array<u32>;

var<workgroup> sm_min: array<u32, C_MAX>;
var<workgroup> sm_row_ink: array<u32, 24>;
var<workgroup> sm_row_any: array<u32, 24>;
var<workgroup> sm_col_any: array<u32, 24>;
var<workgroup> sm_feats: array<u32, 8>;  // ink, h, w, top, left, bottom, right, ink_tol
var<workgroup> sm_kth: array<u32, 1>;
var<workgroup> sm_top_key: array<u32, 320>;  // 64 threads x K
var<workgroup> sm_wdist: array<u32, P_MAX>;
var<workgroup> sm_best_char: array<u32, 1>;

fn allowed_bit(c: u32) -> u32 {
    return (allowed[c / 32u] >> (c % 32u)) & 1u;
}

fn proto_dist(g: u32, proto: u32) -> u32 {
    var d = 0u;
    for (var j = 0u; j < 18u; j++) {
        d += countOneBits(glyph_words[g * 18u + j] ^ tbits[proto * 18u + j]);
    }
    return d;
}

fn filter_ok(proto: u32) -> bool {
    let f0 = feats[proto * 2u];
    let f1 = feats[proto * 2u + 1u];
    let f_ink = f0 & 0xffffu;
    let f_h = (f0 >> 16u) & 0xffu;
    let f_w = f0 >> 24u;
    let f_top = f1 & 0xffu;
    let f_left = (f1 >> 8u) & 0xffu;
    let f_bottom = (f1 >> 16u) & 0xffu;
    let f_right = f1 >> 24u;

    let ink = sm_feats[0u];
    let h = sm_feats[1u];
    let w = sm_feats[2u];
    let top = sm_feats[3u];
    let left = sm_feats[4u];
    let bottom = sm_feats[5u];
    let right = sm_feats[6u];
    let ink_tol = sm_feats[7u];

    let d_ink = select(f_ink - ink, ink - f_ink, ink > f_ink);
    let d_h = select(f_h - h, h - f_h, h > f_h);
    let d_w = select(f_w - w, w - f_w, w > f_w);
    let d_top = select(f_top - top, top - f_top, top > f_top);
    let d_left = select(f_left - left, left - f_left, left > f_left);
    let d_bottom = select(f_bottom - bottom, bottom - f_bottom, bottom > f_bottom);
    let d_right = select(f_right - right, right - f_right, right > f_right);
    return d_ink <= ink_tol
        && d_h <= 1u
        && d_w <= 1u
        && d_top <= 2u
        && d_left <= 2u
        && d_bottom <= 2u
        && d_right <= 2u;
}

// Insert v into the ascending 5-slot list (keeps the K smallest).
fn insert5(kk_vals: ptr<function, array<u32, 5>>, cnt: ptr<function, u32>, v: u32) {
    if (*cnt < K) {
        var j = *cnt;
        while (j > 0u && (*kk_vals)[j - 1u] > v) {
            (*kk_vals)[j] = (*kk_vals)[j - 1u];
            j -= 1u;
        }
        (*kk_vals)[j] = v;
        *cnt += 1u;
    } else if (v < (*kk_vals)[K - 1u]) {
        var j = K - 1u;
        while (j > 0u && (*kk_vals)[j - 1u] > v) {
            (*kk_vals)[j] = (*kk_vals)[j - 1u];
            j -= 1u;
        }
        (*kk_vals)[j] = v;
    }
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let g = wgid.x;
    let tid = lid.x;
    let C = params.a.y;
    let p = params.a.z;
    let area = params.b.x;
    let fill = params.b.y;

    // ---- glyph coarse features ----
    if (tid < 24u) {
        // Row tid: 3 bytes of the packed bitset.
        var ink = 0u;
        var any = 0u;
        for (var j = 0u; j < 3u; j++) {
            let b = tid * 3u + j;
            let w = glyph_words[g * 18u + b / 4u];
            let byte = (w >> ((b % 4u) * 8u)) & 0xffu;
            ink += countOneBits(byte);
            if (byte != 0u) { any = 1u; }
        }
        sm_row_ink[tid] = ink;
        sm_row_any[tid] = any;
        // Column tid: any ink over the 24 rows.
        var ca = 0u;
        for (var r = 0u; r < 24u; r++) {
            let pos = r * 24u + tid;
            let w = glyph_words[g * 18u + pos / 32u];
            if (((w >> (pos % 32u)) & 1u) != 0u) {
                ca = 1u;
                break;
            }
        }
        sm_col_any[tid] = ca;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var ink = 0u;
        for (var r = 0u; r < 24u; r++) {
            ink += sm_row_ink[r];
        }
        var h = 0u;
        var w = 0u;
        var top = 0u;
        var left = 0u;
        var bottom = 0u;
        var right = 0u;
        if (ink > 0u) {
            var first = 24u;
            var last = 24u;
            for (var r = 0u; r < 24u; r++) {
                if (sm_row_any[r] != 0u) {
                    if (first == 24u) { first = r; }
                    last = r;
                }
            }
            top = first;
            h = last - first + 1u;
            bottom = 23u - last;
            var fc = 24u;
            var lc = 24u;
            for (var c = 0u; c < 24u; c++) {
                if (sm_col_any[c] != 0u) {
                    if (fc == 24u) { fc = c; }
                    lc = c;
                }
            }
            left = fc;
            w = lc - fc + 1u;
            right = 23u - lc;
        }
        sm_feats[0u] = ink;
        sm_feats[1u] = h;
        sm_feats[2u] = w;
        sm_feats[3u] = top;
        sm_feats[4u] = left;
        sm_feats[5u] = bottom;
        sm_feats[6u] = right;
        sm_feats[7u] = ink / 5u + 3u;
    }
    workgroupBarrier();

    // ---- pass A: coarse filter + exact scan, per-character minimum ----
    for (var c = tid; c < C; c += 64u) {
        var m = fill;
        if (allowed_bit(c) != 0u) {
            for (var t = 0u; t < p; t++) {
                let proto = c * p + t;
                if (filter_ok(proto)) {
                    let d = proto_dist(g, proto);
                    if (d < m) { m = d; }
                }
            }
        }
        sm_min[c] = m;
    }
    workgroupBarrier();

    // ---- k-th character distance over allowed chars (thread 0) ----
    if (tid == 0u) {
        var kk_vals: array<u32, 5>;
        var cnt = 0u;
        for (var i = 0u; i < K; i++) { kk_vals[i] = 0xffffffffu; }
        for (var c = 0u; c < C; c++) {
            if (allowed_bit(c) != 0u) {
                insert5(&kk_vals, &cnt, sm_min[c]);
            }
        }
        if (cnt == 0u) {
            sm_kth[0u] = fill;
        } else {
            sm_kth[0u] = kk_vals[min(K, cnt) - 1u];
        }
    }
    workgroupBarrier();

    // ---- pass B: ink-band fallback scan (|ink delta| <= kth, not filter-ok) ----
    let ink = sm_feats[0u];
    for (var c = tid; c < C; c += 64u) {
        if (allowed_bit(c) != 0u) {
            var m = sm_min[c];
            for (var t = 0u; t < p; t++) {
                let proto = c * p + t;
                let f_ink = feats[proto * 2u] & 0xffffu;
                let d_ink = select(f_ink - ink, ink - f_ink, ink > f_ink);
                if (d_ink <= sm_kth[0u] && !filter_ok(proto)) {
                    let d = proto_dist(g, proto);
                    if (d < m) { m = d; }
                }
            }
            sm_min[c] = m;
        }
    }
    workgroupBarrier();

    // ---- Top-K over allowed chars, ordered by (dist, char_id) ----
    var keys: array<u32, 5>;
    var kcnt = 0u;
    for (var i = 0u; i < K; i++) { keys[i] = 0xffffffffu; }
    for (var c = tid; c < C; c += 64u) {
        if (allowed_bit(c) != 0u) {
            let d = sm_min[c];
            insert5(&keys, &kcnt, d * C + c);
        }
    }
    for (var i = 0u; i < K; i++) {
        sm_top_key[tid * K + i] = keys[i];
    }
    workgroupBarrier();

    if (tid == 0u) {
        var out: array<u32, 5>;
        var ocnt = 0u;
        for (var i = 0u; i < K; i++) { out[i] = 0xffffffffu; }
        for (var i = 0u; i < 320u; i++) {
            insert5(&out, &ocnt, sm_top_key[i]);
        }
        let base = g * 15u;
        for (var i = 0u; i < K; i++) {
            let key = out[i];
            if (key == 0xffffffffu) {
                results[base + i] = 0xffffffffu;
                results[base + K + i] = fill;
            } else {
                results[base + i] = key % C;
                results[base + K + i] = key / C;
            }
        }
        if (out[0u] == 0xffffffffu) {
            sm_best_char[0u] = 0xffffffffu;
        } else {
            sm_best_char[0u] = out[0u] % C;
        }
    }
    workgroupBarrier();

    // ---- winning prototype inside the best character ----
    // CPU: argmin over every prototype the two passes actually scanned
    // (filter-passing OR ink-band fallback); the rest stay at fill.
    let bc = sm_best_char[0u];
    for (var t = tid; t < p; t += 64u) {
        let proto = bc * p + t;
        let f_ink = feats[proto * 2u] & 0xffffu;
        let d_ink = select(f_ink - ink, ink - f_ink, ink > f_ink);
        if (filter_ok(proto) || d_ink <= sm_kth[0u]) {
            sm_wdist[t] = proto_dist(g, proto);
        } else {
            sm_wdist[t] = fill;
        }
    }
    workgroupBarrier();

    if (tid == 0u) {
        var best = fill;
        var best_t = 0u;
        for (var t = 0u; t < p; t++) {
            let d = sm_wdist[t];
            if (d < best) {
                best = d;
                best_t = t;
            }
        }
        let proto = bc * p + best_t;
        let m = protos_meta[proto];
        let base = g * 15u;
        results[base + 10u] = proto;
        results[base + 11u] = m & 0xffu;
        results[base + 12u] = (m >> 8u) & 0xffu;
        results[base + 13u] = (m >> 16u) & 0xffu;
        results[base + 14u] = m >> 24u;
    }
}
