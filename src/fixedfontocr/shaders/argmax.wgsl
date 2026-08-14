// Per-glyph argmax over [N, out_c] logits.
//
// Each glyph is handled by one 64-thread workgroup. Threads scan disjoint
// classes, keep local top-1/top-2, and thread 0 performs the final merge.
// Output is one 12-byte record per glyph:
//   best_id (u32), best_score (f32), second_score (f32).

struct Params {
    a: vec4<u32>,  // n, out_c, unused, unused
};

struct Result {
    best_id: u32,
    best_score: f32,
    second_score: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

var<workgroup> shared_score: array<f32, 64>;
var<workgroup> shared_id: array<u32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = wgid.x;
    let tid = lid.x;
    let out_c = params.a.y;

    var s1 = -3.402823466e+38;
    var i1 = 0u;
    var s2 = -3.402823466e+38;
    var i2 = 0u;
    for (var c = tid; c < out_c; c += 64u) {
        let s = logits[n * out_c + c];
        if (s > s1) {
            s2 = s1;
            i2 = i1;
            s1 = s;
            i1 = c;
        } else if (s > s2) {
            s2 = s;
            i2 = c;
        }
    }

    shared_score[tid] = s1;
    shared_id[tid] = i1;
    workgroupBarrier();

    if (tid == 0u) {
        var r1 = shared_score[0u];
        var r1_id = shared_id[0u];
        var r2 = -3.402823466e+38;
        var r2_id = 0u;
        for (var i = 1u; i < 64u; i++) {
            let s = shared_score[i];
            let id = shared_id[i];
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
        results[n] = Result(r1_id, r1, r2);
    }
}
