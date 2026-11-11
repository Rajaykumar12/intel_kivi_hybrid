#!/bin/bash
# ==========================================================================
# Package the Python-free runtime (Phase 5 A.9):
#   dist/kivi-runtime-<ver>-linux-x86_64/{bin,lib,licenses,README.txt,...}
#   dist/kivi-runtime-<ver>-linux-x86_64.tar.zst
#
#   bash scripts/package_runtime.sh                        # build + package
#   bash scripts/package_runtime.sh --check artifacts/qwen25 [golden.json]
#       ... then extract the tarball to a scratch dir and run it in a clean
#       environment (env -i: no conda, no LD_LIBRARY_PATH, no oneAPI) and
#       verify no mapped library comes from outside the package except the
#       OS and GPU driver.
#
# The package carries its own LibTorch / SYCL runtime / MKL libraries (copied
# from the torch XPU wheel), so the target needs only: x86-64 Linux with the
# glibc / libstdc++ versions listed in README.txt, and the Intel GPU driver
# stack (Level Zero loader + compute runtime) for --device xpu.
# ==========================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ENV_DIR="${KIVI_ENV:-$HOME/miniforge3/envs/intel_xpu}"
BUILD_DIR="${KIVI_NATIVE_BUILD_DIR:-build-native}"

CHECK_MODEL="" CHECK_GOLDEN=""
if [ "${1:-}" = "--check" ]; then
    CHECK_MODEL="${2:?--check needs an artifact directory}"
    CHECK_GOLDEN="${3:-}"
fi

VER="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' src/kivi_sycl/_version.py)"
REV="$(git rev-parse --short HEAD)"
git diff --quiet HEAD -- src CMakeLists.txt third_party || REV="${REV}-dirty"
NAME="kivi-runtime-${VER}+${REV}-linux-x86_64"
OUT="dist/${NAME}"

echo "== build"
# preinstall = all targets + relinking those whose install RPATH cannot be
# patched in place (kivi-run: DT_RPATH, see CMakeLists.txt) into
# CMakeFiles/CMakeRelink.dir, which cmake --install copies.
KIVI_SKIP_TESTS=1 KIVI_BUILD_TARGET=preinstall bash scripts/build_native.sh >/dev/null

echo "== install -> ${OUT}"
rm -rf "$OUT"
cmake --install "$BUILD_DIR" --prefix "$OUT" >/dev/null

echo "== bundle libraries"
TORCH_LIB="$(dirname "$(find "$ENV_DIR/lib" -path '*site-packages/torch/lib/libtorch_cpu.so' | head -1)")"
SYCL_DIR="$(grep -m1 '^KIVI_SYCL_RUNTIME_DIR:' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2-)"
mkdir -p "$OUT/licenses/nlohmann_json"
cp third_party/nlohmann/LICENSE.MIT "$OUT/licenses/nlohmann_json/"
python3 scripts/bundle_runtime_libs.py --lib-dir "$OUT/lib" \
    --root "$BUILD_DIR/kivi-run" --root "$BUILD_DIR/libkivi_core.so" \
    --dlopen-dir "$SYCL_DIR" --dlopen-dir "$TORCH_LIB" --env "$ENV_DIR" \
    --notices "$OUT/THIRD_PARTY_NOTICES.txt" --licenses-dir "$OUT/licenses"

echo "== check: every dependency resolves inside the package or the OS"
LDD_OUT="$(env -i PATH=/usr/bin:/bin ldd "$OUT/bin/kivi-run")"
if echo "$LDD_OUT" | grep -q "not found"; then
    echo "$LDD_OUT" | grep "not found"; exit 1
fi
# Paths may contain spaces (whole path between "=> " and " (0x").
LDD_PATHS="$(echo "$LDD_OUT" | sed -n -E 's/.*=> (.*) \(0x[0-9a-f]+\)$/\1/p')"
OUTSIDE="$(echo "$LDD_PATHS" | grep -v -F "$(realpath "$OUT")/" | grep -v -E "^/(usr/)?lib(64)?/" || true)"
if [ -n "$OUTSIDE" ]; then
    echo "resolves outside the package and the OS:"; echo "$OUTSIDE"; exit 1
fi
echo "ldd: clean ($(echo "$LDD_PATHS" | grep -c -F "$(realpath "$OUT")/") from the package," \
     "$(echo "$LDD_PATHS" | grep -c -E '^/(usr/)?lib')" "from the OS)"

TORCH_VER="$(basename "$(ls -d "$(dirname "$(dirname "$TORCH_LIB")")"/torch-*.dist-info | head -1)" .dist-info)"
TORCH_VER="${TORCH_VER#torch-}"
# Oldest OS the binaries accept: highest glibc / libstdc++ symbol versions.
need() { objdump -T "$OUT"/bin/* "$OUT"/lib/*.so* 2>/dev/null | grep -o "$1_[0-9.]*" | sort -uV | tail -1; }
GLIBC_NEED="$(need GLIBC)"
GLIBCXX_NEED="$(need GLIBCXX)"

cat > "$OUT/README.txt" <<EOF
kivi-run ${VER}+${REV} — Python-free KIVI (2-bit KV cache) inference runtime

Usage
  bin/kivi-run --model <artifact dir> --chat
  bin/kivi-run --model <artifact dir> --prompt "Once upon a time" --max-new-tokens 64
  bin/kivi-run --help

Artifact directories are produced once, offline, by the Python exporter:
  python -m kivi_sycl.export --model Qwen/Qwen2.5-0.5B-Instruct --out artifacts/qwen25
An artifact runs only with the runtime built from the same torch version
(kivi_config.json "torch_version" must be ${TORCH_VER}, the LibTorch in lib/)

Requirements on the target machine
  x86-64 Linux, glibc >= ${GLIBC_NEED#GLIBC_}, libstdc++ with ${GLIBCXX_NEED}
  --device xpu (default): Intel GPU driver stack — Level Zero loader
    (libze_loader.so.1) and the Intel compute runtime (libze_intel_gpu.so.1)
  --device cpu: nothing more

Everything else (LibTorch, SYCL runtime, oneMKL, ICU, PCRE2, ...) is in lib/;
bin/kivi-run finds it through its RPATH (\$ORIGIN/../lib), so the directory
can be moved anywhere. No Python, conda or oneAPI installation is used.

TMPDIR: the compiled model is extracted to \$TMPDIR (default /tmp) at start.

Third-party components: THIRD_PARTY_NOTICES.txt, licenses/.
EOF

echo "== tarball"
tar -C dist -I 'zstd -T0 -6' -cf "dist/${NAME}.tar.zst" "$NAME"
echo "$(du -sh "$OUT" | cut -f1) unpacked, $(du -sh "dist/${NAME}.tar.zst" | cut -f1) as dist/${NAME}.tar.zst"

[ -z "$CHECK_MODEL" ] && exit 0

echo "== clean-environment run from the extracted tarball"
SCRATCH="${KIVI_TMPDIR:-$HOME/.cache/kivi_tmp}/pkgcheck"
rm -rf "$SCRATCH" && mkdir -p "$SCRATCH"
tar -C "$SCRATCH" -I zstd -xf "dist/${NAME}.tar.zst"
BIN="$SCRATCH/$NAME/bin/kivi-run"
MODEL="$(realpath "$CHECK_MODEL")"
ARGS=(--model "$MODEL" --prompt "The capital of France is" --max-new-tokens 16)
[ -n "$CHECK_GOLDEN" ] && ARGS=(--model "$MODEL" --golden "$(realpath "$CHECK_GOLDEN")")
env -i HOME="$HOME" PATH=/usr/bin:/bin TMPDIR="$SCRATCH" "$BIN" "${ARGS[@]}" \
    > "$SCRATCH/stdout.txt" 2> "$SCRATCH/stderr.txt" &
PID=$!
# The GPU stack loads lazily (at the KV engine's first 2-bit flush), so take
# the union of the mappings over the whole run. pathname = everything after
# the 5th field (it may contain spaces).
: > "$SCRATCH/maps.txt"
while kill -0 "$PID" 2>/dev/null; do
    sed -E 's/^([^ ]+ +){5}//' "/proc/$PID/maps" 2>/dev/null | grep '\.so' >> "$SCRATCH/maps.txt" || true
    sleep 1
done
wait $PID && RC=0 || RC=$?
sort -u "$SCRATCH/maps.txt" -o "$SCRATCH/maps.txt"
cat "$SCRATCH/stdout.txt"; grep -E "kivi-run: (loaded|prefill|golden|error)" "$SCRATCH/stderr.txt" || true
# Expected outside the package: the OS (incl. the GPU driver) and the
# compiled model the package loader extracts to $TMPDIR.
FOREIGN="$(grep -v -F "$SCRATCH/" "$SCRATCH/maps.txt" | grep -v -E "^/(usr/)?lib(64)?/|^/etc/ld\.so\.cache" || true)"
echo "mapped: $(wc -l < "$SCRATCH/maps.txt") libraries, $(grep -c -F "$SCRATCH/$NAME/lib/" "$SCRATCH/maps.txt") from the package," \
     "$(grep -c -E "libze_loader|libze_intel_gpu" "$SCRATCH/maps.txt") Level Zero driver libraries from the OS"
if [ -n "$FOREIGN" ]; then
    echo "FAIL: mapped from outside the package and the OS:"; echo "$FOREIGN"; exit 1
fi
if [ -n "$CHECK_GOLDEN" ] && ! grep -q "libze_loader" "$SCRATCH/maps.txt"; then
    echo "FAIL: the GPU (Level Zero) was never used"; exit 1
fi
[ "$RC" -eq 0 ] || { echo "FAIL: kivi-run exited $RC"; exit 1; }
rm -rf "$SCRATCH"
echo "== package check passed"
