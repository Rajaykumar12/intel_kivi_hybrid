"""Collect the shared libraries a relocatable kivi-run needs (Phase 5 A.9).

    python3 scripts/bundle_runtime_libs.py --lib-dir dist/.../lib \
        --root build-native/kivi-run --root build-native/libkivi_core.so \
        --dlopen-dir <torch's SYCL runtime dir> --env <conda env>

* Closure: `ldd` on the build-tree binaries (their RPATHs still point at the
  real libraries), plus the Unified Runtime adapters and tracing libraries
  that libsycl / libur_loader dlopen() by name (invisible to ldd), plus their
  own ldd closures.
* Bundled: everything outside the system library directories, plus ICU and
  PCRE2 (their sonames are version-specific, e.g. libicuuc.so.77, so a
  different distro would not have them). Left to the system: glibc,
  libstdc++/libgcc from the OS, TBB, and the GPU driver stack (Level Zero
  loader, Intel GPU / IGC drivers), which is loaded by dlopen from the
  system and must match the installed kernel driver.
* Each library is copied (symlinks resolved) under the name the loader asks
  for (its soname / NEEDED name).
* Provenance: every bundled file is traced to the pip wheel (dist-info
  RECORD), conda package (conda-meta) or system package (rpm -qf) it came
  from; license files are copied to <out>/licenses/ and a
  THIRD_PARTY_NOTICES.txt is written.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

SYSTEM_DIRS = ("/lib64/", "/usr/lib64/", "/lib/", "/usr/lib/")
# System libraries bundled anyway: version-specific sonames.
BUNDLE_FROM_SYSTEM = re.compile(r"^lib(icuuc|icudata|icui18n|pcre2-8)\.so")
# dlopen()ed by libsycl / libur_loader / libpti_view by name; not in any NEEDED.
DLOPEN = ["libur_adapter_level_zero.so.0", "libur_adapter_level_zero_v2.so.0",
          "libur_adapter_opencl.so.0", "libxptifw.so", "libpti.so"]


def ldd(path, missing=None):
    """NEEDED closure of `path` as {name: resolved path}. Unresolvable names
    are an error unless `missing` (a list) is given to collect them."""
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    out = subprocess.run(["ldd", path], capture_output=True, text=True, env=env, check=True).stdout
    deps = {}
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r"(\S+) => (not found|\S+)", line)
        if m:
            if m.group(2) == "not found":
                if missing is None:
                    raise SystemExit(f"bundle: {path}: {m.group(1)} not found")
                missing.append(m.group(1))
                continue
            deps[m.group(1)] = m.group(2)
    return deps


def wanted(name, path):
    if BUNDLE_FROM_SYSTEM.match(name):
        return True
    return not path.startswith(SYSTEM_DIRS)


def provenance_index(env):
    """realpath -> (kind, package, version, license, [license files])."""
    idx = {}
    site = glob.glob(os.path.join(env, "lib", "python3*", "site-packages"))
    for sp in site:
        for di in glob.glob(os.path.join(sp, "*.dist-info")):
            meta = os.path.join(di, "METADATA")
            name = ver = lic = "?"
            if os.path.exists(meta):
                with open(meta, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if line.startswith("Name: "): name = line[6:].strip()
                        elif line.startswith("Version: "): ver = line[9:].strip()
                        elif line.startswith("License: ") and lic == "?": lic = line[9:].strip()
                        elif line.startswith("License-Expression: "): lic = line[20:].strip()
                        elif not line.strip(): break
            lfiles = [f for f in glob.glob(os.path.join(di, "licenses", "**", "*"), recursive=True)
                      if os.path.isfile(f)] + glob.glob(os.path.join(di, "LICENSE*"))
            rec = os.path.join(di, "RECORD")
            if not os.path.exists(rec):
                continue
            with open(rec, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    rel = line.split(",")[0]
                    if ".so" in rel:
                        idx[os.path.realpath(os.path.normpath(os.path.join(sp, rel)))] = (
                            "pip", name, ver, lic, lfiles)
    for cm in glob.glob(os.path.join(env, "conda-meta", "*.json")):
        with open(cm) as fh:
            j = json.load(fh)
        for rel in j.get("files", []):
            if ".so" in rel:
                idx.setdefault(os.path.realpath(os.path.join(env, rel)),
                               ("conda", j.get("name", "?"), j.get("version", "?"),
                                j.get("license", "?"), []))
    return idx


def system_owner(path):
    try:
        out = subprocess.run(["rpm", "-qf", "--qf", "%{NAME}\t%{VERSION}\t%{LICENSE}", path],
                             capture_output=True, text=True, check=True).stdout
        n, v, l = out.split("\t")
        return ("system", n, v, l, sorted(glob.glob(f"/usr/share/licenses/{n}/*")))
    except (OSError, subprocess.CalledProcessError, ValueError):
        return ("system", "?", "?", "?", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib-dir", required=True)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--dlopen-dir", action="append", default=[])
    ap.add_argument("--env", required=True)
    ap.add_argument("--notices", required=True, help="THIRD_PARTY_NOTICES.txt to write")
    ap.add_argument("--licenses-dir", required=True)
    a = ap.parse_args()

    deps = {}
    for r in a.root:
        deps.update(ldd(r))
    for name in DLOPEN:
        hit = next((os.path.join(d, name) for d in a.dlopen_dir if os.path.exists(os.path.join(d, name))), None)
        if hit is None:
            raise SystemExit(f"bundle: {name} not found in {a.dlopen_dir}")
        deps[name] = hit
        # A dlopen()ed library without an RPATH may not resolve on its own
        # (libxptifw.so -> libsvml.so); in the process its NEEDED libraries
        # are already loaded by name, so they only have to be in the bundle.
        missing = []
        deps.update(ldd(hit, missing))
        unresolved = [m for m in missing if m not in deps]
        if unresolved:
            raise SystemExit(f"bundle: {hit}: {', '.join(unresolved)} not found")
    own = {os.path.basename(r) for r in a.root}
    bundle = {n: p for n, p in deps.items() if wanted(n, p) and n not in own}

    os.makedirs(a.lib_dir, exist_ok=True)
    idx = provenance_index(a.env)
    rows, total = [], 0
    os.makedirs(a.licenses_dir, exist_ok=True)
    for name, path in sorted(bundle.items()):
        dst = os.path.join(a.lib_dir, name)
        shutil.copy2(os.path.realpath(path), dst)
        os.chmod(dst, 0o755)
        size = os.path.getsize(dst)
        total += size
        real = os.path.realpath(path)
        prov = idx.get(real) or (system_owner(real) if real.startswith(SYSTEM_DIRS) else ("?", "?", "?", "?", []))
        kind, pkg, ver, lic, lfiles = prov
        if lfiles and pkg != "?":
            d = os.path.join(a.licenses_dir, pkg)
            if not os.path.isdir(d):
                os.makedirs(d)
                for f in lfiles:
                    shutil.copy2(f, os.path.join(d, os.path.basename(f)))
        rows.append((name, size, kind, pkg, ver, lic))
    with open(a.notices, "w") as fh:
        fh.write("Third-party libraries bundled in lib/ (copied unmodified).\n"
                 "License texts, where the source package ships them, are in licenses/<package>/.\n\n")
        fh.write(f"{'library':40} {'MiB':>7}  {'from':6} {'package':32} {'version':22} license\n")
        for name, size, kind, pkg, ver, lic in rows:
            fh.write(f"{name:40} {size / 2**20:7.1f}  {kind:6} {pkg:32} {ver:22} {lic}\n")
        fh.write("\nVendored source: nlohmann/json 3.11.3 (MIT), licenses/nlohmann_json/.\n")
    unknown = [r[0] for r in rows if r[3] == "?"]
    print(f"bundled {len(rows)} libraries, {total / 2**20:.0f} MiB"
          + (f"; no provenance for: {', '.join(unknown)}" if unknown else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
