# Getting started: build, verify, and submit your smoke-test entry

Au26 GPUCompute (UW ECE EEP 524 A/B) · ResNet-18 inference engine final project

This guide takes you from the two downloads (this kit and the data pack) to a verified engine and your team's first
entry on the leaderboard. Do the steps in order; each has a pass signal. Budget about 30 minutes, most of it waiting.
The smoke-test entry is the unmodified starter: it tests your toolchain and the submission path, not your kernels.

Smoke-test deadline: **22 Nov, 11:59 PM**

## Key dates

| Event | Due |
|---|---|
| Project team formation / registration | 16 Nov, 11:59 PM* |
| Project starter kit released | 16 Nov, 9:00 AM |
| Project proposal (2 pages) due | 20 Nov, 11:59 PM |
| Tier T0 smoke-test leaderboard submission due | 22 Nov, 11:59 PM |
| Team presentation date/time sign-up | 24 Nov, 5:59 PM* |
| Tiers T1 - Tn leaderboard submission window | 22 Nov - 15 Dec |
| Project design specification due | 2 Dec, 11:59 PM |
| Leaderboard freeze; final project code submission due | 15 Dec, 5:00 PM* |
| Project team presentations, 10 min + 5 min questions | 15-17 Dec, 6-9 PM* |
| Final report (with code analysis) due | 17 Dec, 11:59 PM* |

* not eligible for Late Days usage

## Conventions

- **Shell (Windows):** use the *x64 Native Tools Command Prompt for Visual Studio 2026* (or the Developer PowerShell
  for VS 2026), from the Start menu, so `cl.exe`, `nvcc` and `cmake` are on PATH. Commands below work in cmd and PowerShell. On Linux use `/` in paths and `./build/engine`
  instead of `build\Release\engine.exe`.
- **KIT** means the folder this file is in. Commands run from KIT unless a step says otherwise.
- **PACK** means the folder that contains `manifest.json`, the `*.bin` files, `ref\` and `checksums.sha256`. On the
  course AWS T4 instances it is `C:\resnet18\resnet18_pack` (pre-installed); on your own machine, wherever you unzipped it, for
  example `C:\data\resnet18_pack`. Use an absolute path. It is the folder holding `manifest.json`, not its parent: if unzipping
  nested it (`...\resnet18_pack\resnet18_pack\manifest.json`), pass the inner folder.
- Scores are measured on the course T4 (compute capability 7.5). You may develop on any GPU with CC 7.5 or newer.

## Working on the course AWS T4 instances

The course instance image comes with everything this project needs already installed: Visual Studio 2026 (C++),
the CUDA toolkit, CMake, Git for Windows, and the data pack at `C:\resnet18\resnet18_pack`. So on an instance:

- **Do not download the pack.** Use `C:\resnet18\resnet18_pack` as PACK and just run the verification in step 1.
  Treat that folder as read-only: write your outputs (`out\`, builds, zips) inside your own working folder.
- **Keep your code in Git.** Do not count on anything you leave on an instance surviving to your next session. Keep
  the team's engine in a private Git repository (UW GitLab or a private GitHub repository) shared only with your
  teammates, `git clone` it at the start of a session, and `git commit` + `git push` before you log out. The starter
  includes a `.gitignore` that keeps build folders, outputs and submission zips out of the repository; rebuild after
  each clone (step 2), it takes a minute or two.

A minimal session:

```
git clone <your team's private repository URL> resnet18
cd resnet18
cd starter && cmake -B build && cmake --build build --config Release -j && cd ..
python tools\check.py --pack C:\resnet18\resnet18_pack --out out\debug --set debug --layers --engine starter\build\Release\engine.exe
rem ... work ...
git add -A && git commit -m "describe the change" && git push
```

(Put the whole student kit in the repository, `tools\` included, so the commands above work from the repository root.)

## Step 0: Prerequisites

```
nvcc --version          # CUDA 13.x
cmake --version         # 4.2 or newer to use Visual Studio 2026 (older CMake: use -G Ninja, see step 2)
python --version        # 3.9 or newer
nvidia-smi              # your GPU is listed
pip install numpy pillow
```

On the course T4 instances all of this is pre-installed. On your own Windows machine you need Visual Studio 2026
(or its Build Tools) with the C++ workload and CUDA 13.2 or newer, the first CUDA release that supports Visual Studio
2026 (Visual Studio 2022 with any CUDA 13.x also works); on Linux, gcc/g++. You do not need PyTorch, cuDNN or cuBLAS for
anything in this project.

## Step 1: Get and verify the data pack

**On a course T4 instance**, the pack is already at `C:\resnet18\resnet18_pack`: skip the download and run the two checks below
with PACK = `C:\resnet18\resnet18_pack`.

**On your own machine**, download it from Course Canvas: Files/Final Project - ResNet18/resnet18_pack.zip and unzip it.

Then verify the copy. Both checks must pass before you trust any result.

```
python tools\check.py --pack PACK --verify-pack --no-e2e
```

Expected: `pack verification: OK (...)`. Then compare the hash of the checksum file itself with the published value:

```
certutil -hashfile PACK\checksums.sha256 SHA256        (Windows)
sha256sum PACK/checksums.sha256                         (Linux)
```

Published SHA-256 of `checksums.sha256`: `359370C13F34961BDF159DEC3221347A0BA605E1140B017E058FC9FCF7A38E94`
Published SHA-256 of the pack zip: `177780D29AF38D8CEBFD2E836DDF60A297C5CD05EAD7D9635B5429CF3283E3D2`

If either check fails, download the pack again. Do not debug against a pack that fails verification.

## Step 2: Build the starter

```
cd starter
cmake -B build
cmake --build build --config Release -j
cd ..
```

The default build targets the T4 (sm_75). To also get native code for your own GPU, add it at configure time and keep
75 in the list: `cmake -B build -DCMAKE_CUDA_ARCHITECTURES="75;89"` (RTX 40-series, L4), `"75;86"` (RTX 30-series),
`"75;80"` (A100), `"75;90"` (H100/H200).

The engine is `starter\build\Release\engine.exe` on Windows and `starter/build/engine` on Linux. Warnings are fine.

If CMake cannot find Visual Studio 2026 at all, your CMake is older than 4.2 (the first release that knows VS 2026);
update it, or use the Ninja form below. If CMake reports that no CUDA toolset is found (Windows), the CUDA installer's
Visual Studio Integration is missing;
install it, or configure with Ninja from the same prompt:
`cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` (the engine is then `build\engine.exe`).

## Step 3: Check every layer (seconds)

```
python tools\check.py --pack PACK --out out\debug --set debug --layers --engine starter\build\Release\engine.exe
```

This runs the engine on the 8 debug images with `--dump` and compares every tensor in the network with the PyTorch
reference in `PACK\ref\`. Pass signal: all 25 tensors `PASS` with relative error around 1e-6, then `OVERALL: PASS`.

This is the command you will run most this quarter. When you replace a kernel, the first tensor that fails tells you
which layer (and therefore which kernel and shape class) is wrong; everything downstream of it fails too.

## Step 4: End-to-end on the dev set (about a minute)

```
python tools\check.py --pack PACK --out out\dev --set dev --engine starter\build\Release\engine.exe --batch 32 --iters 3 --warmup 1
```

500 images, one warm-up pass and three timed passes. Pass signal: top-1 agreement 100%, top-1 and top-5 error equal to
the reference's, `top5.bin consistent with own logits` 100%, `OVERALL: PASS`. The `timing.json` line reports images/s
for your GPU: that is your Tier 0 baseline. Write it down; every optimization is measured against it.

The starter is slow on purpose (one thread per output element, no data reuse). For scale: on a laptop RTX 4090 it
manages about 92 images/s, i.e. roughly 43 s per pass over the full 3,925-image val set. Measured on the course T4:
18.5 images/s, median 212485.23 ms. Use the debug set for correctness and the dev set for timing. The dev set is also the scored set: the
grader times your engine on these same 500 images, with this same 1 warm-up + 3 timed passes (median), so this
command reproduces your leaderboard measurement on your own GPU. Note the two "reference" error figures it prints;
those are what the board will show for a correct engine. The full val set (3,925 images) is there for broader testing.

## Step 5: Package the smoke-test submission

```
python tools\submit.py --team "your-team-name" --members "First Last, First Last, First Last" --category fp32 --batch 32 --tier 0 --src starter --outdir submissions
```

This zips `CMakeLists.txt`, `src\`, `include\` and `third_party\` (no build folders, binaries or data) together with a
`submission.json`, and prints the path of the zip. The file name carries your team name, the tier you declared and a
UTC time stamp, for example `submissions\your-team-name_T0_20261121T190512Z.zip`; do not rename it. The grader builds your engine from that zip alone, for sm_75, and
runs it with exactly these arguments: `--pack`, `--set`, `--out`, `--batch` (the value you gave), `--precision` (your
category), `--iters`, `--warmup`, plus anything you pass in `--engine-args "..."`.

Choose your team name once and keep it: the leaderboard tracks each team's best valid run under that name. The
leaderboard is a public web page, so choose a team name that does not contain your real names.

## Step 6: Hand it in and find yourself on the board

How to submit the zip: Use submit.py to create submission ZIP, upload to course Canvas assignment

Leaderboard: overall: https://faculty.washington.edu/colinrei/au26-leaderboard/board.html ; by tier: https://faculty.washington.edu/colinrei/au26-leaderboard/board_by_tier.html

Submissions are graded in rounds, one at a time on the course T4, so your entry appears after the next round, not
immediately. Every round uses the same protocol as step 4: the dev set, 1 warm-up + 3 timed passes, median. What you
should see: your team in the FP32 category at tier 0 with a small images/s figure, and top-1 and top-5 error equal to
the reference figures that step 4 printed. After each round your team receives a short report per submission: the
measured figures, or the stage that failed and where. If your submission is listed under "Invalid submissions", the
reason column says which stage failed.

The board has two views: overall (ranked by images/s within each category) and by tier (ranked within the tier you
declared with `--tier`). The staff verify declared tiers against your design specification and report.

## What makes a submission invalid

The grader runs, in order: a source scan, the build, the per-layer check on the debug set, the timed run on the dev
set, and the end-to-end gate. Any failure means the run is not posted.

- **Forbidden libraries.** The scan is a strict, case-insensitive text search of every source and CMake file for these
  identifiers: cublas, cudnn, cutlass, cub/, thrust, torch, onnxruntime, nvinfer, tensorrt. It does not parse code, so
  a match in a *comment or a string* counts exactly like an #include or a call. Do not mention these names anywhere in
  your sources, not even in a comment such as "faster than ...". The report lists the file and line of every match.
- **Does not build for sm_75.** Instructions the T4 lacks (cp.async, bf16, the m16n8k16 fp16 MMA shape, wgmma,
  clusters) fail on the grader even if they ran on your GPU. Keep 75 in your architecture list so you see it first.
- **Per-layer check fails.** Run step 3 before every submission.
- **Gate fails.** Top-1 agreement with the reference below 99.5%, top-1 or top-5 error more than 0.5 percentage points
  from the reference, non-finite logits, or a `top5.bin` that disagrees with your own `logits.bin`.

## After the smoke test

- Read `starter\README.md` (where each kernel lives and what to replace) and sections 6 to 9 of the spec.
- Print the graph the executor runs: `starter\build\Release\engine.exe --pack PACK --print-graph --set debug --out out\g --batch 8`.
- Look at what the engine sees: `python tools\pack_images.py --pack PACK --set debug --sheet debug_images.png`.
- Profile Tier 0 with Nsight Compute before changing anything; your proposal needs the per-layer time breakdown.
- Your own conv path: add a mode next to `direct` and `im2col` in `starter\src\executor.cu`, select it with `--conv`
  while developing, and tell the grader with `submit.py --engine-args "--conv yourmode"`.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `cannot open ...manifest.json` | PACK points at the wrong folder (often the parent, or a nested unzip). Pass the folder that contains `manifest.json`. |
| `No CUDA toolset found` at configure | CUDA's Visual Studio Integration is not installed; install it or use `-G Ninja`. |
| CMake cannot find Visual Studio 2026 | CMake older than 4.2 does not know Visual Studio 2026. Update CMake, or configure with `-G Ninja` from the VS 2026 developer prompt. |
| `nvcc: unsupported Microsoft Visual Studio version` | CUDA older than 13.2 with Visual Studio 2026 (your own machine). Update CUDA to 13.2 or newer, or build with Visual Studio 2022. |
| Code missing after logging in to a new instance | Instance storage is not your workspace: clone your team repository again. Commit and push before logging out. |
| Build errors mentioning `cl.exe` | Not in a VS developer prompt. Open the x64 Native Tools Command Prompt and rerun. |
| Engine not found by `check.py` | Windows builds put it in `build\Release\engine.exe`; Linux and Ninja in `build/engine`. |
| `--dump needs the whole set in one batch` | Use `--set debug` (8 images) for `--layers`, or a batch at least as large as the set. |
| `fp16 is not compiled in` | Configure with `-DENABLE_FP16=ON` (Tier 4; the reduced-precision category). |
| Step 3 fails on an unmodified starter | The pack is damaged (redo step 1) or the build is stale (delete `starter\build` and rebuild). |
| Errors near 1e-3 against your own PyTorch run | TF32 in PyTorch's cuDNN convolutions, not your kernel; see the README note. |

Questions: class Teams channel: Final Project - ResNet18
