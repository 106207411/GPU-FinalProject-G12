# ResNet-18 inference engine - final project (student kit)

Au26 GPUCompute (UW ECE EEP 524 A/B) · built from MASTER kit v09 (2026-11-19)

    GETTING_STARTED.md   start here: build, verify, and submit your first (smoke-test) leaderboard entry
    starter/             the engine you start from: complete, correct, deliberately naive (see starter/README.md)
    tools/               check.py (correctness + gate), submit.py (package a submission), pack_images.py (view pack images),
                         pack_schema.py (the manifest schema; imported by the other three, do not move it)
    Au26_GPUCompute_ResNet18_FinalProject_Spec.pdf   the full specification
    REFERENCES.md        sources

The data pack is distributed separately: Course Canvas: Files/Final Project - ResNet18/resnet18_pack.zip
On the course AWS T4 instances it is pre-installed at C:\resnet18\resnet18_pack (along with Visual Studio 2026, CUDA, CMake and
Git), so there is nothing to download there; see "Working on the course AWS T4 instances" in GETTING_STARTED.md.

If you compare against your own PyTorch run on an Ampere-or-newer GPU, set torch.backends.cudnn.allow_tf32 = False
first (PyTorch uses TF32 for cuDNN convolutions by default there). The pack's references are true fp32 and check.py
never uses torch; your CUDA engine cannot use TF32 unless it asks for it, and the T4 has no TF32 hardware.
