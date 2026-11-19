# Part 3 — Cartesian block decomposition: review and LLM audit checklist

- [ ] **Export the full real user prompts and AI responses** for this development conversation (and prior conversations that informed the code) with timestamps, model, team member, and changed-file list. This file is only a checklist, NOT the transcript required by the professor.
- [ ] Review exact requirements in `heat_mpi_project.pdf` Part 3, and our mathematical Task 8.2 communication model.
- [ ] Understand `MPI_Dims_create`, nonperiodic `MPI_Cart_create`, `MPI_Cart_shift`, and quotient/remainder block partitioning.
- [ ] Explain the four `MPI_Sendrecv` calls and `MPI_Type_vector` with stride `nx+2`; no manual column packing.
- [ ] Show why corner ghosts are unused for a five-point stencil, but needed for a nine-point stencil.
- [ ] Build on all three Ubuntu VMs; run Part 1 regression, Part 2 regression, Part 3 local P=1..4 regression.
- [ ] Save evidence for a two-VM P=4/2x2 block run with `cmp` matching serial (real Open MPI).
- [ ] Save multi-VM correctness results for P=1,2,3,4,6,8 and uneven blocks.
- [ ] Verify that only MPI rank 0 writes the field, including physical boundaries, with exact matching bytes.
- [ ] Record the actual rank-to-host placement (`mpirun ... hostname`).
- [ ] Run and interpret corrected Part 0-based predictions, documenting per-byte to per-double conversion.
- [ ] Measure P=4,8 and N=512,2048,8192 and compare predicted vs actual strips/blocks (Part 3), with repeats and medians (Part 4 measurement rules).
- [ ] Preserve C++ sources, Python scripts, raw logs and CSVs; do not commit credentials, keys or private tokens.

## Actual implementation change manifest (2026-11-19, GPT-6)

- New: `src/heat_blocks.cpp`, `scripts/test_blocks.py`, `scripts/compare_decomps.py`, `PART3_BLOCKS_GUIDE.md`, `data/part0_corrected_fit_summary.csv`, this checklist.
- Modified: `src/heat.hpp`, `src/heat_main.cpp`, `CMakeLists.txt`, `README.md`, `scripts/test_serial.py` (replace obsolete expected rejection of `blocks` with an unknown decomposition).
- Unchanged: numerical body of `src/heat_serial.cpp`, `src/heat_strips.cpp`, and `src/heat_io.cpp`.
- Test context: local C++17 compilation and file-based MPI protocol emulation; genuine multi-VM MPI tests **not yet run** in the authoring environment.
- Team member / real prompt transcript: team must populate from their actual records, not invent.
