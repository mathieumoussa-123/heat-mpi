# Part 2 review and LLM-record checklist

- [ ] Export and preserve the COMPLETE Part 2 user prompt and complete AI responses with timestamp/time zone, model, team member, and changed-file list. **This checklist is not a log or a substitute for the complete transcript.**
- [ ] Understand uneven strip partition, ghost layout, and the two MPI_Sendrecv calls; be able to explain why they cannot deadlock like naive blocking sends.
- [ ] Verify the C++17 build using Open MPI on all VMs, Part 1 regression tests, and CTest.
- [ ] Run N=8 Case 2, one step, P=2 and P=3 and inspect ghost rows.
- [ ] Verify byte-for-byte field equality for P=1,2,3,4 across several N including uneven partitions.
- [ ] Save the actual test_strips.py CSV/log output from **at least two VMs**, and record host placement.
- [ ] If mixed architectures, separately report maximal absolute output difference and same-architecture bitwise equality.
- [ ] Note timing field semantics: each rank's loop vs independent max computation/communication reductions.
- [ ] Include logs and results/evidence in repository/submission without leaking private SSH keys or tokens.
