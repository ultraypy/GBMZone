# GBMZone checkpoints

## Checkpoints for Figures 4 and S26

Figure S26 compares six input configurations on the same 38 GBMspace sections and pathology scoring positions as Figure 4. The fold order is AT10, AT13, AT15, AT3, AT4 and AT6, with one held-out patient per fold.

| Group | Configuration | File | SHA-256 |
| --- | --- | --- | --- |
| 1 | RNA, pathways and H&E | `pathway+he+tran.pt` | `36613b267e35699bdbd5a3c96ce07c91495293a39fc6378a0874934d9f602a77` |
| 2 | RNA and H&E | `he+tran.pt` | `bbd8043d5a2d031059970d0a26a129d9698ab46b805e5ec588f92c81140f5092` |
| 3 | RNA and pathways | `pathway+tran_2of3.pt` | `f8b3a35245d9d74953c8f27a5a3c7c62f877bcb5b36b076c686b6e54e4d79633` |
| 4 | RNA | `tran.pt` | `e1bfffe8406c10d6b7bc7aefa4ec619d29e90c208f0fc2c0bd693576919960e9` |
| 5 | H&E | `he.pt` | `25610ef6703e14f80bd7265d8a1892bae1fea8ec259ab3fcf066d191d74e8fa9` |
| 6 | Pathways and H&E | `pathway+he.pt` | `a7bdbccd1c8430c2e6d79b6bf01410a77d94031d6356ffbac7728ee11680bd60` |

## Original checkpoints

The four original archives each contain six folds trained on the external GBMspace cohort. In fold order, the held-out patients were AT10, AT13, AT15, AT3, AT4 and AT6. Internal Stereo-seq regional analyses used the mean of six RNA-plus-pathway predictions and an RNA-only sensitivity analysis.

| Configuration | Original file | SHA-256 |
| --- | --- | --- |
| RNA | `tran.pt` | `5d8c9e6568f971c266a7655f93ad4c632a376a7831bdc5ef5f4c7c2fc156061a` |
| RNA and pathways | `pathway+tran.pt` | `88893383c1958a68bddc8a4d1cb5c1011c2d7b601234d801d501b412867aa06c` |
| RNA and H&E | `he+tran.pt` | `b15541bcca4e2d9376faf4197c45c7e16c5267cbdb28b60669c31cfcf6a17aa2` |
| RNA, pathways and H&E | `pathway+he+tran.pt` | `13caabc40588ef3d6911bdd4fc2b132c4a3048b330aa68b8f4533d92bbf35a77` |

This code release includes the training and inference programs and the identifiers above. Requests for the original checkpoint files can be directed to the study lead contact listed in the manuscript. The numerical figure workflows use saved predictions and patient measurements and run without loading a checkpoint.
