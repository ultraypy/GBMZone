# Original GBMZone checkpoints

The four original archives each contain six folds trained on the external GBMspace cohort. In fold order, the held-out patients were AT10, AT13, AT15, AT3, AT4 and AT6. Internal Stereo-seq regional analyses used the mean of six RNA-plus-pathway predictions and an RNA-only sensitivity analysis.

| Configuration | Original file | SHA-256 |
| --- | --- | --- |
| RNA | `tran.pt` | `5d8c9e6568f971c266a7655f93ad4c632a376a7831bdc5ef5f4c7c2fc156061a` |
| RNA and pathways | `pathway+tran.pt` | `88893383c1958a68bddc8a4d1cb5c1011c2d7b601234d801d501b412867aa06c` |
| RNA and H&E | `he+tran.pt` | `b15541bcca4e2d9376faf4197c45c7e16c5267cbdb28b60669c31cfcf6a17aa2` |
| RNA, pathways and H&E | `pathway+he+tran.pt` | `13caabc40588ef3d6911bdd4fc2b132c4a3048b330aa68b8f4533d92bbf35a77` |

This code release includes the training and inference programs and the identifiers above. Requests for the original checkpoint files can be directed to the study lead contact listed in the manuscript. The numerical figure workflows use saved predictions and patient measurements and run without loading a checkpoint.
