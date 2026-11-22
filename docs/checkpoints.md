# GBMZone checkpoints

## Checkpoints for Figures 4 and S26

Figure 4 uses the RNA-plus-pathway checkpoint listed as group 3. Figure S26 compares six configurations on a shared subset of 19 benchmark sections. The fold order is AT10, AT13, AT15, AT3, AT4 and AT6, with one held-out patient per fold.

| Group | Configuration | File | SHA-256 |
| --- | --- | --- | --- |
| 1 | RNA, pathways and H&E | `pathway+he+tran_1of3.pt` | `fb4afcad3ac4936d038e87325dc661e4d529f1cbe56970edab2b5da68e03fdfd` |
| 2 | RNA and H&E | `he+tran_1of4.pt` | `beb2013a111d7b7f32ec60eb78bb8b8755caa44d0891d29b86d13df9ea8bfc88` |
| 3 | RNA and pathways | `pathway+tran_2of3.pt` | `f8b3a35245d9d74953c8f27a5a3c7c62f877bcb5b36b076c686b6e54e4d79633` |
| 4 | RNA | `tran_1of3.pt` | `2f557e85ffca8650af3e192487472589b9c16e280c202ec7ea80bd4dc7095195` |
| 5 | H&E | `he.pt` | `3f70fd7cd9ce7721e5f476b4307fa6df351e870165c32010339a7a17b291c86a` |
| 6 | Pathways and H&E | `pathway+he_1of4.pt` | `8eeecdf57682e7a908f550b95ff8c1af099c36e3456e67f51b318cf4cb7478df` |

## Original checkpoints

The four original archives each contain six folds trained on the external GBMspace cohort. In fold order, the held-out patients were AT10, AT13, AT15, AT3, AT4 and AT6. Internal Stereo-seq regional analyses used the mean of six RNA-plus-pathway predictions and an RNA-only sensitivity analysis.

| Configuration | Original file | SHA-256 |
| --- | --- | --- |
| RNA | `tran.pt` | `5d8c9e6568f971c266a7655f93ad4c632a376a7831bdc5ef5f4c7c2fc156061a` |
| RNA and pathways | `pathway+tran.pt` | `88893383c1958a68bddc8a4d1cb5c1011c2d7b601234d801d501b412867aa06c` |
| RNA and H&E | `he+tran.pt` | `b15541bcca4e2d9376faf4197c45c7e16c5267cbdb28b60669c31cfcf6a17aa2` |
| RNA, pathways and H&E | `pathway+he+tran.pt` | `13caabc40588ef3d6911bdd4fc2b132c4a3048b330aa68b8f4533d92bbf35a77` |

This code release includes the training and inference programs and the identifiers above. Requests for the original checkpoint files can be directed to the study lead contact listed in the manuscript. The numerical figure workflows use saved predictions and patient measurements and run without loading a checkpoint.
