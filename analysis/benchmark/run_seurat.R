# Use Seurat's reference PCA, anchors, and class-score transfer on each held-out patient.
suppressPackageStartupMessages({
    library(hdf5r)
    library(Matrix)
    library(Seurat)
})

args <- commandArgs(trailingOnly = TRUE)
patients <- if (length(args)) args else c("AT10", "AT13", "AT15", "AT3", "AT4", "AT6")
destination <- file.path("fold_predictions", "Seurat")
dir.create(destination, recursive = TRUE, showWarnings = FALSE)
classes <- c("C0", "C1", "C2")
set.seed(1130)

make_assay <- function(group, genes, prefix) {
    expression <- group[["expression"]]
    dimensions <- rev(as.integer(h5attr(expression, "shape")))
    m <- sparseMatrix(
        i = 1L + as.integer(expression[["indices"]][]),
        p = as.integer(expression[["indptr"]][]),
        x = as.numeric(expression[["data"]][]),
        dims = dimensions,
        dimnames = list(genes, paste0(prefix, seq_len(dimensions[[2]])))
    )
    CreateSeuratObject(counts = CreateAssay5Object(data = m), assay = "RNA")
}

run_fold <- function(patient, file) {
    h5 <- H5File$new(file.path("fold_inputs", paste0(patient, ".h5")), mode = "r")
    on.exit(h5$close_all(), add = TRUE)
    stopifnot(h5attr(h5, "held_out_patient") == patient)
    reference <- h5[["reference"]]
    query_group <- h5[["query"]]
    stopifnot(!"label" %in% names(query_group))
    genes <- h5[["genes"]][]
    ref <- make_assay(reference, genes, "ref")
    query <- make_assay(query_group, genes, "query")
    labels <- setNames(classes[1L + as.integer(reference[["label"]][])], colnames(ref))
    stopifnot(setequal(unique(labels), classes))
    ref <- ScaleData(ref, features = genes, verbose = FALSE)
    ref <- RunPCA(ref, features = genes, npcs = 30, seed.use = 1130, verbose = FALSE)
    anchors <- FindTransferAnchors(
        reference = ref, query = query, reduction = "pcaproject",
        reference.reduction = "pca", features = genes, dims = 1:30
    )
    predictions <- TransferData(anchorset = anchors, refdata = labels, dims = 1:30)
    scores <- predictions[colnames(query), paste0("prediction.score.", classes), drop = FALSE]
    stopifnot(nrow(scores) == ncol(query), all(is.finite(as.matrix(scores))))
    stopifnot(max(abs(rowSums(scores) - 1)) < 1e-5)
    names(scores) <- c("CT", "LE_IT", "necrosis_related")
    output <- data.frame(
        sample = query_group[["sample"]][], unit_id = query_group[["unit_id"]][],
        held_out_patient = patient, scores, row.names = NULL
    )
    write.csv(output, file = paste0(file, ".build"), row.names = FALSE)
    stopifnot(file.rename(paste0(file, ".build"), file))
}

cat("Seurat", as.character(packageVersion("Seurat")), "\n")
for (patient in patients) {
    file <- file.path(destination, paste0(patient, ".csv"))
    if (!file.exists(file)) {
        cat("BEGIN", patient, format(Sys.time()), "\n")
        run_fold(patient, file)
        cat("DONE", patient, format(Sys.time()), "\n")
        gc()
    }
}
