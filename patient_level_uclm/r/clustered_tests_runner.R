# clustered_tests_runner.R -- runs the clustered tests for every job in a manifest.
#
#   Durkalski et al. (2003): clust.bin.pair 0.1.2, method = "durkalski"
#   Obuchowski (1997):       clusteredROC() from funcs_clusteredROC.R (Lerner
#                            Research Institute), sourced UNCHANGED
#
# Usage: Rscript clustered_tests_runner.R <manifest.csv> <results.csv> <funcs_clusteredROC.R>
#
# Refuses to run (stops with an error) if any input has missing values, the wrong
# number of rows or patients, or values outside 0/1 where 0/1 is required.
# Writes every number with 17 significant digits so Python reads it back exactly.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript clustered_tests_runner.R <manifest.csv> <results.csv> <funcs_clusteredROC.R>")
}

suppressPackageStartupMessages(library(clust.bin.pair))
if (as.character(packageVersion("clust.bin.pair")) != "0.1.2") {
  stop("clust.bin.pair must be version 0.1.2")
}
source(args[3])   # defines clusteredROC(); the file itself is never modified

manifest <- read.csv(args[1], stringsAsFactors = FALSE, colClasses = c(job_id = "character"))

as_text <- function(v) {
  if (length(v) != 1) stop("internal error: result field is not a single value")
  if (is.na(v)) return("NA")
  if (is.numeric(v)) return(sprintf("%.17g", as.double(v)))
  as.character(v)
}

results <- vector("list", nrow(manifest))

for (k in seq_len(nrow(manifest))) {
  job <- manifest[k, ]
  fail <- function(msg) stop(sprintf("job %s: %s", job$job_id, msg), call. = FALSE)

  d <- read.csv(job$file, stringsAsFactors = FALSE, colClasses = c(patient_id = "character"))
  if (anyNA(d)) fail("missing values in input")
  if (nrow(d) != job$expected_n) fail(sprintf("%d rows, expected %d", nrow(d), job$expected_n))
  n_clusters <- length(unique(d$patient_id))
  if (n_clusters != job$expected_clusters) {
    fail(sprintf("%d patients, expected %d", n_clusters, job$expected_clusters))
  }

  res <- list(job_id = job$job_id, test = job$test, n_obs = nrow(d), n_clusters = n_clusters,
              auc_a = NA, auc_b = NA, se_a = NA, se_b = NA, abs_diff = NA, diff_se = NA,
              statistic = NA, p_value = NA, method = NA, n_discordant_patients = NA)

  if (job$test == "obuchowski") {
    if (!all(d$y %in% c(0, 1))) fail("y must be 0/1")
    if (length(unique(d$patient_id[d$y == 1])) < 2 || length(unique(d$patient_id[d$y == 0])) < 2) {
      fail("needs at least 2 patients with positives and 2 with negatives")
    }
    # clusteredROC() prints a summary; capture it so the console stays readable
    printed <- capture.output(
      r <- clusteredROC(predictor1 = d$score_a, predictor2 = d$score_b, response = d$y,
                        clusterID = d$patient_id, alpha = 0.05, level = 1))
    if (is.na(r$p.value)) fail("p-value is NA (standard error of the difference is 0?)")
    res$auc_a    <- r$auc[1]
    res$auc_b    <- r$auc[2]
    res$se_a     <- r$auc.se[1]
    res$se_b     <- r$auc.se[2]
    res$abs_diff <- r$diff
    res$diff_se  <- r$diff.se
    res$p_value  <- r$p.value
    res$method   <- "obuchowski1997"

  } else if (job$test == "durkalski") {
    if (!all(d$correct_a %in% c(0, 1)) || !all(d$correct_b %in% c(0, 1))) {
      fail("correct_a and correct_b must be 0/1")
    }
    # per-patient 2x2 counts: a = both right, b = A right/B wrong, c = A wrong/B right, d = both wrong
    patients <- sort(unique(d$patient_id))
    g <- match(d$patient_id, patients)
    nb <- length(patients)
    ak <- tabulate(g[d$correct_a == 1 & d$correct_b == 1], nbins = nb)
    bk <- tabulate(g[d$correct_a == 1 & d$correct_b == 0], nbins = nb)
    ck <- tabulate(g[d$correct_a == 0 & d$correct_b == 1], nbins = nb)
    dk <- tabulate(g[d$correct_a == 0 & d$correct_b == 0], nbins = nb)
    if (sum(ak + bk + ck + dk) != nrow(d)) fail("per-patient counts do not add up to the number of images")
    res$n_discordant_patients <- sum(bk + ck > 0)

    # Pre-specified rule: if every patient's disagreements cancel (b_k == c_k for all k,
    # including no disagreements at all), the statistic is 0/0 -> report p = 1, labelled.
    if (sum(((bk - ck) / (ak + bk + ck + dk))^2) == 0) {
      res$p_value <- 1
      res$method  <- "no_net_discordance"
    } else {
      r <- clust.bin.pair(ak, bk, ck, dk, method = "durkalski")
      res$statistic <- unname(r$statistic)
      res$p_value   <- unname(r$p.value)
      res$method    <- "durkalski"
    }

  } else {
    fail(sprintf("unknown test '%s'", job$test))
  }
  results[[k]] <- res
}

cols <- names(results[[1]])
lines <- c(paste(cols, collapse = ","),
           vapply(results, function(res) {
             paste(vapply(cols, function(cn) as_text(res[[cn]]), character(1)), collapse = ",")
           }, character(1)))
writeLines(lines, args[2])
cat(sprintf("clustered_tests_runner.R: %d jobs completed\n", length(results)))
