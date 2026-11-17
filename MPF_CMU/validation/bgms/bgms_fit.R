# Fit an Ising model with the R package bgms (Bayesian, pseudo-likelihood based) to a data file in mpf's
# format, and write the result in mpf's parameterisation, p(s) ~ exp(sum_{a<b} J_ab s_a s_b + sum_a h_a s_a)
# with s = +1 / -1.
#
#   Rscript bgms_fit.R DATA OUT_PREFIX EDGE_SELECTION(0|1) NA_ACTION(listwise|impute) SEED [cores=4] [iter=2000] [warmup=2000] [scale=S] [draws=K]
#
# Everything not given is the package default: 4 chains of 2000 warm-up + 2000 kept iterations of NUTS, a
# Normal(0, 1) prior on the pairwise parameters (scale=S changes its standard deviation), a Bernoulli(1/2)
# prior on each edge when EDGE_SELECTION is 1. `cores` is the number of chains run at once.
#
# Writes OUT_PREFIX_params.dat   posterior mean, n(n-1)/2 couplings then n fields (readable by `mpf -k`)
#        OUT_PREFIX_summary.tsv  per parameter: posterior mean, 2.5% and 97.5% quantiles, and for couplings
#                                the posterior inclusion probability (edge selection only)
#        OUT_PREFIX_meta.txt     timings, package version, sampler diagnostics
#        OUT_PREFIX_draws.txt    (draws=K) K evenly spaced posterior draws, one per line, in the order of _params.dat
#
# bgms codes answers 0/1 and writes p(x) ~ exp(sum_a mu_a x_a + sum_{a<b} 2 omega_ab x_a x_b), so
#   J_ab = omega_ab / 2,   h_a = (mu_a + sum_b omega_ab) / 2
# (checked against a known model by bgms_check.py). The conversion is applied to every posterior draw, so
# the intervals are intervals for J and h. A question that every row answers the same way cannot be given
# to bgms; it is left out of the fit, its couplings are set to zero (with no interval) and its field to the
# value that reproduces the frequency (count + 1/2) / (rows + 1). If fewer than two questions vary (it
# happens with strongly ordered data), bgms cannot be run at all and every question is treated that way.
# Row weights must all be 1.
lib <- Sys.getenv("BGMS_LIB"); if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) stop("usage: Rscript bgms_fit.R DATA OUT_PREFIX EDGE_SELECTION NA_ACTION SEED [cores=] [iter=] [warmup=] [scale=] [draws=]")
opt <- list(cores = "4", iter = "2000", warmup = "2000", scale = "", draws = "0")
for (kv in strsplit(args[-(1:5)], "=")) { if (length(kv) != 2 || !(kv[1] %in% names(opt))) stop("unknown option"); opt[[kv[1]]] <- kv[2] }
cores <- as.integer(opt$cores); iter <- as.integer(opt$iter); warmup <- as.integer(opt$warmup); ndraws <- as.integer(opt$draws)
# the package runs its chains on a thread pool that otherwise takes every core of the machine
Sys.setenv(RCPP_PARALLEL_NUM_THREADS = as.character(cores), OMP_NUM_THREADS = "1")
suppressPackageStartupMessages(library(bgms))
if (requireNamespace("RcppParallel", quietly = TRUE)) RcppParallel::setThreadOptions(numThreads = cores)

lines <- readLines(args[1]); m <- as.integer(lines[1]); n <- as.integer(lines[2])
rows <- strsplit(trimws(lines[3:(2 + m)]), "[ \t]+")
x <- t(sapply(rows, function(r) { ch <- strsplit(r[1], "")[[1]]; suppressWarnings(ifelse(ch == "X", NA_integer_, as.integer(ch))) }))
if (any(sapply(rows, function(r) as.numeric(r[2])) != 1)) stop("bgms has no row weights: all weights must be 1")
if (args[4] == "listwise") x <- x[stats::complete.cases(x), , drop = FALSE]
constant <- apply(x, 2, function(v) length(unique(v[!is.na(v)])) < 2)
ran <- sum(!constant) >= 2                 # bgms needs at least two questions that vary;
if (!ran) constant[] <- TRUE               # otherwise nothing is fitted and every question is treated as constant
keep <- which(!constant); nk <- length(keep)
pair <- which(upper.tri(matrix(0, nk, nk)), arr.ind = TRUE); pair <- pair[order(pair[, 1], pair[, 2]), , drop = FALSE]
Jd <- matrix(0, 1, 0); hd <- matrix(0, 1, 0); incl <- numeric(0)      # posterior draws (rows) of the fitted couplings and fields
meta <- c(prior_scale = "not_run", chains = "0", iter = "0", warmup = "0", elapsed_seconds = "0", cpu_seconds = "0",
          max_rhat = "NaN", min_ess = "NaN", divergences = "NA")
if (ran) {
  xk <- x[, keep, drop = FALSE]; colnames(xk) <- paste0("V", keep)
  t0 <- proc.time()
  call <- list(xk, edge_selection = (args[3] == "1"), na_action = args[4], seed = as.integer(args[5]), chains = 4, cores = cores,
               iter = iter, warmup = warmup, display_progress = "none", verbose = FALSE)
  if (nzchar(opt$scale)) call$interaction_prior <- normal_prior(scale = as.numeric(opt$scale))
  fit <- do.call(bgm, call)
  el <- proc.time() - t0
  stopifnot(identical(fit$raw_samples$parameter_names$pairwise, paste0("V", keep[pair[, 1]], "-V", keep[pair[, 2]])))
  om <- do.call(rbind, fit$raw_samples$pairwise); mu <- do.call(rbind, fit$raw_samples$main)      # draws x parameters
  Jd <- om / 2
  hd <- mu / 2
  for (e in seq_len(nrow(pair))) { hd[, pair[e, 1]] <- hd[, pair[e, 1]] + om[, e] / 2; hd[, pair[e, 2]] <- hd[, pair[e, 2]] + om[, e] / 2 }
  incl <- if (args[3] == "1") fit$posterior_mean_indicator[pair] else rep(NA, nrow(pair))
  sp <- fit$posterior_summary_pairwise; sm <- fit$posterior_summary_main
  meta <- c(prior_scale = ifelse(nzchar(opt$scale), opt$scale, "default"), chains = "4", iter = as.character(iter), warmup = as.character(warmup),
            elapsed_seconds = sprintf("%.2f", el[["elapsed"]]),
            cpu_seconds = sprintf("%.2f", el[["user.self"]] + el[["sys.self"]] + ifelse(is.na(el[["user.child"]]), 0, el[["user.child"]] + el[["sys.child"]])),
            max_rhat = sprintf("%.4f", max(c(sp$Rhat, sm$Rhat), na.rm = TRUE)), min_ess = sprintf("%.1f", min(c(sp$n_eff, sm$n_eff), na.rm = TRUE)),
            divergences = as.character(tryCatch(sum(unlist(fit$nuts_diag$divergent)), error = function(e) NA)))
}

# posterior means and intervals for all n questions; a parameter bgms did not fit has no interval
J <- matrix(0, n, n); Jlo <- matrix(NA, n, n); Jhi <- Jlo; Jin <- Jlo; h <- numeric(n); hlo <- rep(NA, n); hhi <- hlo
for (e in seq_len(nrow(pair))) {
  a <- keep[pair[e, 1]]; b <- keep[pair[e, 2]]; q <- quantile(Jd[, e], c(0.025, 0.975))
  J[a, b] <- mean(Jd[, e]); Jlo[a, b] <- q[1]; Jhi[a, b] <- q[2]; Jin[a, b] <- incl[e]
}
for (k in seq_len(nk)) { q <- quantile(hd[, k], c(0.025, 0.975)); h[keep[k]] <- mean(hd[, k]); hlo[keep[k]] <- q[1]; hhi[keep[k]] <- q[2] }
for (a in which(constant)) {            # p(s_a = +1) = (count + 1/2) / (rows + 1), no couplings
  v <- x[, a]; pr <- (sum(v == 1, na.rm = TRUE) + 0.5) / (sum(!is.na(v)) + 1); h[a] <- 0.5 * log(pr / (1 - pr))
}
ab <- which(upper.tri(J), arr.ind = TRUE); ab <- ab[order(ab[, 1], ab[, 2]), , drop = FALSE]
cat(sprintf("%.10e", c(J[ab], h)), file = paste0(args[2], "_params.dat"), sep = " ")
out <- rbind(data.frame(kind = "J", a = ab[, 1] - 1, b = ab[, 2] - 1, mean = J[ab], q025 = Jlo[ab], q975 = Jhi[ab], inclusion = Jin[ab]),
             data.frame(kind = "h", a = seq_len(n) - 1, b = NA, mean = h, q025 = hlo, q975 = hhi, inclusion = NA))
write.table(out, paste0(args[2], "_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
if (ndraws > 0) {                       # evenly spaced draws, mapped to all n nodes like the posterior mean
  pick <- unique(round(seq(1, nrow(Jd), length.out = ndraws)))
  D <- matrix(0, length(pick), nrow(ab) + n)
  col_of <- function(a, b) which(ab[, 1] == a & ab[, 2] == b)
  for (e in seq_len(nrow(pair))) D[, col_of(keep[pair[e, 1]], keep[pair[e, 2]])] <- Jd[pick, e]
  for (k in seq_len(nk)) D[, nrow(ab) + keep[k]] <- hd[pick, k]
  for (a in which(constant)) D[, nrow(ab) + a] <- h[a]
  write.table(format(D, digits = 10, scientific = TRUE), paste0(args[2], "_draws.txt"), quote = FALSE, row.names = FALSE, col.names = FALSE)
}
meta <- c(bgms = as.character(packageVersion("bgms")), rows = nrow(x), nodes = n, constant_columns = sum(apply(x, 2, function(v) length(unique(v[!is.na(v)])) < 2)),
          edge_selection = args[3], na_action = args[4], cores = cores, meta)
writeLines(paste(names(meta), meta), paste0(args[2], "_meta.txt"))
