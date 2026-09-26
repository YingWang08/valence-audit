#!/usr/bin/env Rscript
# =============================================================================
# analyze_lme4.R  --  mixed-effects sensitivity analyses (S1 File)
#
# Replaces both analyze_lme4.R and src/analyze_lme4.R of v1.0.0. Every number in a table
# comes from ONE fitted model, so t-test and F-test p-values of the same term agree
# (in v1.0.0 the S1 File combined an R fit and a statsmodels fit: language p = 0.022 vs 0.108).
#
# Specifications (asymmetry = cell-level a, strict parser, retained models only):
#   M0  a ~ dimension + language + (1|model) + (1|frame)                as submitted (treatment coding)
#   M1  a ~ 0 + dimension + lang_c + (1|model) + (1|model:dimension) + (1|frame)
#       mixed-model analogue of the primary model-level analysis: one estimate per dimension
#       tested against zero; model-by-dimension variation is a random effect
#   M2  as M1 with (1|family/model) instead of (1|model)                hierarchical, Reviewer 1 #13
#   M3  a ~ 0 + dimension + dimension:lang_c + (1|model) + (1|model:dimension) + (1|frame)
#       per-dimension English-Chinese differences
# frame = format|language|template; lang_c = +0.5 (zh) / -0.5 (en).
#
# Run from the repository root:
#   Rscript analyze_lme4.R                 # strict parser (primary)
#   Rscript analyze_lme4.R rating_legacy   # v1.0.0 parser, for comparison
#   Rscript analyze_lme4.R rating data/r1  # another data root
# Requires: install.packages(c("lme4", "lmerTest"))
# =============================================================================
suppressMessages({
  if (!requireNamespace("lme4", quietly = TRUE) || !requireNamespace("lmerTest", quietly = TRUE))
    stop("Install the packages first: install.packages(c('lme4','lmerTest'))")
  library(lmerTest)
})

args   <- commandArgs(trailingOnly = TRUE)
target <- if (length(args) >= 1) args[1] else "rating"
root   <- if (length(args) >= 2) args[2] else "data"
excluded <- c("openai/gpt-oss-20b", "nvidia/llama-3.3-nemotron-super-49b-v1.5")

d <- read.csv(file.path(root, "measured", "items.csv"), stringsAsFactors = FALSE)
d <- d[d$measure == target & !(d$model %in% excluded), ]
if (nrow(d) == 0) stop(sprintf("no rows with measure == '%s'", target))
dims <- c("productivity", "decision", "trust", "reliability", "moral", "emotion", "creativity", "worth")
d$dimension <- factor(d$dimension, levels = intersect(dims, unique(d$dimension)))
d$model  <- factor(d$model)
d$family <- factor(d$family)
d$frame  <- factor(d$frame)
d$language <- factor(d$language, levels = c("en", "zh"))
d$lang_c <- ifelse(d$language == "zh", 0.5, -0.5)
out <- file.path(root, "results")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
cat(sprintf("[lme4] measure=%s  N=%d  models=%d  families=%d  frames=%d\n",
            target, nrow(d), nlevels(d$model), nlevels(d$family), nlevels(d$frame)))

ctrl <- lmerControl(check.conv.singular = .makeCC("ignore", tol = 1e-4))
fit <- function(f) lmerTest::lmer(as.formula(f), data = d, REML = TRUE, control = ctrl)

save_fit <- function(m, name, bh_rows = NULL) {
  co <- as.data.frame(summary(m)$coefficients)
  co$term <- rownames(co)
  names(co) <- c("estimate", "se", "df", "t", "p", "term")
  ci <- qt(0.975, co$df)
  co$ci_lo <- co$estimate - ci * co$se
  co$ci_hi <- co$estimate + ci * co$se
  co$p_BH <- NA
  if (!is.null(bh_rows)) {
    idx <- grepl(bh_rows, co$term)
    co$p_BH[idx] <- p.adjust(co$p[idx], method = "BH")
  }
  co <- co[, c("term", "estimate", "se", "df", "t", "p", "p_BH", "ci_lo", "ci_hi")]
  write.csv(co, file.path(out, sprintf("lme4_%s_%s_fixed.csv", target, name)), row.names = FALSE)
  vc <- as.data.frame(lme4::VarCorr(m))[, c("grp", "vcov", "sdcor")]
  vc$pct <- 100 * vc$vcov / sum(vc$vcov)
  write.csv(vc, file.path(out, sprintf("lme4_%s_%s_varcomp.csv", target, name)), row.names = FALSE)
  cat(sprintf("\n==== %s ====  singular: %s\n", name, lme4::isSingular(m)))
  print(format(co, digits = 3), row.names = FALSE)
  print(format(vc, digits = 3), row.names = FALSE)
  invisible(co)
}

m0 <- fit("asymmetry ~ dimension + language + (1|model) + (1|frame)")
save_fit(m0, "M0_as_submitted")
a0 <- as.data.frame(anova(m0, type = 2))
a0$term <- rownames(a0)
write.csv(a0, file.path(out, sprintf("lme4_%s_M0_anova_type2.csv", target)), row.names = FALSE)
cat("\nType II ANOVA (Satterthwaite), same fit as M0:\n"); print(a0)

m1 <- fit("asymmetry ~ 0 + dimension + lang_c + (1|model) + (1|model:dimension) + (1|frame)")
save_fit(m1, "M1_dimension_means", bh_rows = "^dimension")

m2 <- fit("asymmetry ~ 0 + dimension + lang_c + (1|family/model) + (1|frame)")
save_fit(m2, "M2_family_nested", bh_rows = "^dimension")

m3 <- fit("asymmetry ~ 0 + dimension + dimension:lang_c + (1|model) + (1|model:dimension) + (1|frame)")
save_fit(m3, "M3_language_by_dimension", bh_rows = ":lang_c")

sink(file.path(out, "lme4_session_info.txt")); print(sessionInfo()); sink()
cat(sprintf("\n[lme4] done; results in %s/lme4_%s_*.csv\n", out, target))
