#!/usr/bin/env Rscript
# =============================================================================
# analyze_lme4.R —— 论文级最终推断（交叉随机效应）
#
# 为什么需要它：Python 端 analyze.py 是“开发级”推断（以模型为单元的 cluster
# bootstrap），稳健但功效有限、且只把“模型”当随机效应。审稿人最认可的，是把
# 「模型」和「措辞框架(frame)」同时作为交叉随机效应（crossed random effects）的
# 混合模型——它对题项措辞与模型双重非独立做部分汇聚(partial pooling)，功效更高，
# 也最贴近本设计的生成结构。
#
#   asymmetry ~ dimension + language [+ alignment_stage] + (1|model) + (1|frame)
#
#   - dimension      : 固定效应（H2 的核心：各维度不对称的方向与大小）
#   - language       : 固定效应（控制中英差异；只有一种语言时自动省略）
#   - alignment_stage: 固定效应（H3：base vs instruct 的关联，仅当存在 base 时纳入）
#   - (1|model)      : 模型随机截距（同一模型的多条观测不独立）
#   - (1|frame)      : 措辞框架随机截距（frame = format|language|template，
#                      即“同一句式”的多条观测不独立——心理语言学里的 item 随机效应）
#
# 注意：dimension 是固定效应，所以不要再加 (1|dimension)（会与固定效应共线）。
#
# 输入：data/measured/items.csv（measure.py 已同时导出 csv 与 parquet）
# 输出：data/results/lme4_H1.csv, lme4_H2.csv, lme4_H3.csv
#
# 运行： Rscript analyze_lme4.R              # 默认主指标 = rating
#        Rscript analyze_lme4.R text_m1      # 改用自由文本 M1 复核
#
# 依赖： install.packages(c("lme4","lmerTest"))   # arrow 可选（仅在读 parquet 时需要）
# =============================================================================

suppressWarnings(suppressMessages({
  ok <- require(lme4) && require(lmerTest)
}))
if (!ok) {
  stop("缺少 lme4 / lmerTest。请先在 R 里运行：install.packages(c('lme4','lmerTest'))")
}

args   <- commandArgs(trailingOnly = TRUE)
target <- ifelse(length(args) >= 1, args[1], "rating")   # 主指标：rating（默认）或 text_m1

# ---- 读数据（优先 csv；没有就尝试 parquet via arrow）----
csv_path <- "data/measured/items.csv"
pq_path  <- "data/measured/items.parquet"
if (file.exists(csv_path)) {
  d <- read.csv(csv_path, stringsAsFactors = FALSE)
} else if (file.exists(pq_path) && requireNamespace("arrow", quietly = TRUE)) {
  d <- as.data.frame(arrow::read_parquet(pq_path))
} else {
  stop("找不到 data/measured/items.csv。请先跑 python run_all.py（或 measure 阶段）。")
}

d <- d[d$measure == target, ]
if (nrow(d) == 0) stop(sprintf("measure=='%s' 没有数据。可选: rating / text_m1。", target))

# 类型整理
for (col in c("model", "dimension", "language", "alignment_stage", "frame")) {
  if (col %in% names(d)) d[[col]] <- as.factor(d[[col]])
}
cat(sprintf("\n[lme4] 主指标=%s  N=%d  模型数=%d  frame数=%d  维度数=%d\n",
            target, nrow(d), nlevels(d$model), nlevels(d$frame), nlevels(d$dimension)))

n_lang  <- nlevels(d$language)
n_stage <- if ("alignment_stage" %in% names(d)) nlevels(d$alignment_stage) else 1
has_base <- ("alignment_stage" %in% names(d)) && all(c("base", "instruct") %in% levels(d$alignment_stage))

# 随机效应：仅当某因子有 >=2 个水平才纳入，避免奇异/不可识别
re_terms <- c()
if (nlevels(d$model) >= 2) re_terms <- c(re_terms, "(1|model)")
if (nlevels(d$frame) >= 2) re_terms <- c(re_terms, "(1|frame)")
if (length(re_terms) == 0) stop("model 与 frame 都不足 2 个水平，无法做混合模型。")
re_str <- paste(re_terms, collapse = " + ")

results_dir <- "data/results"
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

safe_fit <- function(formula_str) {
  tryCatch(
    lmerTest::lmer(as.formula(formula_str), data = d, REML = TRUE,
                   control = lmerControl(check.conv.singular = .makeCC("ignore", tol = 1e-4))),
    error = function(e) { cat("  [警告] 拟合失败：", conditionMessage(e), "\n"); NULL }
  )
}

# ---------------------------------------------------------------------------
# H1：总体不对称（截距即总体均值，控制语言/阶段为协变量时改看 emmeans；此处给纯截距版）
# ---------------------------------------------------------------------------
cat("\n================ H1 总体不对称 ================\n")
f_h1 <- sprintf("asymmetry ~ 1 + %s", re_str)
m1 <- safe_fit(f_h1)
if (!is.null(m1)) {
  co <- summary(m1)$coefficients
  est <- co["(Intercept)", "Estimate"]
  se  <- co["(Intercept)", "Std. Error"]
  p   <- co["(Intercept)", "Pr(>|t|)"]
  cat(sprintf("  总体不对称 = %+.4f  (SE=%.4f, 95%%CI[%+.4f, %+.4f], p=%.3g)  (>0 抬高机器)\n",
              est, se, est - 1.96 * se, est + 1.96 * se, p))
  write.csv(data.frame(estimate = est, se = se,
                       ci_low = est - 1.96 * se, ci_high = est + 1.96 * se, p = p),
            file.path(results_dir, "lme4_H1.csv"), row.names = FALSE)
}

# ---------------------------------------------------------------------------
# H2：分维度（0 + dimension 让每个系数=该维度均值，直接对 0 检验；再 BH 校正）
# ---------------------------------------------------------------------------
cat("\n================ H2 分维度不对称 ================\n")
fixed_h2 <- "asymmetry ~ 0 + dimension"
if (n_lang >= 2)  fixed_h2 <- paste(fixed_h2, "+ language")
f_h2 <- sprintf("%s + %s", fixed_h2, re_str)
m2 <- safe_fit(f_h2)
if (!is.null(m2)) {
  co <- as.data.frame(summary(m2)$coefficients)
  dim_rows <- grepl("^dimension", rownames(co))
  h2 <- co[dim_rows, , drop = FALSE]
  h2$dimension <- sub("^dimension", "", rownames(h2))
  h2$p_fdr <- p.adjust(h2[["Pr(>|t|)"]], method = "BH")
  h2 <- h2[order(-h2$Estimate), ]
  cat(sprintf("  %-13s %9s %9s %9s\n", "dimension", "estimate", "p_raw", "p_FDR"))
  for (i in seq_len(nrow(h2))) {
    star <- ifelse(h2$p_fdr[i] < 0.05, "*", " ")
    cat(sprintf("  %-13s %+9.4f %9.3g %9.3g %s\n",
                h2$dimension[i], h2$Estimate[i], h2[["Pr(>|t|)"]][i], h2$p_fdr[i], star))
  }
  out <- data.frame(dimension = h2$dimension, estimate = h2$Estimate,
                    se = h2[["Std. Error"]], p_raw = h2[["Pr(>|t|)"]], p_fdr = h2$p_fdr)
  write.csv(out, file.path(results_dir, "lme4_H2.csv"), row.names = FALSE)
  cat("  (competence 维度应 >0、moral/emotion/worth/trust 应 <0；以此核对 H2 的方向假设)\n")
}

# ---------------------------------------------------------------------------
# H3：base vs instruct（关联，非因果）——仅当数据里同时有 base 与 instruct
# ---------------------------------------------------------------------------
cat("\n================ H3 base vs instruct（关联，非因果）================\n")
if (has_base && n_stage >= 2) {
  fixed_h3 <- "asymmetry ~ alignment_stage + dimension"
  if (n_lang >= 2) fixed_h3 <- paste(fixed_h3, "+ language")
  f_h3 <- sprintf("%s + %s", fixed_h3, re_str)
  m3 <- safe_fit(f_h3)
  if (!is.null(m3)) {
    co <- as.data.frame(summary(m3)$coefficients)
    rows <- grepl("^alignment_stage", rownames(co))
    h3 <- co[rows, , drop = FALSE]
    print(round(h3, 4))
    write.csv(cbind(term = rownames(h3), h3),
              file.path(results_dir, "lme4_H3.csv"), row.names = FALSE)
    cat("  注意：这是观测关联（不同对齐阶段模型间的差异），不能解读为对齐过程的因果效应。\n")
  }
} else {
  cat("  跳过：数据中没有同时包含 base 与 instruct（NIM 通常无 base 权重）。\n")
  cat("  —— 此时论文正文主线为 H1 + H2 + H5；H3 作为探索性/未来工作。\n")
}

cat("\n[lme4] 完成。结果写入 data/results/lme4_H1.csv, lme4_H2.csv, lme4_H3.csv\n")
