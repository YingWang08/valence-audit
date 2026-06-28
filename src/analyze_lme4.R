#!/usr/bin/env Rscript
# =====================================================================
# analyze_lme4.R —— 论文级最终推断：交叉随机效应（model + frame 同时为随机效应）
#
# 用法：
#   Rscript analyze_lme4.R              # 默认主指标 rating
#   Rscript analyze_lme4.R text_m1      # 复核：自由文本 M1
#
# 依赖：install.packages(c("lme4","lmerTest"))
#   —— 读取 measure 阶段导出的 data/measured/items.csv，【无需安装 arrow】。
#
# 模型（已对齐最终数据）：
#   asymmetry ~ dimension + language + (1 | model) + (1 | frame)
#   · dimension 是固定效应 → 不再加 (1|dimension)（会共线）。
#   · frame = format|language|template（同一句式的多条观测不独立，心理语言学 item 随机效应）。
#   · 【刻意不含 alignment_stage】：最终数据里 frontier 阶段只有 1 个模型
#     （nemotron-mini-4b），把 alignment_stage 当固定效应会与 (1|model) 近乎共线/奇异，
#     且无法支撑对齐阶段层面的推断。对齐相关问题在论文里降为 future work，由 H5 描述性承载。
#
# 输出：data/results/lme4_fixed_effects.csv（固定效应表）
#       data/results/lme4_variance_components.csv（随机效应方差成分）
#       控制台打印维度联合检验（Type-II ANOVA, Satterthwaite）+ BH 校正 + 方向核对。
# =====================================================================

suppressMessages({
  ok <- requireNamespace("lme4", quietly = TRUE) &&
        requireNamespace("lmerTest", quietly = TRUE)
})
if (!ok) {
  stop("缺少依赖。请先在 R 里运行： install.packages(c(\"lme4\",\"lmerTest\"))")
}
library(lme4)
library(lmerTest)

args <- commandArgs(trailingOnly = TRUE)
measure_sel <- if (length(args) >= 1) args[1] else "rating"

# ---- 定位 items.csv（相对脚本所在目录的 data/measured/）----
get_script_dir <- function() {
  a <- commandArgs(FALSE)
  m <- grep("^--file=", a, value = TRUE)
  if (length(m)) return(dirname(normalizePath(sub("^--file=", "", m[1]))))
  return(getwd())
}
root <- get_script_dir()
csv_path <- file.path(root, "data", "measured", "items.csv")
if (!file.exists(csv_path)) {
  # 兜底：当前工作目录
  alt <- file.path(getwd(), "data", "measured", "items.csv")
  if (file.exists(alt)) csv_path <- alt else
    stop(paste0("找不到 ", csv_path, "。请先运行 python -m src.measure 生成 items.csv。"))
}
cat(sprintf("[lme4] 读取 %s\n", csv_path))
df <- read.csv(csv_path, stringsAsFactors = FALSE, fileEncoding = "UTF-8")

# ---- 选主指标子集 ----
d <- df[df$measure == measure_sel, ]
if (nrow(d) == 0) stop(sprintf("measure=%s 没有数据。可用: %s",
                               measure_sel, paste(unique(df$measure), collapse=", ")))
cat(sprintf("[lme4] 主指标 = %s，观测数 = %d\n", measure_sel, nrow(d)))

# 因子化
d$dimension <- factor(d$dimension)
d$language  <- factor(d$language)
d$model     <- factor(d$model)
if (!("frame" %in% names(d))) {
  d$frame <- paste(d$format, d$language, d$template, sep = "|")
}
d$frame <- factor(d$frame)

n_models <- nlevels(d$model)
n_frames <- nlevels(d$frame)
cat(sprintf("[lme4] 随机效应层级： model=%d 个， frame=%d 个\n", n_models, n_frames))

# ---- 拟合交叉随机效应 ----
# 若 model 数太少(<3)，(1|model) 方差难估，仍拟合但给出警告
form <- asymmetry ~ dimension + language + (1 | model) + (1 | frame)
cat("[lme4] 拟合： asymmetry ~ dimension + language + (1|model) + (1|frame)\n")
m <- tryCatch(
  lmerTest::lmer(form, data = d,
                 control = lmerControl(optimizer = "bobyqa",
                                       check.conv.singular = .makeCC("warning", tol = 1e-4))),
  error = function(e) { cat("拟合报错：", conditionMessage(e), "\n"); NULL }
)
if (is.null(m)) {
  # 退路：去掉 frame 随机效应再试
  cat("[lme4] 改用简化模型： asymmetry ~ dimension + language + (1|model)\n")
  m <- lmerTest::lmer(asymmetry ~ dimension + language + (1 | model), data = d,
                      control = lmerControl(optimizer = "bobyqa"))
}

cat("\n================= 固定效应（lmerTest, Satterthwaite df）=================\n")
co <- summary(m)$coefficients
print(round(co, 4))

# 维度的联合显著性（Type-II ANOVA）
cat("\n================= 维度联合检验（Type-II ANOVA）=================\n")
av <- tryCatch(anova(m, type = 2), error = function(e) NULL)
if (!is.null(av)) print(round(av, 4))

# ---- 各维度相对基准的 BH 校正 + 方向核对 ----
# 提取 dimension 各水平系数（相对参照水平），做 BH
cat("\n================= 各维度系数 BH 校正 + 期望方向核对 =================\n")
expected <- c(creativity="+", reliability="+", productivity="+", decision="+",
              moral="-", emotion="-", trust="-", worth="-")
rn <- rownames(co)
dim_rows <- grep("^dimension", rn)
if (length(dim_rows) > 0) {
  est <- co[dim_rows, "Estimate"]
  pv  <- co[dim_rows, grep("Pr", colnames(co))]
  nm  <- sub("^dimension", "", rn[dim_rows])
  p_bh <- p.adjust(pv, method = "BH")
  ref_level <- levels(d$dimension)[1]
  out <- data.frame(dimension = nm,
                    estimate_vs_ref = round(est, 4),
                    p_raw = signif(pv, 3),
                    p_BH  = signif(p_bh, 3),
                    expected = expected[nm],
                    stringsAsFactors = FALSE)
  cat(sprintf("（注：系数为相对参照维度 '%s' 的差异；正负方向需结合参照维度自身的截距解读）\n", ref_level))
  print(out, row.names = FALSE)
  results_dir <- file.path(root, "data", "results")
  if (!dir.exists(results_dir)) results_dir <- file.path(getwd(), "data", "results")
  write.csv(out, file.path(results_dir, "lme4_fixed_effects.csv"), row.names = FALSE)
}

# ---- 方差成分 ----
cat("\n================= 随机效应方差成分 =================\n")
vc <- as.data.frame(VarCorr(m))
vc_out <- vc[, c("grp", "vcov", "sdcor")]
names(vc_out) <- c("group", "variance", "std_dev")
total_var <- sum(vc_out$variance)
vc_out$pct_of_random <- round(100 * vc_out$variance / total_var, 1)
print(vc_out, row.names = FALSE)

results_dir <- file.path(root, "data", "results")
if (!dir.exists(results_dir)) results_dir <- file.path(getwd(), "data", "results")
write.csv(vc_out, file.path(results_dir, "lme4_variance_components.csv"), row.names = FALSE)

cat(sprintf("\n[lme4] 完成。固定效应与方差成分已写到 %s\n", results_dir))
cat("[lme4] 论文 Methods：headline 用 Python cluster bootstrap；本 lme4 结果作 Supplementary 确认性互证。\n")
