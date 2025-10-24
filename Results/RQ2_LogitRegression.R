# Load required libraries
library(readr)
library(dplyr)
library(stringr)
library(purrr)
library(tidyr)
library(lme4)
library(car)
library(MuMIn)
library(lmerTest)  # For p-values in mixed models


# Set working directory
setwd("~/Desktop/ReplicationPackages/ReplicationPerturbationAny/PerturbationsReplication/Results")
print(getwd())

# Load the CSV file
df <- read_csv("Perturbation_Metrics_Results.csv")

# Define mutation types and models
BaseTypes <- c("IfElseSwap",
               "DeadException", "DeadVariable", "TryNcatchWrapper", "IndependentLineSwap",
               "ReturnViaVariable", "defUseBreak", "RandomNames", "ShuffleNames")

Models <- c("T5", "LLaMaLoRa", "Llama3.3_70B", "gpt-3.5-turbo", "DeepSeek-V3")

# Construct long-format data frame
long_data <- map_dfr(BaseTypes, function(base) {
  map_dfr(Models, function(model) {
    exm_col <- paste0(model, "_", base, "_Org_EXM")
    metrics <- c("org_pert_tokenEdit", "pert_fix_tokenEdit", "Pert_length", "pos", "pert_dist")
    metric_cols <- paste0(base, "_", metrics)
    
    if (!all(c(exm_col, metric_cols) %in% names(df))) return(NULL)
    
    df %>%
      select(all_of(c(exm_col, metric_cols))) %>%
      rename(
        EXM = !!exm_col,
        org_pert_tokenEdit = !!paste0(base, "_org_pert_tokenEdit"),
        pert_fix_tokenEdit = !!paste0(base, "_pert_fix_tokenEdit"),
        Pert_length = !!paste0(base, "_Pert_length"),
        Pos = !!paste0(base, "_pos"),
        pert_dist = !!paste0(base, "_pert_dist")
      ) %>%
      mutate(
        EXM = as.numeric(EXM),
        LLM_Model = model,
        Perturbation_type = base
      )
  })
})

# Clean and prepare data
long_data <- long_data %>%
  drop_na(EXM, Pos, pert_dist,  pert_fix_tokenEdit, Pert_length, org_pert_tokenEdit) %>%
  mutate(
    Success = ifelse(EXM >= 1.0, 1, 0),
    Pos = factor(Pos, levels = c("Before", "After", "Inside", "Overlap-Both",
                                 "Overlap-Before", "Overlap-After", "Surrounding")),
    Perturbation_type = factor(Perturbation_type),
    LLM_Model = factor(LLM_Model),
    pert_dist = scale(pert_dist),
    org_pert_tokenEdit = scale(org_pert_tokenEdit),
    pert_fix_tokenEdit = scale(pert_fix_tokenEdit),
    Pert_length = scale(Pert_length)
  )

# Optional: Check multicollinearity
vif_model <- lm(EXM ~ pert_dist + org_pert_tokenEdit + pert_fix_tokenEdit + Pert_length, data = long_data)
print("VIF values:")
print(vif(vif_model))

# Optional: Spearman correlation
spearman_corr <- cor(select(long_data, pert_dist, org_pert_tokenEdit, pert_fix_tokenEdit, Pert_length),
                     method = "spearman", use = "complete.obs")
print("Spearman correlation matrix:")
print(round(spearman_corr, 2))

# Mixed-effects logistic regression model (with p-values via lmerTest)
global_logit_model <- glmer(
  Success ~ Pos + pert_dist + org_pert_tokenEdit + pert_fix_tokenEdit + Pert_length +
    (1 | Perturbation_type) + (1 | LLM_Model),
  data = long_data,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

# Display full model summary with estimates and p-values
print(summary(global_logit_model))

# ============================================================
# Compute Odds Ratios (OR) and 95% Confidence Intervals
# ============================================================
or_table <- exp(cbind(
  OR = fixef(global_logit_model),
  confint(global_logit_model, parm = "beta_", method = "Wald")
))

print("Odds Ratios (with 95% Confidence Intervals):")
print(round(or_table, 3))

# ============================================================
# Get R-squared values
# ============================================================
r2_values <- suppressWarnings(r.squaredGLMM(global_logit_model))
print("Pseudo R-squared values:")
print(r2_values)

