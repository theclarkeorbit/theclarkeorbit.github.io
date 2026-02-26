---
title: "A causal workflow in R with coupon marketing data"
date: 2025-12-04
output:
  html_document:
    df_print: paged
---



This post is code for the demo associated with a talk on causal inference presented at [Fifth Elephant Winter 2025](https://hasgeek.com/fifthelephant/2025-winter/schedule). The video of the talk will be available soon, and I'll link to it here.



We walk through a complete causal inference workflow using real marketing data from the [AmExpert 2019 Kaggle competition](https://www.kaggle.com/datasets/vasudeva009/predicting-coupon-redemption). The dataset contains transaction records, coupon assignments, and customer demographics from a retail store that ran 18 marketing campaigns. Our goal is to estimate the causal effect of receiving specific coupon types on customer spending during campaign periods.

Our analysis follows the methodology of [Langen & Huber (2023)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0278937), who applied causal machine learning to evaluate coupon effectiveness. Their key finding: **drugstore coupons have a significant positive effect on sales**, while **ready-to-eat food coupons show no significant effect**. We replicate this finding using a simplified classical approach, demonstrating how DAG-based identification translates directly into a regression specification.

### The Causal Structure

We model the coupon-sales relationship with the following DAG:


``` r
# Simple visualization for HTML output
library(ggdag)
dag <- dagify(
  Y ~ X + Z + C,
  X ~ Z + C,
  C ~ Z,
  exposure = "X",
  outcome = "Y",
  labels = c(Y = "Sales (Y)", X = "Focal Coupon (X)",
             Z = "Propensity (Z)", C = "Other Coupons (C)")
)
ggdag(dag, text = FALSE, use_labels = "label") +
  theme_dag_blank() +
  labs(title = "Coupon Campaign DAG")
```

<div class="figure" style="text-align: center">
<img src="/figures/coupon_workflow_reorganized/dag-visual-1.png" alt="center"  />
<p class="caption">center</p>
</div>

**Variables:**

- **Z (Propensity)**: Pre-campaign average daily spending—a scalar index of how likely a customer is to buy. High-Z customers are more likely to receive coupons (retailer targeting) *and* more likely to spend (baseline behavior).
- **X (Focal Coupon)**: Binary indicator for whether the customer received a coupon of the category we're studying (e.g., drugstore).
- **C (Other Coupons)**: Binary indicator for whether the customer received coupons from *other* categories in the same campaign.
- **Y (Sales)**: Average daily expenditures during the campaign period.


## Data Preparation

The dataset consists of six CSV files: `train.csv` (coupon assignments), `campaign_data.csv` (campaign dates), `customer_transaction_data.csv` (purchase records), `item_data.csv` (product categories), `coupon_item_mapping.csv` (coupon-to-product mappings), and `customer_demographics.csv`. We construct a customer × campaign panel where each row contains the propensity index Z, treatment indicators X and C, and outcome Y.

Following [Langen & Huber (2023)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0278937), we compare each coupon category against customers who received **no coupons at all** (not "no coupon of this type"). This ensures the control group is uncontaminated by other coupon effects. The transaction data contains extreme outliers (wholesale/B2B accounts with transactions in the millions), which we filter at the 95th percentile to focus on retail customers.















