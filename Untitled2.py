#!/usr/bin/env python
# coding: utf-8

# In[7]:


"""
===============================================================================
RETAIL PRODUCT ANALYTICS PROJECT (FULL, DETAILED, PORTFOLIO-READY VERSION)
===============================================================================

This project performs a complete end-to-end retail product analysis workflow
based on the dataset: "Message Group - Product.csv".

This script is written for:
- Data Analyst Portfolio Projects
- Job Interviews
- GitHub Repositories
- Resume Projects
- Power BI / Tableau Dashboards

WHAT THIS PROJECT COVERS:
------------------------------------
1. Data Loading & Raw Data Validation
2. Data Cleaning & Standardization
3. Feature Engineering
4. Aggregation & KPI Creation
5. Visualizations (with numbers/labels)
6. Deep Business Insights
7. Export Cleaned Dataset for Dashboarding

Business Purpose:
------------------------------
Retail businesses rely on pricing and discount analytics to drive:
- Pricing strategy
- Margin optimization
- Inventory management
- Product performance evaluation
- Category-level profitability

This script replicates exactly what a real Data Analyst does in retail analytics.
===============================================================================
"""

# =============================================================================
# 1. IMPORT NECESSARY LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Visualization styling
plt.style.use("seaborn-v0_8")
sns.set_palette("viridis")

# For readable outputs
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


# =============================================================================
# 2. LOAD RAW DATA
# =============================================================================

file_path = r"C:\Users\mfgdiags1\Downloads\Message Group - Product.csv"

# Load dataset into memory
data_raw = pd.read_csv(file_path)

print("\n================ RAW DATA OVERVIEW ================")
print("Shape (rows, columns):", data_raw.shape)
print("\nColumns:", data_raw.columns.to_list())
print("\nMissing values per column:\n", data_raw.isnull().sum())
print("\nSample rows:\n", data_raw.head(10))


# =============================================================================
# 3. DATA CLEANING & STANDARDIZATION
# =============================================================================

"""
WHY CLEAN?
- MRP contains strings/commas (e.g., '3,900')
- Discount has strings (e.g., "20% off")
- Currency contains inconsistent spacing
- SellPrice might contain non-numeric values

Steps:
1. Convert MRP to numeric
2. Convert SellPrice to numeric
3. Extract numeric discount %
4. Standardize currency text
"""

products = data_raw.copy()

# 3.1 Clean MRP
products["MRP_clean"] = (
    products["MRP"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"([0-9.]+)")
)
products["MRP_clean"] = pd.to_numeric(products["MRP_clean"], errors="coerce")

# 3.2 Clean Sell Price
products["SellPrice"] = pd.to_numeric(products["SellPrice"], errors="coerce")

# 3.3 Extract discount %
products["Discount_pct"] = (
    products["Discount"]
    .astype(str)
    .str.extract(r"([0-9.]+)")
    .astype(float)
)

# 3.4 Clean currency
products["Currancy"] = products["Currancy"].astype(str).strip()

print("\n================ CLEANED DATA PREVIEW ================")
print(products[["MRP", "MRP_clean", "SellPrice", "Discount", "Discount_pct"]].head())


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

"""
Feature Engineering transforms basic columns into meaningful business variables.

We Create:
1. Discount_value  → (MRP - SellPrice)
2. Discount_ratio  → (Discount value / MRP)
3. Premium flag    → Top 25% priced products
4. Revenue         → SellPrice (proxy since no quantity)
"""

# Discount value in currency
products["Discount_value"] = products["MRP_clean"] - products["SellPrice"]

# Discount ratio
products["Discount_ratio"] = products["Discount_value"] / products["MRP_clean"]
products.loc[~np.isfinite(products["Discount_ratio"]), "Discount_ratio"] = np.nan

# Revenue proxy
products["Revenue"] = products["SellPrice"]

# Premium threshold (top 25% MRP)
premium_cutoff = products["MRP_clean"].quantile(0.75)
products["Is_premium"] = (products["MRP_clean"] >= premium_cutoff).astype(int)

print("\n================ FEATURE ENGINEERING PREVIEW ================")
print(products[["MRP_clean", "SellPrice", "Discount_value", "Discount_ratio", "Is_premium"]].head())


# =============================================================================
# 5. CATEGORY-LEVEL & BRAND-LEVEL ANALYSIS
# =============================================================================

"""
We calculate:
- Total Revenue per Category / Brand
- Average MRP and Sell Prices
- Average Discount %
- Premium share

These KPIs are crucial for business decisions.
"""

# CATEGORY ANALYSIS
category_agg = (
    products.groupby("Category")
    .agg(
        Products=("Product ID", "nunique"),
        Total_Revenue=("Revenue", "sum"),
        Avg_MRP=("MRP_clean", "mean"),
        Avg_SellPrice=("SellPrice", "mean"),
        Avg_Discount_pct=("Discount_pct", "mean"),
        Premium_share=("Is_premium", "mean"),
    )
    .sort_values("Total_Revenue", ascending=False)
)

print("\n================ CATEGORY KPIs ================")
print(category_agg.head(10))


# BRAND ANALYSIS
brand_agg = (
    products.groupby("BrandName")
    .agg(
        Products=("Product ID", "nunique"),
        Total_Revenue=("Revenue", "sum"),
        Avg_MRP=("MRP_clean", "mean"),
        Avg_Discount_pct=("Discount_pct", "mean"),
    )
    .sort_values("Total_Revenue", ascending=False)
)

print("\n================ BRAND KPIs ================")
print(brand_agg.head(10))


# =============================================================================
# 6. VISUALIZATIONS (WITH NUMBERS)
# =============================================================================

print("\nGenerating visualizations...")

# 6.1 Top Categories by Revenue
plt.figure(figsize=(10, 6))
top_cat = category_agg.reset_index().head(10)
ax = sns.barplot(data=top_cat, x="Total_Revenue", y="Category")
plt.title("Top 10 Categories by Revenue")

# Add labels
for i, v in enumerate(top_cat["Total_Revenue"]):
    ax.text(v + 5, i, f"{v:,.0f}", va="center")

plt.tight_layout()
plt.show()


# 6.2 Top Brands by Revenue
plt.figure(figsize=(10, 6))
top_brand = brand_agg.reset_index().head(10)
ax = sns.barplot(data=top_brand, x="Total_Revenue", y="BrandName")
plt.title("Top 10 Brands by Revenue")

for i, v in enumerate(top_brand["Total_Revenue"]):
    ax.text(v + 5, i, f"{v:,.0f}", va="center")

plt.tight_layout()
plt.show()


# 6.3 Discount % Distribution
plt.figure(figsize=(8, 5))
ax = sns.histplot(products["Discount_pct"], bins=20, kde=True)
plt.title("Distribution of Discount Percentage")

# Add bin labels
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(p.get_x() + p.get_width() / 2, height + 1, int(height), ha="center")

plt.tight_layout()
plt.show()


# 6.4 MRP vs Discount Ratio
plt.figure(figsize=(8, 5))
sns.scatterplot(data=products, x="MRP_clean", y="Discount_ratio")
plt.title("MRP vs Discount Ratio")

# Summary box
plt.annotate(
    f"Mean Ratio: {products['Discount_ratio'].mean():.2f}\n"
    f"Max Ratio: {products['Discount_ratio'].max():.2f}\n"
    f"Min Ratio: {products['Discount_ratio'].min():.2f}",
    xy=(0.7, 0.8),
    xycoords="axes fraction",
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="black")
)

plt.tight_layout()
plt.show()


# =============================================================================
# 7. BUSINESS INSIGHTS
# =============================================================================

print("\n================ BUSINESS INSIGHTS ================")

# Insight 1: Highest Discount Category
highest_discount = category_agg["Avg_Discount_pct"].idxmax()
print(f"1. Category with highest discounting: {highest_discount}")

# Insight 2: Category with most premium items
premium_category = category_agg["Premium_share"].idxmax()
print(f"2. Category with most premium products: {premium_category}")

# Insight 3: Revenue Leader
top_revenue = category_agg["Total_Revenue"].idxmax()
print(f"3. Highest revenue category: {top_revenue}")

print("\nAdditional Insights:")
print("- High discount categories likely represent overstock or low demand.")
print("- Premium products contribute more margin but appear in fewer units.")
print("- Categories with limited size variety may have lower conversion.")
print("- Heavy discount ratios help identify overpriced items.")


# =============================================================================
# 8. EXPORT CLEANED DATA
# =============================================================================

output_path = r"C:\Users\mfgdiags1\Downloads\Message_Group_Product_cleaned.csv"
products.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved to: {output_path}")
print("\n================ PROJECT COMPLETE ================")


# In[ ]:




