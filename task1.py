import pandas as pd

# Load the raw dataset
data = pd.read_csv("raw_sales_data.csv")

# Show first 5 rows
print("First 5 rows:")
print(data.head())

# Show number of rows and columns
print("\nShape of dataset:")
print(data.shape)

# Show information about columns
print("\nDataset information:")
data.info()

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Show rows containing missing values
print("\nRows with missing values:")
print(data[data.isnull().any(axis=1)])

# Handle missing values
data["Quantity"] = data["Quantity"].fillna(data["Quantity"].median())
data["Price"] = data["Price"].fillna(data["Price"].median())
data["Customer_City"] = data["Customer_City"].fillna(data["Customer_City"].mode()[0])

# Check missing values after cleaning
print("\nMissing values after cleaning:")
print(data.isnull().sum())

# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(data.duplicated().sum())

# Display duplicate rows
print("\nDuplicate rows:")
print(data[data.duplicated()])
# Remove duplicate rows
data = data.drop_duplicates()

print("\nDuplicate rows after cleaning:")
print(data.duplicated().sum())

print("\nNumber of rows after removing duplicates:")
print(len(data))

# Detect outliers using IQR method

Q1 = data["Quantity"].quantile(0.25)
Q3 = data["Quantity"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\nLower limit:", lower_limit)
print("Upper limit:", upper_limit)

# Find outliers
outliers = data[
    (data["Quantity"] < lower_limit) |
    (data["Quantity"] > upper_limit)
]

print("\nQuantity outliers:")
print(outliers)

# Handle the extreme outlier
data["Quantity"] = data["Quantity"].clip(upper=25)

print("\nQuantity after handling outlier:")
print(data["Quantity"])

# Create Total Sales column
data["Total_Sales"] = data["Quantity"] * data["Price"]

print("\nDataset with Total Sales:")
print(data)

# Convert Date column to datetime
data["Date"] = pd.to_datetime(data["Date"])

print("\nDate column after conversion:")
print(data["Date"].head())

# Visualization 1: Total Sales by Category

import matplotlib.pyplot as plt
import seaborn as sns

category_sales = data.groupby("Category")["Total_Sales"].sum()

print("\nTotal Sales by Category:")
print(category_sales)

plt.figure(figsize=(8, 5))
sns.barplot(x=category_sales.index, y=category_sales.values)

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")

plt.tight_layout()

# Save the chart
plt.savefig("visualizations/total_sales_by_category.png")

plt.show()

# Visualization 2: Total Sales by City

city_sales = data.groupby("Customer_City")["Total_Sales"].sum()

print("\nTotal Sales by City:")
print(city_sales)

plt.figure(figsize=(8, 5))
sns.barplot(x=city_sales.index, y=city_sales.values)

plt.title("Total Sales by City")
plt.xlabel("Customer City")
plt.ylabel("Total Sales")

plt.tight_layout()

plt.savefig("visualizations/total_sales_by_city.png")

plt.show()

# Visualization 3: Sales Trend by Date

daily_sales = data.groupby("Date")["Total_Sales"].sum()

print("\nDaily Sales:")
print(daily_sales)

plt.figure(figsize=(10, 5))
plt.plot(daily_sales.index, daily_sales.values, marker="o")

plt.title("Sales Trend by Date")
plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)
plt.tight_layout()

# Save the chart
plt.savefig("visualizations/sales_trend_by_date.png")

plt.show()
# Save the cleaned dataset

data.to_csv("cleaned_sales_data.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Key Insights

print("\n===== KEY INSIGHTS =====")

# Total sales
print("\nTotal Sales:")
print(data["Total_Sales"].sum())

# Best-selling category by sales
best_category = data.groupby("Category")["Total_Sales"].sum().idxmax()
print("\nBest Category:")
print(best_category)

# Best city by sales
best_city = data.groupby("Customer_City")["Total_Sales"].sum().idxmax()
print("\nBest City:")
print(best_city)

# Highest selling product
best_product = data.groupby("Product")["Total_Sales"].sum().idxmax()
print("\nBest Product:")
print(best_product)

# Final data quality check

print("\n===== FINAL DATA QUALITY CHECK =====")

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:")
print(data.duplicated().sum())

print("\nFinal number of rows and columns:")
print(data.shape)