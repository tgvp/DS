def cramers_v(x, y):
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    table = pd.crosstab(pair["x"], pair["y"])
    n = table.to_numpy().sum()
    r, k = table.shape

    if n == 0 or min(r, k) < 2:
        return np.nan

    chi2_value = chi2_contingency(table)[0]
    return np.sqrt(chi2_value / (n * min(r - 1, k - 1)))

cat_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()

v_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

for col1 in cat_cols:
    for col2 in cat_cols:
        v_matrix.loc[col1, col2] = (
            1.0 if col1 == col2 else cramers_v(df[col1], df[col2])
        )

display(v_matrix.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(v_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=0, vmax=1, square=True)
plt.title("Cramér's V — categorical associations")
plt.tight_layout()
plt.show()

# Rank unique pairs, excluding the diagonal
upper = v_matrix.where(np.triu(np.ones(v_matrix.shape), k=1).astype(bool))
pairs = upper.stack().sort_values(ascending=False)

print("Strongest associations:")
display(pairs.head(5).to_frame("Cramer's V"))

print("Weakest associations:")
display(pairs.tail(5).to_frame("Cramer's V"))
