import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("pokemon.csv")

# (1) - Generational Trends

# (1.1) - Number of Pokémon Introduced per Generation
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Generation", palette="pastel")
plt.title("Number of Pokémon Introduced per Generation")
plt.show()

stats = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Total"]

# (1.2) - Average total stats per generation
gen_avg = df.groupby("Generation")["Total"].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.lineplot(data=gen_avg, x="Generation", y="Total", marker="o")
plt.title("Average Total Stats per Generation")
plt.show()

# (1.3) - Average of each stat by generation
gen_stats = df.groupby("Generation")[stats].mean().reset_index()
gen_stats_melted = gen_stats.melt(id_vars="Generation", var_name="Stat", value_name="Average")
plt.figure(figsize=(10, 6))
sns.lineplot(data=gen_stats_melted, x="Generation", y="Average", hue="Stat", marker="o")
plt.title("Average of Each Stat by Generation")
plt.show()


# (2) - Type Analysis

# (2.1) - Single vs Dual
df["Single/Dual"] = df["Type 2"].apply(lambda x: "Single" if pd.isna(x) else "Dual")
sns.countplot(data=df, x="Single/Dual", palette="muted")
plt.title("Single vs Dual Type Pokémon")
plt.show()

# (2.2) - Most Common Type of Pokémon
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # define subplot grid

# Type 1
sns.countplot(data=df, x="Type 1", order=df["Type 1"].value_counts().index,
              palette="viridis", ax=axes[0])
axes[0].set_title("Most Common Pokémon Types (Type 1)")
axes[0].set_ylabel("Count")
axes[0].set_xlabel("Type 1")
axes[0].tick_params(axis="x", rotation=45)

# Type 2
sns.countplot(data=df, x="Type 2", order=df["Type 2"].value_counts().index,
              palette="plasma", ax=axes[1])
axes[1].set_title("Most Common Pokémon Types (Type 2)")
axes[1].set_ylabel("Count")
axes[1].set_xlabel("Type 2")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()


# (2.3) - Type Combo Diversity (Common vs Rare Dual Types)
df["Type 2"].fillna("None", inplace=True)
df["Type Combo"] = df["Type 1"] + " / " + df["Type 2"]

combo_counts = df["Type Combo"].value_counts().reset_index()
combo_counts.columns = ["Type Combo", "Count"]

plt.figure(figsize=(12,8))
sns.barplot(data=combo_counts.head(15), x="Count", y="Type Combo", palette="magma")
plt.title("Most Common Pokémon Type Combinations (Top 15)")
plt.xlabel("Count")
plt.ylabel("Type Combo")
plt.show()


# (3) - Statistical Analysis

# (3.1) - Pokemon Stats Distribution
stats = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Total"]

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, stat in enumerate(stats):
    axes[i].hist(df[stat], bins=15, edgecolor="black", color="skyblue")
    axes[i].set_title(f"Distribution of {stat}", fontsize=12)
    axes[i].set_xlabel(stat)
    axes[i].set_ylabel("Count")

fig.delaxes(axes[-1])  # remove empty subplot
plt.suptitle("Pokémon Stats Distribution", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# (4.4) - Legendary vs Non-Legendary: All Stats in One Grid
melted = df.melt(id_vars=["Legendary"], value_vars=stats, var_name="Stat", value_name="Value")

g = sns.FacetGrid(melted, col="Stat", col_wrap=3, hue="Legendary", sharex=False, sharey=False, height=4)
g.map(sns.kdeplot, "Value", fill=True, common_norm=False, alpha=0.4)
g.add_legend()
g.set_titles("{col_name}")
g.fig.suptitle("Legendary vs Non-Legendary Stats", y=1.02, fontsize=16)
plt.show()


# (3.2) - Boxplots of stats grouped by Type 1
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, stat in enumerate(stats):
    sns.boxplot(data=df, x="Type 1", y=stat, ax=axes[i])
    axes[i].set_title(f"{stat} by Type 1")
    axes[i].tick_params(axis="x", rotation=45)

fig.delaxes(axes[-1])
plt.suptitle("Boxplots of Pokémon Stats by Primary Type (Type 1)", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# (3.3) - Boxplots of stats grouped by Type 2
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, stat in enumerate(stats):
    sns.boxplot(data=df, x="Type 2", y=stat, ax=axes[i])
    axes[i].set_title(f"{stat} by Type 2")
    axes[i].tick_params(axis="x", rotation=45)

fig.delaxes(axes[-1])
plt.suptitle("Boxplots of Pokémon Stats by Secondary Type (Type 2)", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# (3.6) - Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[stats].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Between Pokémon Stats")
plt.show()


# (4) - Comparisons & Highlights

# (4.1) - Playstyle classification
df["Offense"] = df["Attack"] + df["Sp. Atk"]
df["Defense_Total"] = df["Defense"] + df["Sp. Def"]

def classify(row):
    if row["Offense"] > row["Defense_Total"] + 20:
        return "Offensive"
    elif row["Defense_Total"] > row["Offense"] + 20:
        return "Defensive"
    else:
        return "Balanced"

df["Playstyle"] = df.apply(classify, axis=1)
sns.countplot(data=df, x="Playstyle", palette="Set1")
plt.title("Pokémon Classification: Offensive vs Defensive vs Balanced")
plt.show()

# (4.2) - Top 10 by each stat
fig, axes = plt.subplots(2, 4, figsize=(22, 12))
axes = axes.flatten()

for i, stat in enumerate(stats):
    top10 = df.nlargest(10, stat)[["Name", stat]]
    sns.barplot(data=top10, x=stat, y="Name", palette="viridis", ax=axes[i])
    axes[i].set_title(f"Top 10 Pokémon by {stat}")
    axes[i].set_xlabel(stat)
    axes[i].set_ylabel("")

fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()

# (4.3) - Strongest Pokémon per Type
best_by_type = df.loc[df.groupby("Type 1")["Total"].idxmax()][["Type 1", "Name", "Total"]]
plt.figure(figsize=(12,6))
sns.barplot(data=best_by_type, x="Type 1", y="Total", palette="viridis",
            order=best_by_type.sort_values("Total", ascending=False)["Type 1"])
plt.xticks(rotation=45)
plt.title("Strongest Pokémon per Type (by Total Stats)")
plt.ylabel("Total Stats")
plt.show()


# (3.5) - Legendary by Type (Heatmap)
plt.figure(figsize=(10,6))
legendary_type = pd.crosstab(df["Type 1"], df["Legendary"])
sns.heatmap(legendary_type, annot=True, fmt="d", cmap="Blues")
plt.title("Legendary vs Non-Legendary by Type 1")
plt.show()