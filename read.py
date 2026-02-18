import pandas as pd

df = pd.read_json("technology_fact_checks.json")
print(df.columns)
df['complete_text'] = f"The claim of the article is " +  df['claim'] + "\n\n The rating of the article is " + df['rating'] + "\n\n" +df['article_text']


df.to_excel("snopes_fact_checks.xlsx", index=False)