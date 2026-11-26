"""Phase 2 - Data cleaning (Pandas)."""
import re, html
import pandas as pd, numpy as np

def clean_text(t):
    if not isinstance(t, str): return ""
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)               # strip HTML
    t = re.sub(r"http\S+|www\.\S+", " ", t)      # urls
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalize_role(title):
    if not isinstance(title, str) or not title.strip():
        return "Other"

    t = title.lower()
    t = re.sub(
        r"^(senior|sr\.?|junior|jr\.?|lead|principal|associate)\s+",
        "",
        t
    )

    rules = [
        (r"data scientist|data science|datascientist", "Data Scientist"),
        (r"machine learning|ml engineer|deep learning engineer", "Machine Learning Engineer"),
        (r"data analyst|business intelligence|bi analyst", "Data Analyst"),
        (r"data engineer|big data engineer", "Data Engineer"),
        (r"artificial intelligence|ai engineer|gen.?ai|ai/ml|applied scientist", "AI Engineer"),
        (r"software engineer|software developer|software development|sde\b|python developer|python development|java developer", "Software Engineer"),
        (r"full.?stack", "Full Stack Developer"),
        (r"front.?end", "Frontend Developer"),
        (r"back.?end", "Backend Developer"),
        (r"web developer|web designer|web application developer|web / ui developer", "Web Developer"),
        (r"react developer", "Frontend Developer"),
        (r"devops|site reliability|sre", "DevOps Engineer"),
        (r"cloud architect|cloud engineer", "Cloud Engineer"),
        (r"security analyst|cybersecurity|information security", "Security Analyst"),
        (r"quality assurance|\bqa\b|test engineer|testing engineer|quality engineer", "QA Engineer"),
        (r"business analyst", "Business Analyst"),
        (r"product manager|product management|product owner|product innovation|product strategy|manager product", "Product Manager"),
        (r"project manager|process manager", "Project Manager"),
        (r"equity analyst|investment banking|trading analyst|equity research|equity capital market|capital markets|investment analyst|portfolio analyst|valuation analyst|real assets financing", "Investment Analyst"),
        (r"credit analyst|credit risk|credit research", "Credit Analyst"),
        (r"aml|kyc|risk analyst|risk management|valuation risk", "Risk and Compliance Analyst"),
        (r"financial analyst|finance analyst|analyst.*finance|financial planning|financial analysis|\bfp&a\b|finance manager", "Financial Analyst"),
        (r"human resources|human resource|\bhr\b|hr manager|hr generalist|talent acquisition", "HR Professional"),
        (r"content writer|copywriter|copy writer|content writing|copy editor|content strategist|content editor|features writer|writer.editor|sub editor", "Content Writer"),
        (r"social media|digital marketing|marketing|shopper marketing|brand marketing|retention marketing", "Marketing Specialist"),
        (r"mechanical|automobile engineer|vehicle dynamics|hvac|design engineer", "Mechanical Engineer"),
        (r"manufacturing engineer|plant simulation|biw process|biw design", "Manufacturing Engineer"),
        (r"aerospace engineer", "Aerospace Engineer"),
        (r"piping engineer", "Piping Engineer"),
        (r"sales|business development|account executive|account manager|relationship manager|channel manager|key account", "Sales Professional"),
        (r"financial systems|hfm|fccs", "Financial Systems Analyst"),
        (r"corporate development", "Business Development Manager"),
    ]

    for pattern, role in rules:
        if re.search(pattern, t):
            return role

    return "Other"


def clean_jobs(df):
    df = df.copy()
    n0 = len(df)
    df["title"] = df["title"].fillna("").map(clean_text)
    df["company"] = df["company"].fillna("Unknown").map(lambda s: clean_text(s).title())
    df["location"] = df["location"].fillna("Unknown").map(clean_text)
    df["description"] = df["description"].map(clean_text)
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    df = df.dropna(subset=["date_posted"])
    df = df[df["description"].str.len() > 30]
    # near-duplicate removal across sources (same title/company/month/desc)
    df["month"] = df["date_posted"].dt.to_period("M").dt.to_timestamp()
    key = df["title"].str.lower() + "|" + df["company"].str.lower() + "|" + df["description"].str.lower().str[:120] + "|" + df["month"].astype(str)
    df = df.loc[~key.duplicated()]
    # normalise role titles: strip seniority prefixes
    df["role"] = df["title"].apply(normalize_role)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df.loc[(df.salary < 1e5) | (df.salary > 5e7), "salary"] = np.nan
    print(f"[clean] {n0} -> {len(df)} rows after cleaning")
    return df.reset_index(drop=True)
