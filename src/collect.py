"""Phase 1 - Multi-source job data collection.

Contains (a) polite scrapers for the real sources and (b) a synthetic generator so the
whole pipeline can be run/tested without internet access.

IMPORTANT (legal/ethical): LinkedIn, Indeed, Naukri and Glassdoor restrict scraping in their
Terms of Service. For the real project use official APIs / public datasets (Kaggle LinkedIn
Job Postings, Glassdoor partner API, Naukri public dumps) or scrape only for academic use with
rate limiting and robots.txt compliance. Selectors below change often - treat them as templates.
"""
import random, time, re
import numpy as np, pandas as pd
from config import RAW, SEED
from skills_catalog import SKILLS, ROLES

UNIFIED_COLUMNS = ["job_id", "source", "country", "title", "company", "location", "description", "date_posted", "salary"]

# ----------------------------------------------------------------------------- scrapers
HEADERS = {"User-Agent": "Mozilla/5.0 (academic-research-bot; contact: you@college.edu)"}

def scrape_static(url_template, pages, parse_fn, delay=(2, 5)):
    """BeautifulSoup scraper for static pages (e.g. Naukri/Indeed listing pages that serve HTML)."""
    import requests
    from bs4 import BeautifulSoup
    rows = []
    for p in range(1, pages + 1):
        r = requests.get(url_template.format(page=p), headers=HEADERS, timeout=20)
        if r.status_code != 200:
            break
        rows += parse_fn(BeautifulSoup(r.text, "lxml"))
        time.sleep(random.uniform(*delay))
    return rows

def parse_naukri(soup):
    out = []
    for card in soup.select("div.srp-jobtuple-wrapper"):
        g = lambda sel: (card.select_one(sel).get_text(strip=True) if card.select_one(sel) else None)
        out.append(dict(source="naukri", title=g("a.title"), company=g("a.comp-name"),
                        location=g("span.locWdth"), description=g("div.job-desc"),
                        date_posted=g("span.job-post-day"), salary=g("span.sal-wrap")))
    return out

def scrape_dynamic_selenium(url, card_css, field_css, scrolls=5):
    """Selenium scraper for JavaScript-rendered pages (LinkedIn public search, Indeed)."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    opts = webdriver.ChromeOptions(); opts.add_argument("--headless=new")
    d = webdriver.Chrome(options=opts); d.get(url); rows = []
    for _ in range(scrolls):
        d.execute_script("window.scrollTo(0, document.body.scrollHeight);"); time.sleep(random.uniform(2, 4))
    for c in d.find_elements(By.CSS_SELECTOR, card_css):
        row = {}
        for k, sel in field_css.items():
            try: row[k] = c.find_element(By.CSS_SELECTOR, sel).text
            except Exception: row[k] = None
        rows.append(row)
    d.quit(); return rows

SCRAPY_SPIDER = '''
# save as jobs_spider.py ; run: scrapy runspider jobs_spider.py -o data/raw/scrapy_jobs.jl
import scrapy
class JobsSpider(scrapy.Spider):
    name = "jobs"
    custom_settings = {"DOWNLOAD_DELAY": 3, "ROBOTSTXT_OBEY": True, "AUTOTHROTTLE_ENABLED": True}
    start_urls = ["https://example-job-board.com/jobs?page=1"]
    def parse(self, response):
        for c in response.css("div.job-card"):
            yield {"title": c.css("h2::text").get(), "company": c.css(".company::text").get(),
                   "location": c.css(".loc::text").get(), "description": " ".join(c.css(".desc ::text").getall()),
                   "date_posted": c.css("time::attr(datetime)").get(), "source": "scrapy"}
        nxt = response.css("a.next::attr(href)").get()
        if nxt: yield response.follow(nxt, self.parse)
'''

def fetch_glassdoor_api(partner_id, key, query, pages=3):
    """Glassdoor partner API (requires approved partner credentials)."""
    import requests
    rows = []
    for p in range(1, pages + 1):
        r = requests.get("https://api.glassdoor.com/api/api.htm",
            params=dict(v=1, format="json", t_p=partner_id, t_k=key, action="jobs-stats", q=query, pn=p),
            headers=HEADERS, timeout=20)
        rows += [dict(source="glassdoor", **j) for j in r.json().get("response", {}).get("jobs", [])]
    return rows

def unify(frames):
    """Merge per-source frames while preserving source and country."""
    df = pd.concat(frames, ignore_index=True)

    for c in UNIFIED_COLUMNS:
        if c not in df:
            df[c] = np.nan

    if df["job_id"].isna().all():
        df["job_id"] = np.arange(len(df))

    df["source"] = df["source"].fillna("unknown").replace("", "unknown")
    df["country"] = df["country"].fillna("Unknown").replace("", "Unknown")

    return df[UNIFIED_COLUMNS]


def load_real(csv_path, source="linkedin", colmap=None, country="India"):
    """Load real job data and preserve source/country metadata."""
    df = pd.read_csv(csv_path)

    if colmap:
        df = df.rename(columns=colmap)

    if "source" not in df.columns:
        df["source"] = source
    else:
        df["source"] = df["source"].fillna(source).replace("", source)

    if "country" not in df.columns:
        df["country"] = country
    else:
        df["country"] = df["country"].fillna(country).replace("", country)

    return unify([df])


# ----------------------------------------------------------------------------- synthetic data
# demand trajectory per skill: (base share, yearly growth, seasonality amplitude, shock month or None)
def _skill_profiles(rng):
    emerging = {"Generative AI": (0.002, 2.8), "LLM": (0.002, 3.0), "LangChain": (0.001, 3.2),
                "Prompt Engineering": (0.002, 2.5), "MLOps": (0.01, 0.6), "Kubernetes": (0.03, 0.35),
                "Terraform": (0.02, 0.4), "FastAPI": (0.01, 0.6), "Snowflake": (0.02, 0.35),
                "Rust": (0.008, 0.45), "TypeScript": (0.05, 0.25), "Power BI": (0.04, 0.2),
                "Cybersecurity": (0.04, 0.2), "PyTorch": (0.03, 0.35), "Deep Learning": (0.06, 0.15),
                "AWS": (0.12, 0.1), "Azure": (0.08, 0.15), "Docker": (0.08, 0.1), "Neo4j": (0.004, 0.3),
                "Airflow": (0.02, 0.25), "Kafka": (0.03, 0.15), "Go": (0.015, 0.3), "NLP": (0.03, 0.2)}
    declining = {"jQuery": (0.05, -0.30), "Hadoop": (0.03, -0.35), "PHP": (0.05, -0.12),
                 "Manual Testing": (0.04, -0.15), "Selenium": (0.04, -0.05), "Angular": (0.05, -0.08),
                 "Blockchain": (0.015, -0.2)}
    prof = {}
    for s in SKILLS:
        base, g = emerging.get(s) or declining.get(s) or (rng.uniform(0.02, 0.12), rng.uniform(-0.03, 0.06))
        prof[s] = dict(base=base, growth=g, season=rng.uniform(0.0, 0.18), phase=rng.uniform(0, 2 * np.pi))
    return prof

ROLE_SKILLS = {
 "Data Scientist": ["Python", "Machine Learning", "SQL", "Pandas", "Scikit-learn", "Deep Learning", "NLP", "R", "Generative AI"],
 "Machine Learning Engineer": ["Python", "Machine Learning", "PyTorch", "TensorFlow", "MLOps", "Docker", "AWS", "Deep Learning", "LLM"],
 "AI Engineer": ["Python", "Generative AI", "LLM", "LangChain", "Prompt Engineering", "PyTorch", "NLP", "FastAPI", "Docker"],
 "Data Analyst": ["SQL", "Excel", "Power BI", "Tableau", "Python", "Communication"],
 "Data Engineer": ["Python", "SQL", "Spark", "Airflow", "Kafka", "Snowflake", "AWS", "Hadoop", "Docker"],
 "Backend Developer": ["Java", "Python", "Spring Boot", "Django", "FastAPI", "SQL", "Docker", "Git", "Go", "MongoDB"],
 "Frontend Developer": ["JavaScript", "TypeScript", "React", "Angular", "jQuery", "Git"],
 "Full Stack Developer": ["JavaScript", "TypeScript", "React", "Node.js", "SQL", "MongoDB", "Docker", "Git", "PHP"],
 "DevOps Engineer": ["Docker", "Kubernetes", "Terraform", "CI/CD", "Linux", "AWS", "Azure", "Git", "Python"],
 "Cloud Architect": ["AWS", "Azure", "GCP", "Kubernetes", "Terraform", "Cybersecurity", "Linux"],
 "Security Analyst": ["Cybersecurity", "Linux", "Python", "AWS", "Communication"],
 "QA Engineer": ["Selenium", "Manual Testing", "Python", "Java", "CI/CD", "SQL"],
 "Business Analyst": ["Excel", "SQL", "Power BI", "Tableau", "Communication", "Leadership"],
}
COMPANIES = ["TCS", "Infosys", "Wipro", "HCLTech", "Accenture", "Google", "Microsoft", "Amazon", "Flipkart",
             "Zoho", "Deloitte", "Capgemini", "Cognizant", "Oracle", "IBM", "Swiggy", "Razorpay", "Freshworks",
             "Paytm", "Tech Mahindra", "Salesforce", "Adobe", "Nvidia", "Meesho", "PhonePe"]
LOCATIONS = ["Hyderabad", "Bengaluru", "Pune", "Chennai", "Mumbai", "Delhi NCR", "Remote"]
SOURCES = ["linkedin", "naukri", "indeed", "glassdoor"]

def generate_synthetic(start="2021-01", end="2026-10", per_month=420, seed=SEED):
    """Synthetic multi-source postings whose skill demand follows realistic growth/decline/seasonality.
    NOT real data - for testing the pipeline only."""
    rng = np.random.default_rng(seed); random.seed(seed)
    prof = _skill_profiles(random)
    months = pd.period_range(start, end, freq="M")
    rows, jid = [], 0
    for mi, m in enumerate(months):
        t = mi / 12.0
        # AI boom: more AI-ish roles over time
        role_w = np.array([1.0 + (0.12 * mi / 6 if r in ("AI Engineer",) else 0) +
                           (0.03 * mi / 6 if r in ("Machine Learning Engineer", "Data Engineer", "DevOps Engineer") else 0)
                           for r in ROLES]); role_w /= role_w.sum()
        n = int(per_month * (1 + 0.15 * np.sin(2 * np.pi * mi / 12 - 1) + rng.normal(0, 0.04)))
        for _ in range(n):
            role = rng.choice(ROLES, p=role_w)
            pool = ROLE_SKILLS[role]
            chosen = set()
            for s in pool:
                p = prof[s]
                level = p["base"] * 6 * ((1 + p["growth"]) ** t) * (1 + p["season"] * np.sin(2 * np.pi * mi / 12 + p["phase"]))
                if rng.random() < min(0.92, max(0.01, level + 0.25)): chosen.add(s)
            # cross-role "leak" so the graph is richer
            for s, p in prof.items():
                lv = p["base"] * 0.5 * ((1 + p["growth"]) ** t)
                if rng.random() < min(0.3, lv * 0.35): chosen.add(s)
            if not chosen: chosen.add(random.choice(pool))
            names = []
            for s in sorted(chosen):
                names.append(random.choice([s] + SKILLS[s][1][:1]) if rng.random() < 0.25 else s)
            random.shuffle(names)
            desc = (f"We are hiring a {role}. You will work with {', '.join(names[:-1]) + ' and ' + names[-1] if len(names) > 1 else names[0]}. "
                    f"{random.choice(['Strong problem solving is a must.', 'Experience with agile teams preferred.', 'Immediate joiners welcome.'])}")
            day = int(rng.integers(1, 28))
            rows.append(dict(job_id=jid, source=random.choice(SOURCES), title=role if rng.random() > 0.3 else f"Senior {role}",
                             company=random.choice(COMPANIES), location=random.choice(LOCATIONS),
                             description=desc, date_posted=f"{m.year}-{m.month:02d}-{day:02d}",
                             salary=int(rng.normal(900000 + 25000 * t * 12 / 12, 250000))))
            jid += 1
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_synthetic(); df.to_csv(RAW / "jobs_raw_synthetic.csv", index=False)
    print(df.shape); print(df.head())
