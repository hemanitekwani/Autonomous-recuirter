# Sample data

All names, emails and companies are fictitious (`example.com` / `example.invalid`).

| File | Purpose | Expected result for "Machine Learning Engineer" (mock mode) |
|---|---|---|
| `jobs.json` | Two demo jobs (ML Engineer, Java Backend) | - |
| `resumes/priya_sharma.txt` | Python, ML, torch (alias), 6 yrs, overlapping-free history | **ELIGIBLE** |
| `resumes/rahul_verma.txt` | Java developer, 1 yr | **NOT_ELIGIBLE** (experience + skills) |
| `resumes/anita_rao.txt` | Year-only dates for main role, internship excluded | **REVIEW_REQUIRED** (experience unknown) |
| `resumes/injection_attempt.txt` | Contains "ignore all previous instructions..." | **NOT_ELIGIBLE** + prompt-injection review flag |

`python -m scripts.seed_demo` turns the `.txt` sources into real PDF/DOCX files in
`resumes/generated/` - upload those through the Candidate Portal.
`python -m scripts.seed_demo --submit` submits them all automatically.

Experience numbers depend on the current month (current roles end "today").
