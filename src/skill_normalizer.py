"""Deterministic skill normalization against the approved taxonomy + alias dictionary.

Matching policy (documented, no ML involved):
1. Lower-case (casefold), trim, collapse whitespace, strip surrounding punctuation.
2. Exact canonical name  -> EXACT
3. Known alias           -> ALIAS
4. Same key with '-', '_', '.', ' ' separators removed matches a canonical name -> EXACT,
   or an alias -> ALIAS   (e.g. "Node JS" -> Node.js, "scikit learn" -> scikit-learn)
5. Anything else         -> UNKNOWN (kept with its evidence, flagged, never guessed)

Related technologies are never treated as aliases (TensorFlow != PyTorch).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.config import PROJECT_ROOT
from src.schemas import ExtractedSkill, MatchType, NormalizedSkill

TAXONOMY_PATH = Path(__file__).resolve().parent / "data" / "skill_taxonomy.json"
CUSTOM_SKILLS_PATH = PROJECT_ROOT / "config" / "custom_skills.json"

# Words that are skills but also common English / resume words; excluded from the
# free-text mention scan (they can still be matched when explicitly listed as skills).
_AMBIGUOUS_SCAN_TERMS = {"go", "rest", "spring", "node", "excel", "swift", "ruby", "vue", "agile", "statistics", "r", "c"}
_SEPARATORS = re.compile(r"[\s\-_.]+")


def normalize_key(text: str) -> str:
    key = re.sub(r"\s+", " ", text.casefold().replace("_", " ")).strip()
    return key.strip(" .,;:()[]{}\"'`*•-")


def _compact(key: str) -> str:
    return _SEPARATORS.sub("", key)


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, str], dict[str, str], dict[str, tuple[str, MatchType]]]:
    data = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    canonical: list[str] = list(data["canonical"])
    if CUSTOM_SKILLS_PATH.exists():
        canonical += [s for s in json.loads(CUSTOM_SKILLS_PATH.read_text(encoding="utf-8")) if isinstance(s, str)]
    canon_by_key = {normalize_key(c): c for c in canonical}
    alias_by_key: dict[str, str] = {}
    for alias, target in data.get("aliases", {}).items():
        if normalize_key(target) not in canon_by_key:
            raise ValueError(f"Alias '{alias}' points to unknown canonical skill '{target}'")
        alias_by_key[normalize_key(alias)] = canon_by_key[normalize_key(target)]
    compact: dict[str, tuple[str, MatchType]] = {}
    for key, target in alias_by_key.items():
        compact.setdefault(_compact(key), (target, MatchType.ALIAS))
    for key, target in canon_by_key.items():
        compact[_compact(key)] = (target, MatchType.EXACT)
    return canon_by_key, alias_by_key, compact


def reload_taxonomy() -> None:
    _load.cache_clear()
    _scan_patterns.cache_clear()


def approved_skills() -> list[str]:
    return sorted(_load()[0].values(), key=str.casefold)


def is_canonical(name: str) -> bool:
    return normalize_key(name) in _load()[0]


def add_custom_skills(names: list[str]) -> list[str]:
    """Approve new canonical skills (HR action). Returns the names actually added."""
    CUSTOM_SKILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: list[str] = []
    if CUSTOM_SKILLS_PATH.exists():
        existing = json.loads(CUSTOM_SKILLS_PATH.read_text(encoding="utf-8"))
    added = []
    for raw in names:
        clean = re.sub(r"\s+", " ", raw).strip()
        if clean and len(clean) <= 100 and not is_canonical(clean) and resolve(clean)[1] == MatchType.UNKNOWN:
            existing.append(clean)
            added.append(clean)
    CUSTOM_SKILLS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    reload_taxonomy()
    return added


def resolve(raw: str) -> tuple[str | None, MatchType]:
    """Map one raw skill string to (canonical_name, match_type)."""
    canon_by_key, alias_by_key, compact = _load()
    key = normalize_key(raw)
    if not key:
        return None, MatchType.UNKNOWN
    if key in canon_by_key:
        return canon_by_key[key], MatchType.EXACT
    if key in alias_by_key:
        return alias_by_key[key], MatchType.ALIAS
    hit = compact.get(_compact(key))
    if hit:
        return hit
    return None, MatchType.UNKNOWN


def normalize_skills(skills: list[ExtractedSkill]) -> list[NormalizedSkill]:
    """Normalize and de-duplicate extracted skills, preserving raw names and evidence."""
    out: list[NormalizedSkill] = []
    seen: set[str] = set()
    for skill in skills:
        canonical, match_type = resolve(skill.raw_name)
        dedupe_key = (canonical or normalize_key(skill.raw_name)).casefold()
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append(
            NormalizedSkill(
                raw_name=skill.raw_name[:100],
                canonical_name=canonical,
                match_type=match_type,
                evidence=skill.evidence[:500],
            )
        )
    return out


def split_skill_input(text: str) -> list[str]:
    """Split HR free-text input ("Python, ML; torch") into individual skill strings."""
    return [p.strip() for p in re.split(r"[,;\n]+", text or "") if p.strip()]


def normalize_job_skills(mandatory: list[str], preferred: list[str]) -> tuple[dict, list[str]]:
    """Build the jobs.skills JSON structure. Returns (skills_dict, unknown_skill_names)."""
    unknown: list[str] = []

    def build(items: list[str]) -> list[dict]:
        reqs = []
        for raw in items:
            canonical, _ = resolve(raw)
            if canonical is None:
                unknown.append(raw)
                continue
            reqs.append({"name": raw, "canonical_name": canonical, "minimum_proficiency": None})
        return reqs

    return {"mandatory": build(mandatory), "preferred": build(preferred)}, unknown


@lru_cache(maxsize=1)
def _scan_patterns() -> list[tuple[re.Pattern[str], str]]:
    canon_by_key, alias_by_key, _ = _load()
    terms = list(canon_by_key.items()) + list(alias_by_key.items())
    patterns = []
    for key, canonical in terms:
        if key in _AMBIGUOUS_SCAN_TERMS or (len(key) <= 2 and key not in {"c++", "c#"}):
            continue
        patterns.append((re.compile(r"(?<![\w+#])" + re.escape(key) + r"(?![\w+#])"), canonical))
    return patterns


def scan_text_for_skills(text: str) -> list[str]:
    """Canonical skills whose name/alias appears anywhere in the text (whole-word, case-insensitive)."""
    haystack = re.sub(r"\s+", " ", (text or "").casefold())
    found = {canonical for pattern, canonical in _scan_patterns() if pattern.search(haystack)}
    return sorted(found, key=str.casefold)
