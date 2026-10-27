"""Deterministic skill normalization (tests 7-10) + validator guard-rails."""
from __future__ import annotations

from src.profile_validator_agent import reconcile_skills
from src.schemas import (
    ExtractedSkill,
    MatchType,
    NormalizedSkill,
    ProfileValidationOutput,
    SkillValidation,
    SkillValidationStatus,
)
from src.skill_normalizer import normalize_job_skills, normalize_skills, resolve, scan_text_for_skills


def test_alias_ml_to_machine_learning():
    assert resolve("ML") == ("Machine Learning", MatchType.ALIAS)
    assert resolve(" machine-learning ") == ("Machine Learning", MatchType.ALIAS)


def test_alias_torch_to_pytorch():
    assert resolve("torch") == ("PyTorch", MatchType.ALIAS)
    assert resolve("PyTorch") == ("PyTorch", MatchType.EXACT)


def test_unknown_skill_is_flagged_not_guessed():
    assert resolve("Quantum Basket Weaving") == (None, MatchType.UNKNOWN)
    # related technologies are NOT aliases
    assert resolve("TensorFlow")[0] == "TensorFlow"
    assert resolve("TensorFlow")[0] != "PyTorch"


def test_separator_variants_resolve():
    assert resolve("Node JS")[0] == "Node.js"
    assert resolve("scikit learn")[0] == "scikit-learn"
    assert resolve("C++") == ("C++", MatchType.EXACT)


def test_deduplication_keeps_first_evidence():
    skills = [ExtractedSkill(raw_name="ML", evidence="ML models"),
              ExtractedSkill(raw_name="Machine Learning", evidence="machine learning"),
              ExtractedSkill(raw_name="torch", evidence="using torch"),
              ExtractedSkill(raw_name="Foo", evidence="Foo"),
              ExtractedSkill(raw_name="foo", evidence="foo")]
    out = normalize_skills(skills)
    assert [(s.raw_name, s.canonical_name) for s in out] == [("ML", "Machine Learning"), ("torch", "PyTorch"),
                                                             ("Foo", None)]
    assert out[0].evidence == "ML models"


def test_job_skill_normalization_reports_unknown():
    data, unknown = normalize_job_skills(["Python", "Blorp"], ["k8s"])
    assert unknown == ["Blorp"]
    assert data["preferred"][0]["canonical_name"] == "Kubernetes"


def test_text_scan_whole_words_only():
    found = scan_text_for_skills("Built pipelines with PyTorch; javascripting is not a word. Use of Java.")
    assert "PyTorch" in found and "Java" in found and "JavaScript" not in found


RESUME = "Skills: ML, torch\nDeveloped predictive models for churn.\nTrained neural networks using torch."


def _norm(raw, canonical, match_type, evidence):
    return NormalizedSkill(raw_name=raw, canonical_name=canonical, match_type=match_type, evidence=evidence)


def test_validator_contextual_mapping_requires_review():
    normalized = [_norm("predictive modeling", None, MatchType.UNKNOWN, "Developed predictive models for churn.")]
    llm = ProfileValidationOutput(validated_skills=[SkillValidation(
        raw_name="predictive modeling", canonical_name="Machine Learning", match_type=MatchType.CONTEXTUAL,
        validation_status=SkillValidationStatus.VALID, evidence="Developed predictive models for churn.")])
    out = reconcile_skills(RESUME, normalized, llm)
    assert out[0].match_type == MatchType.CONTEXTUAL
    assert out[0].validation_status == SkillValidationStatus.REVIEW_REQUIRED  # never auto-VALID


def test_validator_cannot_change_deterministic_canonical_or_add_skills():
    normalized = [_norm("torch", "PyTorch", MatchType.ALIAS, "Trained neural networks using torch.")]
    llm = ProfileValidationOutput(validated_skills=[
        SkillValidation(raw_name="torch", canonical_name="TensorFlow", match_type=MatchType.EXACT,
                        validation_status=SkillValidationStatus.VALID, evidence="using torch"),
        SkillValidation(raw_name="Kubernetes", canonical_name="Kubernetes", match_type=MatchType.EXACT,
                        validation_status=SkillValidationStatus.VALID, evidence="invented")])
    out = reconcile_skills(RESUME, normalized, llm)
    assert len(out) == 1 and out[0].canonical_name == "PyTorch" and out[0].validation_status.value == "VALID"


def test_ungrounded_skill_is_rejected():
    normalized = [_norm("Rust", "Rust", MatchType.EXACT, "Rust services")]
    out = reconcile_skills(RESUME, normalized, ProfileValidationOutput())
    assert out[0].validation_status == SkillValidationStatus.REJECTED and out[0].grounded is False
