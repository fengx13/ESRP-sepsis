"""Lists and cut-points for finding Sepsis-3 in the raw MIMIC-IV and MC-MED files.

Copied from the study's extraction code (XLab ``sepsis3_defs``) so this package is
self-contained. Definitions: eTable 1, Panel C of the appendix; Seymour et al.,
JAMA 2016 (suspected infection); Vincent et al. 1996 (SOFA).
"""
from __future__ import annotations
import re

ABX_AFTER_CULTURE_H = 72     # culture first: antibiotic within 72 h
CULTURE_AFTER_ABX_H = 24     # antibiotic first: culture within 24 h
SOFA_WINDOW_BEFORE_H = 48    # SOFA worst values from 48 h before suspicion ...
SOFA_WINDOW_AFTER_H = 24     # ... to 24 h after

# Systemic antibacterials: lower-case stems matched against the drug name
# (MIT-LCP mimic-code antibiotic.sql, audited against MIMIC-IV and MC-MED).
ANTIBIOTICS = [
    "adoxa", "ala-tet", "alodox", "amikacin", "amikin", "amoxicillin", "cefoxitin",
    "ampicillin", "augmentin", "avelox", "azactam", "azithromycin", "aztreonam", "bactocill",
    "bactrim", "bethkis", "biaxin", "bicillin", "cayston", "cefazolin", "cefepime", "cefotan",
    "cefotetan", "cefotaxime", "cefpodoxime", "ceftazidime", "ceftriaxone", "cefuroxime",
    "cephalexin", "chloramphenicol", "cipro", "ciprofloxacin", "claforan", "clarithromycin",
    "cleocin", "clindamycin", "cofatrim", "colistin", "coly-mycin", "cotrim", "cubicin",
    "dalacin", "daptomycin", "declomycin", "dicloxacillin", "dirithromycin", "doryx", "doxy",
    "doxycycline", "duricef", "dynacin", "ery-tab", "eryped", "eryc", "erythrocin",
    "erythromycin", "factive", "flagyl", "fortaz", "furadantin", "garamycin", "gentamicin",
    "kanamycin", "keflex", "ketek", "levaquin", "levofloxacin", "lincocin", "linezolid",
    "macrobid", "macrodantin", "maxipime", "mefoxin", "meropenem", "merrem", "methicillin",
    "metronidazole", "minocin", "minocycline", "monodox", "monurol", "morgidox", "moxatag",
    "moxifloxacin", "myrac", "nafcillin", "neomycin", "nitrofurantoin", "norfloxacin",
    "noroxin", "ocudox", "ofloxacin", "omnicef", "oracea", "oraxyl", "oxacillin", "pediazole",
    "penicillin", "periostat", "pfizerpen", "piperacillin", "primsol", "proquin", "raniclor",
    "rifadin", "rifampin", "rocephin", "smz-tmp", "septra", "solodyn", "spectracef",
    "streptomycin", "sulfadiazine", "sulfamethoxazole", "sulfatrim", "sulfisoxazole", "suprax",
    "synercid", "tazicef", "tetracycline", "timentin", "tobramycin", "tobi", "trimethoprim",
    "unasyn", "vancocin", "vancomycin", "vantin", "vibativ", "vibra-tabs", "vibramycin",
    "zinacef", "zithromax", "zosyn", "zyvox", "imipenem", "ertapenem", "invanz", "doripenem",
    "tigecycline", "tygacil", "ceftaroline", "teflaro", "telavancin", "dalbavancin",
    "oritavancin", "fosfomycin", "polymyxin", "ceftolozane", "fidaxomicin", "omadacycline",
    "tinidazole", "demeclocycline", "paromomycin", "cefdinir", "gatifloxacin",
]
# Surface preparations whose names match a stem (eye drops, ointments, catheter locks ...).
ANTIBIOTIC_EXCLUDE_NAME = (
    r"opht|\botic\b|topical|\btp\b|cream|ointment|\boint\b|lotion|gel\b|"
    r"\beye\b|\bear\b|drop\b|drops\b|drps\b|nasal|inhal|nebu|"
    r"irrig|enema|bladder|lock|intra-cathet|powder|shampoo|douche|"
    r"vaginal|\bvag\b|troche|lozenge")
TRUNCATED_MIN = 6


def name_matches(name: str, stems=ANTIBIOTICS) -> bool:
    """Drug name refers to a stem; also accepts a truncated leading token (``Vancomyci``)."""
    n = name.lower()
    if any(s in n for s in stems):
        return True
    head = re.match(r"[a-z]+", n)
    if head is None or len(head.group()) < TRUNCATED_MIN:
        return False
    tok = head.group()
    return any(s.startswith(tok) and len(s) > len(tok) for s in stems)


# ---- MC-MED -----------------------------------------------------------------
MCMED_LAB_ITEMS = {
    "PLATELET COUNT (PLT)": "platelets",
    "BILIRUBIN, TOTAL": "bilirubin",
    "CREATININE": "creatinine",
    "POC:CREATININE,ISTAT": "creatinine",
    "POC:LACTATE, ISTAT": "lactate",
    "LACTATE.WHOLE BLD": "lactate",
}
MCMED_CULTURE_ORDER_TYPES = ("Microbiology Culture", "Microbiology")
MCMED_VASOPRESSOR_NAME = r"norepinephrine|epinephrine|phenylephrine|vasopressin|dopamine|dobutamine|levophed"
MCMED_VASOPRESSOR_EXCLUDE = (r"lidocaine|ophth|opht\b|drop|atin\b|auto-?inject|"
                             r"pen\b|nasal|topical|inhal|racemic")
MCMED_VASOPRESSOR_REQUIRE = r"iv soln|iv infusion|/250 ml|/100 ml|mcg/ml|unit/ml"
# Monitor artefacts removed before taking a minimum (MC-MED charts once a minute).
PLAUSIBLE = {"MAP": (20, 200), "SBP": (40, 300), "DBP": (15, 200)}


# ---- MIMIC-IV ---------------------------------------------------------------
MIMIC_LAB_ITEMS = {
    51265: "platelets",     # Platelet Count, Blood, Hematology
    50885: "bilirubin",     # Bilirubin, Total, Blood, Chemistry
    50912: "creatinine",    # Creatinine, Blood, Chemistry
    50813: "lactate",       # Lactate, Blood, Blood Gas (septic shock only)
}
# Continuous vasopressors in icu/inputevents (mimic-code vasoactive_agent.sql).
MIMIC_VASOPRESSORS = {221906: "norepinephrine", 221289: "epinephrine", 221662: "dopamine",
                      221653: "dobutamine", 222315: "vasopressin", 221749: "phenylephrine",
                      229617: "angiotensin_ii"}
# prescriptions.route values that deliver to a surface rather than the bloodstream.
NON_SYSTEMIC_ROUTES = {"TP", "TD", "NAS", "IRR", "LOCK", "VAGINAL", "BOTH EYES", "LEFT EYE",
                       "RIGHT EYE", "OU", "OS", "OD", "BOTH EARS", "LEFT EAR", "RIGHT EAR",
                       "AU", "AS", "AD"}
# emar.event_txt values meaning the dose reached the patient.
EMAR_GIVEN = ["Administered", "Administered in Other Location", "Confirmed Administered",
              "Delayed Administered", "Partial Administered", "Started", "Restarted"]
SUSPICION_BEFORE_ARRIVAL_H = 24   # suspicion counted from 24 h before ED arrival to ED departure
MAP_PLAUSIBLE = (20, 200)


# ---- White-cell count for SIRS (comparator only) -----------------------------
# First count from 1 h before to 6 h after ED arrival (MIMIC: by charttime);
# MC-MED: first count ordered from arrival to 6 h after.
MIMIC_WBC_ITEMS = (51300, 51301, 51755, 51756)
MCMED_WBC_COMPONENT = "WHITE BLOOD CELLS (WBC)"
WBC_WINDOW_H = {"mimic": (-1, 6), "mcmed": (0, 6)}
