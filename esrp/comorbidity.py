"""Charlson and Elixhauser categories used by the ESRP study.

Only the categories the study needs are kept: the Elixhauser fluid and
electrolyte disorder (an ESRP variable), the seven that set the baseline SOFA
for the Sepsis-3 definition (eTable 1, Panel C), and the coded immunocompromise
proxy used for a subgroup (eTable 9, Panel C).

The prefix lists are copied verbatim from the ``medcodes`` package shipped with
the MIMIC-IV-ED benchmark (Xie et al. 2022), which implements Quan et al. 2005
(Med Care 43:1130). They are kept exactly as that package has them, including
its few departures from the published tables (for example, ICD-10 E22.2 is not
in the fluid and electrolyte list, and I13.0 is in the Elixhauser but not the
Charlson heart-failure list), because the published numbers were produced with
it. A code belongs to a category when it starts with any listed prefix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ICD9 = {
    "cci_CHF": ("39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493",
                "4254", "4255", "4256", "4257", "4258", "4259", "428"),
    "eci_CHF": ("39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493",
                "4254", "4255", "4256", "4257", "4258", "4259", "428"),
    "cci_Liver1": ("07022", "07023", "07032", "07033", "07044", "07054", "0706", "0709",
                   "5733", "5734", "5738", "5739", "V427", "570", "571"),
    "cci_Liver2": ("4560", "4561", "4562", "5722", "5723", "5724", "5725", "5726", "5727", "5728"),
    "cci_Renal": ("40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493",
                  "5830", "5831", "5832", "5833", "5834", "5835", "5836", "5837", "5880",
                  "V420", "V451", "585", "586", "V56", "582"),
    "eci_Renal": ("40301", "40311", "40391", "40402", "40403", "40493", "5880", "V420",
                  "585", "586", "V56", "V451"),
    "eci_Coagulopathy": ("2871", "2873", "2874", "2875", "286"),
    "eci_FluidsLytes": ("2536", "276"),
    # Coded immunocompromise proxy (eTable 9, Panel C): HIV, lymphoma or malignancy, i.e. any of
    # Charlson AIDS/HIV, malignancy, metastatic solid tumour and Elixhauser AIDS/HIV, lymphoma,
    # metastatic cancer, solid tumour without metastasis.
    "immunocompromise": ('042', '043', '044', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '2386'),
}
ICD10 = {
    "cci_CHF": ("I099", "I110", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429",
                "P290", "I50", "I43"),
    "eci_CHF": ("I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429",
                "P290", "I43", "I50"),
    "cci_Liver1": ("K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715", "K717",
                   "K760", "K762", "K763", "K764", "K768", "K769", "Z944", "B18", "K73", "K74"),
    "cci_Liver2": ("I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765", "K766", "K767"),
    "cci_Renal": ("I120", "I131", "N032", "N033", "N034", "N035", "N036", "N037",
                  "N052", "N053", "N054", "N055", "N056", "N057", "N250",
                  "Z490", "Z491", "Z492", "Z940", "Z992", "N18", "N19"),
    "eci_Renal": ("I120", "I131", "N250", "Z490", "Z491", "Z492", "Z940", "Z992", "N18", "N19"),
    "eci_Coagulopathy": ("D691", "D693", "D694", "D695", "D696", "D65", "D66", "D67", "D68"),
    "eci_FluidsLytes": ("E86", "E87"),
    "immunocompromise": ('B20', 'B21', 'B22', 'B24', 'C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C30', 'C31', 'C32', 'C33', 'C34', 'C37', 'C38', 'C39', 'C40', 'C41', 'C43', 'C45', 'C46', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C60', 'C61', 'C62', 'C63', 'C64', 'C65', 'C66', 'C67', 'C68', 'C69', 'C70', 'C71', 'C72', 'C73', 'C74', 'C75', 'C76', 'C77', 'C78', 'C79', 'C80', 'C81', 'C82', 'C83', 'C84', 'C85', 'C88', 'C90', 'C91', 'C92', 'C93', 'C94', 'C95', 'C96', 'C97'),
}
CATEGORIES = tuple(ICD9)


def flags(codes: pd.Series, version: pd.Series) -> pd.DataFrame:
    """One 0/1 column per category for each code (dots and case ignored)."""
    c = codes.astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
    v = np.asarray(pd.to_numeric(pd.Series(version), errors="coerce"), dtype=float)
    out = {}
    for k in CATEGORIES:
        out[k] = np.where(v == 9, c.str.startswith(ICD9[k]),
                          np.where(v == 10, c.str.startswith(ICD10[k]), False)).astype("int8")
    return pd.DataFrame(out, index=codes.index)
