from pathlib import Path

multiSiteMri_int_to_site = {0: 'ISBI', 1: "ISBI_1.5", 2: 'I2CVB', 3: "UCL", 4: "BIDMC", 5: "HK"}
multiSiteMri_site_to_int = {v: k for k, v in multiSiteMri_int_to_site.items()}
cc359_data_path = Path('**Add path**')
cc359_splits_dir = Path('**Add path**')
cc359_results = Path('**Add path**')
msm_data_path = Path('**Add path**')
msm_splits_dir = Path('**Add path**')
msm_results = Path('**Add path**')