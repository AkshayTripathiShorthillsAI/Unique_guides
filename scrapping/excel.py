import openpyxl

data = [
    "2p16.3 (NRXN1) Deletions",
    "ADNP Related Syndrome",
    "Alazami Syndrome",
    "ANKRD11 And KBG Syndrome",
    "ARID1B Syndrome",
    "ATR-X",
    "Au-Kline syndrome (HNRNPK LOF variants and 9q21.32 microdeletions)",
    "Bainbridge-Roper Syndrome ASXL3",
    "Bohring-Opitz Syndrome (BOS)",
    "BRPF1-related disorder",
    "BWCFF Baraitser-Winter Cerebrofrontofacial Syndrome",
    "CACNA1A-related disorders",
    "CACNA1C Timothy syndrome",
    "Cantu Syndrome",
    "CASK-related disorders",
    "Chitayat Syndrome (ERF Variant)",
    "CLTC-related ID",
    "CNOT3-related disorder",
    "CTNNB1 syndrome",
    "DDX3X Syndrome",
    "DYRK1A and 21q22.13 deletion syndrome",
    "Floating-Harbor Syndrome",
    "FOXP1 syndrome",
    "FOXP2 Syndrome",
    "GATA6",
    "GATAD2B-associated neurodevelopmental disorder (GAND) GATAD2B syndrome",
    "GRIN2A related syndrome",
    "GRIN2B-related neurodevelopmental disorder",
    "HNRNPH2-NDD",
    "HNRNPU-related disorder",
    "HUWE1 related ID",
    "IQSEC2-related Disorder",
    "KAT6A Syndrome",
    "KIF11 Associated Disorder",
    "KIF1A",
    "Kleefstra Syndrome",
    "Koolen-De Vries Syndrome",
    "Li-Ghorbani-Weisz-Hubshman syndrome LIGOWS KAT8 variants",
    "MED12 Related Disorders",
    "MED13L",
    "MEF2C haploinsufficiency syndrome",
    "MPPH Syndrome",
    "MYT1L syndrome (MYT1L variants and 2p25.3 deletions)",
    "NAA10-related disorder Arg83Cys",
    "NONO-associated X-linked ID syndrome",
    "Norrie Disease",
    "PACS1 Related Syndrome",
    "PCDH19-related epilepsy",
    "PCGF2 Related Syndrome",
    "PUF60-related developmental disorder (Verheij syndrome)",
    "PURA And 5q31",
    "RHOBTB2 Syndrome",
    "SATB2 Syndrome (Glass syndrome)",
    "Say-Barber-Biesecker Syndrome",
    "SCN2A Related Conditions",
    "SETD2",
    "SETD5",
    "SIN3A Witteveen-Kolk syndrome WITKOS",
    "Single Gene Disorders - Autosomal Dominant Inheritance",
    "Single Gene Disorders - Autosomal Recessive Inheritance",
    "SLC12A2 syndrome and SLC12A2-related deafness",
    "SOX11 Syndrome And 2p25.2 Deletions",
    "SOX5 Syndrome Lamb Shaffer Syndrome 12p12 Deletions",
    "STXBP1 Disorders",
    "SYNGAP1 Syndrome",
    "TAB2-related syndrome",
    "TBCK Syndrome",
    "TBR1 related disorder",
    "TRAF7-related neurodevelopmental disorder",
    "TUBA1A - associated tubulinopathy",
    "USP7 related disorder",
    "WAC Syndrome (DeSanto-Shinawi Syndrome)",
    "X Chromosome Deletions Duplications and Single Gene Disorders",
    "ZMYND11 related syndromic intellectual disability"
]

# Create a new Workbook
wb = openpyxl.Workbook()

# Get the active worksheet
ws = wb.active

# Iterate through the data and write it to the first column of the worksheet
for idx, item in enumerate(data, start=1):
    ws.cell(row=idx, column=1, value=item)

# Save the workbook
wb.save("syndrome_data.xlsx")
