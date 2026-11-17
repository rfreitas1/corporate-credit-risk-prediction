"""Attribute dictionary for the UCI Polish bankruptcy dataset.

Maps each ``AttrN`` column to its official definition (copied verbatim from
the UCI metadata, where it is called ``XN``), an analytical category, and
notes on quirks found while checking the data.

Categories (one per attribute, judgement calls documented in ``note``):
    profitability  - how much profit the firm earns relative to assets/sales
    liquidity      - ability to meet short-term obligations
    leverage       - capital structure and capacity to service debt
    activity       - efficiency: turnover ratios, collection/payment days
    size           - scale of the firm

Translation caveat: "gross profit" in this dataset is most likely the Polish
"zysk brutto", i.e. profit *before tax*, not sales minus cost of goods sold.
"""

from __future__ import annotations

import pandas as pd

CATEGORIES = ["profitability", "liquidity", "leverage", "activity", "size"]

# (attribute number, official definition, category, note)
_ATTRIBUTES: list[tuple[int, str, str, str]] = [
    (1, "net profit / total assets", "profitability", "Return on assets (ROA)."),
    (2, "total liabilities / total assets", "leverage", "Debt ratio."),
    (3, "working capital / total assets", "liquidity", "Altman Z input X1."),
    (4, "current assets / short-term liabilities", "liquidity", "Current ratio."),
    (5, "[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365", "liquidity", "Days of operating cash costs covered by net liquid assets (defensive interval)."),
    (6, "retained earnings / total assets", "profitability", "Altman Z input X2 (cumulative profitability)."),
    (7, "EBIT / total assets", "profitability", "Altman Z input X3. Identical to Attr14 and Attr18 in every row."),
    (8, "book value of equity / total liabilities", "leverage", "Altman Z' input X4 (book-value version for private firms)."),
    (9, "sales / total assets", "activity", "Asset turnover; Altman Z input X5. Same definition as Attr36 but values differ in ~38% of rows."),
    (10, "equity / total assets", "leverage", "Equity ratio. Equals 1 - Attr2 in only ~70% of rows (other balance-sheet items sit between equity and liabilities)."),
    (11, "(gross profit + extraordinary items + financial expenses) / total assets", "profitability", ""),
    (12, "gross profit / short-term liabilities", "leverage", "Profit relative to short-term debt (coverage); could also be read as liquidity."),
    (13, "(gross profit + depreciation) / sales", "profitability", "Cash-flow-style margin."),
    (14, "(gross profit + interest) / total assets", "profitability", "Identical to Attr7 in every row (redundant)."),
    (15, "(total liabilities * 365) / (gross profit + depreciation)", "leverage", "Debt payback period in days."),
    (16, "(gross profit + depreciation) / total liabilities", "leverage", "Cash-flow-to-debt coverage. Exactly 365 / Attr15."),
    (17, "total assets / total liabilities", "leverage", "Exactly 1 / Attr2 (redundant up to a transformation)."),
    (18, "gross profit / total assets", "profitability", "Identical to Attr7 in every row (redundant)."),
    (19, "gross profit / sales", "profitability", "Pre-tax margin (see translation caveat)."),
    (20, "(inventory * 365) / sales", "activity", "Inventory days. The authors dropped rows where this was missing (ARFF header filter)."),
    (21, "sales (n) / sales (n-1)", "activity", "Sales growth; requires the previous year's statement."),
    (22, "profit on operating activities / total assets", "profitability", "Operating ROA."),
    (23, "net profit / sales", "profitability", "Net margin."),
    (24, "gross profit (in 3 years) / total assets", "profitability", "Multi-year profit; requires prior statements."),
    (25, "(equity - share capital) / total assets", "leverage", "Accumulated reserves relative to assets."),
    (26, "(net profit + depreciation) / total liabilities", "leverage", "Cash-flow-to-debt coverage."),
    (27, "profit on operating activities / financial expenses", "leverage", "Interest coverage. Undefined for firms with no financial expenses."),
    (28, "working capital / fixed assets", "liquidity", ""),
    (29, "logarithm of total assets", "size", "Log scale, so it is the only well-behaved size measure."),
    (30, "(total liabilities - cash) / sales", "leverage", "Net debt to sales."),
    (31, "(gross profit + interest) / sales", "profitability", ""),
    (32, "(current liabilities * 365) / cost of products sold", "activity", "Payables days."),
    (33, "operating expenses / short-term liabilities", "liquidity", "Ambiguous: could be read as activity."),
    (34, "operating expenses / total liabilities", "leverage", "Ambiguous: could be read as activity."),
    (35, "profit on sales / total assets", "profitability", ""),
    (36, "total sales / total assets", "activity", "Asset turnover; see Attr9."),
    (37, "(current assets - inventories) / long-term liabilities", "liquidity", "Undefined for firms with no long-term liabilities, so missingness is informative."),
    (38, "constant capital / total assets", "leverage", "Constant capital = equity + long-term liabilities."),
    (39, "profit on sales / sales", "profitability", "Operating margin on sales."),
    (40, "(current assets - inventory - receivables) / short-term liabilities", "liquidity", "Cash ratio."),
    (41, "total liabilities / ((profit on operating activities + depreciation) * (12/365))", "leverage", "Debt payback; the 12/365 factor is odd as documented."),
    (42, "profit on operating activities / sales", "profitability", "Operating margin."),
    (43, "rotation receivables + inventory turnover in days", "activity", "Operating cycle. Exactly Attr44 + Attr20."),
    (44, "(receivables * 365) / sales", "activity", "Receivables days (DSO)."),
    (45, "net profit / inventory", "profitability", "Undefined for firms with no inventory."),
    (46, "(current assets - inventory) / short-term liabilities", "liquidity", "Quick ratio."),
    (47, "(inventory * 365) / cost of products sold", "activity", "Inventory days on a cost basis."),
    (48, "EBITDA (profit on operating activities - depreciation) / total assets", "profitability", "Documented as '- depreciation'; EBITDA normally adds it back. Possible documentation error."),
    (49, "EBITDA (profit on operating activities - depreciation) / sales", "profitability", "Same caveat as Attr48."),
    (50, "current assets / total liabilities", "liquidity", ""),
    (51, "short-term liabilities / total assets", "leverage", "Short-term debt ratio."),
    (52, "(short-term liabilities * 365) / cost of products sold)", "activity", "Payables days."),
    (53, "equity / fixed assets", "leverage", "How much of the fixed assets equity funds."),
    (54, "constant capital / fixed assets", "leverage", "Golden balance-sheet rule: long-term assets funded by long-term capital."),
    (55, "working capital", "liquidity", "Absolute amount in currency, NOT a ratio: scales with firm size."),
    (56, "(sales - cost of products sold) / sales", "profitability", "Gross margin in the usual sense."),
    (57, "(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)", "liquidity", ""),
    (58, "total costs /total sales", "profitability", "Cost ratio (higher = worse)."),
    (59, "long-term liabilities / equity", "leverage", "Long-term debt to equity."),
    (60, "sales / inventory", "activity", "Inventory turnover."),
    (61, "sales / receivables", "activity", "Receivables turnover."),
    (62, "(short-term liabilities *365) / sales", "activity", "Short-term liabilities in days of sales."),
    (63, "sales / short-term liabilities", "activity", ""),
    (64, "sales / fixed assets", "activity", "Fixed-asset turnover."),
]

# Columns that duplicate another column exactly (verified on all 43,405 rows).
REDUNDANT = {"Attr14": "Attr7", "Attr18": "Attr7"}

# Altman Z' (1983 private-firm model, book value of equity) mapped to the
# available attributes. Every input exists in the dataset as-is.
#   Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
#   Zones: Z' < 1.23 distress, 1.23-2.90 grey, > 2.90 safe.
ALTMAN_Z_PRIME = {
    "Attr3": 0.717,  # X1 working capital / total assets
    "Attr6": 0.847,  # X2 retained earnings / total assets
    "Attr7": 3.107,  # X3 EBIT / total assets
    "Attr8": 0.420,  # X4 book equity / total liabilities
    "Attr9": 0.998,  # X5 sales / total assets
}


def attribute_table() -> pd.DataFrame:
    """Return the dictionary as a DataFrame indexed by column name (``AttrN``)."""
    table = pd.DataFrame(
        _ATTRIBUTES, columns=["number", "definition", "category", "note"]
    )
    table.index = "Attr" + table["number"].astype(str)
    table.index.name = "attribute"
    return table


def describe(attribute: str) -> str:
    """Return a short human-readable label, e.g. ``'Attr1: net profit / total assets'``."""
    return f"{attribute}: {attribute_table().loc[attribute, 'definition']}"


# Fail fast at import time if the table is ever edited incorrectly.
assert [a[0] for a in _ATTRIBUTES] == list(range(1, 65)), "attributes must be 1..64"
assert all(a[2] in CATEGORIES for a in _ATTRIBUTES), "unknown category"
