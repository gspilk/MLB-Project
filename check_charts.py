from openpyxl import load_workbook
wb = load_workbook('output/mariners_report_20260901.xlsx')
ws = wb['Team Summary']
chart = ws._charts[0]
print('Number of series in this chart:', len(chart.series))
for s in chart.series:
    print(' -', s.val.numRef.f if s.val and s.val.numRef else s)