#!/usr/bin/env python3
"""11번 줄의 튀어나온 한쪽 분사 끝만 인접한 10·12번 줄에 맞춘 시뮬레이션 사본."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parent
source = root / 'ES300h_front_bumper/painting_trajectory.csv'
target = root / 'ES300h_front_bumper/painting_trajectory_adjusted.csv'
with source.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)
out = []
removed = disabled = 0
for row in rows:
    if int(row['row_index']) == 11:
        n = int(row['point_index'])
        if n < 8:
            removed += 1
            continue
        if n <= 10 and row['paint'] == '1':
            row['paint'] = '0'
            disabled += 1
    out.append(row)
with target.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(out)
print(f'{target}: 11번 줄 시작점 0~7 제거({removed}점), 8~10 분사 OFF({disabled}점). 원본 유지.')
