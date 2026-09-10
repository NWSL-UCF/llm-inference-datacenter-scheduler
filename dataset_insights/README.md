# Dataset insights

Plots of the Azure LLM inference traces ([code](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv), [conversation](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv)). Raw CSVs live in `../dataset/` and are not in git. To download them, see [`dataset_download.txt`](dataset_download.txt).

Each request has `TIMESTAMP`, `ContextTokens`, `GeneratedTokens`.

- **Code:** 10th May 2024, Friday → 16th May 2024, Thursday
- **Conversation:** 12th May 2024, Sunday → 18th May 2024, Saturday

## Layout

```
code_conv_hourly.{png,pdf}   all 7 days, both traces
dataset_download.txt         how to download the raw CSVs
mon|tues|wed|thu|fri|sat|sun/
  png|pdf/
    code_day / conv_day      whole day (x-axis = hour 0–23)
    code_h00–h23             one hour (x-axis = minute 0–59)
    conv_h00–h23
```

Friday and Saturday use different calendar dates for the two traces (code is the earlier week, conversation the later).

## How to Get Insight About the Dataset

Start with `code_conv_hourly.png` to compare both traces across all seven days.

Then open a weekday folder. Example, Monday (13th May 2024, Monday):

- Whole day: `mon/png/code_day.png` and `mon/png/conv_day.png` — diurnal shape and weekday vs weekend load.
- One hour: `mon/png/code_h19.png` — burstiness inside 19:00–19:59 UTC.

Every panel shows request count (left axis) and summed context tokens (right axis), plus min / max / avg / median / std. Y-scales are shared within whole-day plots and within hour plots so panels are comparable.
