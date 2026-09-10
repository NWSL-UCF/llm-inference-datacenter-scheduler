# Dataset insights

Plots of the Azure LLM inference traces ([code](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv), [conversation](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv)). Raw CSVs live in `../dataset/` and are not in git.

Each request has `TIMESTAMP`, `ContextTokens`, `GeneratedTokens`. **Code** is 2024-05-10–16; **conversation** is 2024-05-12–18.

## Layout

```
hourly/{png,pdf}/     one panel = one full day; x-axis is hour 0–23 UTC
minutely/{png,pdf}/   one panel = one hour on one day; x-axis is minute 0–59
  code|conv/<weekday>_<date>/h00–h23
```

Start with `hourly/png/code_conv_hourly.png` (all days, both traces). Per-day files are `*_day_{mon|…|sun}_YYYY-MM-DD.*`.

**Hourly:** diurnal pattern and weekday vs weekend load.  
**Minutely:** burstiness inside a given hour. Example: `minutely/png/code/mon_2024-05-13/h19.png`.

Every panel shows request count (left axis) and summed context tokens (right axis), plus min / max / avg / median / std. Y-scales are shared within hourly plots and within minutely plots so panels are comparable.
