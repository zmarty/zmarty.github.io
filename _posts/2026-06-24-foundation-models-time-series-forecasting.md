---
layout: post
title: "Forecasting with Foundation Models: Capacity Planning and Incident Detection"
date: 2026-06-24 10:02:12 -0700
categories: [AI]
tags: [llm, forecasting, timesfm, time-series, capacity-planning]
description: "Foundation models like TimesFM forecast numerical time series zero-shot. I tested them on payment transaction volume, added holiday covariates, and learned where they help and where they hurt."
---

When most people hear "foundation model," they picture an LLM writing emails or a diffusion model painting images. But the same architecture that predicts the next word can also predict the next number. There is a small family of foundation models trained on billions of real-world time-series points instead of text, and they can forecast numerical values out of the box, with no training on your data.

One of these is Google's [TimesFM](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/), a decoder-only model with 200M parameters (tiny next to today's LLMs) that was pre-trained on 100 billion time-points from sources like Google Trends and Wikipedia pageviews. It treats a patch of contiguous time-points the way an LLM treats a token, then predicts the next patch. In practice it gives good zero-shot forecasts on data it has never seen.

## Two use cases: capacity planning and incident detection

If you run production systems, there are two obvious applications.

The first is capacity planning. If you can forecast transaction volume, request rate, or resource utilization a week ahead, you can provision infrastructure before you need it instead of scrambling once you are already overloaded.

The second is live site incident detection. A forecast comes with a prediction interval, which is the band where the next value should land. When live traffic falls outside that band, you have a statistical signal that something is wrong, often earlier than a static threshold alert would catch it. The forecast number is useful, but the uncertainty band is what makes this work for alerting. It is the difference between "alert me when CPU hits 90%" and "alert me when reality diverges from what the model expected."

## Adding context with covariates

Plain history is not always enough. Real-world numbers get pushed around by known events: a holiday, a sale, a product launch. TimesFM lets you supply covariates, which are extra signals that run alongside the main series, so the model can account for things the raw history can't explain on its own.

For payment forecasting, the obvious covariate is the calendar. You can tell the model that a given day is a holiday, or that it is Black Friday, and let it adjust. In theory this should sharpen the forecast around exactly the days that matter most for a payments business. In practice, as I found out, covariates help in some cases and hurt in others.

## The experiment: forecasting payment volume

I tested this on real transaction data. The full write-up and code are in my [timesfm-experiments repo](https://github.com/zmarty/timesfm-experiments), but here is the short version.

I used the Brazilian E-Commerce Public Dataset by Olist (about 100k anonymized orders from 2016 to 2018) and treated every purchase as one transaction. After aggregating to a clean hourly series of roughly 14,000 hours, the data has the structure you would hope for: a strong daily cycle (quiet overnight, busy midday), a clear weekly rhythm, and steady growth across 2017 and 2018.

<img width="1000" alt="Hourly transaction volume over time, with holiday markers in red and commercial-event markers in green. The tallest spike is Black Friday 2017." src="output/overview_hourly.png" />

I then ran TimesFM 2.5 (200M) zero-shot, with no fine-tuning, holding out the last 7 days (168 hours) and comparing two versions of the model. The baseline sees only the past transaction counts. The covariate version also sees Brazilian holidays and commercial events.

To model the calendar properly, I split events into two opposite forces rather than one vague "holiday" flag. Some days are dips, like statutory holidays and São Paulo state holidays, when volume drops. Others are spikes, like Black Friday, when volume surges. That distinction turned out to matter.

### An ordinary week: the baseline is already good enough

On a normal week with no events, TimesFM reproduced the routine structure zero-shot and landed within about 3 transactions per hour of reality. Adding covariates made it worse, by roughly 10% on MAE. With no event to explain, the covariates had nothing to contribute except noise. The lesson is that covariates are not free, and you should only use them when the forecast window actually contains the events you are modeling.

<img width="1000" alt="Ordinary-week forecast vs actual: baseline on top, covariate model on the bottom, with 80% prediction-interval bands." src="output/forecast.png" />

### Black Friday: the model can't invent a spike it has never seen

Black Friday 2017 was the first one in the dataset, so the model had no prior occurrence to calibrate against. Both versions badly missed the spike, forecasting around 10 transactions per hour against an actual peak near 87. A binary flag can say that something is happening today, but it cannot say to expect five times normal. The model has to learn that magnitude from at least one prior occurrence.

The prediction interval coverage is just as revealing. It collapsed from about 80% on calm weeks to about 48% during Black Friday. The model was overconfident exactly when volatility was highest, which is the kind of signal you would want feeding an incident-detection system.

<img width="1000" alt="Black Friday week forecast vs actual. Both the baseline and covariate models miss the large spike highlighted by the shaded event band." src="output/forecast_blackfriday.png" />

### Mother's Day: the most important lesson

This was the most interesting result. With enough history for the model to have seen a previous Mother's Day, adding a naive single-day holiday flag made the forecast 38.5% worse.

The reason is specific to shipping-based e-commerce: the order surge for a gift holiday happens in the run-up days, not on the day itself. Mother's Day is a Sunday, which is naturally a low-volume trough. Flagging the calendar day told the model to expect a spike on a quiet Sunday, which pushed the forecast the wrong way.

```
Mother's Day 2018 (flagged day = Sun 2018-05-13 = 207 orders):
  run-up:  Mon 05-07 = 372   Tue 05-08 = 331   Wed 05-09 = 344   <- real surge
  the day: Sun 05-13 = 207                                       <- FLAGGED (trough!)
  post:    Mon 05-14 = 364   Tue 05-15 = 352
```

Redesigning the covariate into a pre-event run-up window, flagging the days before the holiday instead of the day itself, roughly halved the damage and restored the prediction interval coverage. The takeaway is that covariate alignment matters more than covariate presence. A correctly named but mistimed signal is worse than no signal at all.

<img width="1000" alt="Mother's Day week forecast vs actual after redesigning the covariate into a pre-event run-up window." src="output/forecast_mothersday.png" />

## What I learned

A 200M-parameter model reproduced the hourly and weekly rhythm of a payments business with zero training, which is impressive on its own. The prediction interval is the part I would actually build on, because watching reality leave that band is a clean signal for both capacity planning and incident detection. Covariates are more conditional: they add value only when the event is present in the forecast window and aligned with where demand actually lands. And no model can predict the magnitude of an unprecedented spike, since the first Black Friday needs at least one prior occurrence to calibrate against.

On this dataset, the plain covariate-free baseline was the most accurate model. That is less a flaw in covariates than a consequence of three fixable data limits: the baseline is already near the noise floor, there are too few event repetitions to learn from, and binary flags can't express magnitude. With several years of history and a magnitude-aware encoding instead of 0/1 flags, the result would likely change.

The broader point is that the same transformer idea behind chatbots is now a practical, zero-shot tool for the numerical forecasting problems that operations and SRE teams deal with every day. If you run a system that produces a steady stream of numbers, it is worth a look.

The full code, methodology, and a detailed log of every problem I hit are in the [timesfm-experiments repo](https://github.com/zmarty/timesfm-experiments).
