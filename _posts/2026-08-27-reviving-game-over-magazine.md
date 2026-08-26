---
layout: post
title: "Reviving a 30-year-old video game magazine with LLMs"
date: 2026-08-27 09:00:00 -0700
categories: [AI, Development]
tags: [llm, claude, opus, document-ai, digital-preservation]
description: "I converted all 16 issues of Game Over, Romania's first PC gaming magazine, from raw scanned PDFs into a searchable website. Plain OCR cannot do this because of the layout. I tested six models and Opus 5 was the only one that could."
---

I liberated a 30-year-old video games magazine I used to buy as a kid using the best technology known to man. *Game Over* was the first Romanian magazine about PC games, published from 1996 to 1999, shortly after the fall of Communism. All 16 issues survived only as raw scanned PDFs, and until recently it would have been impossible to automaticallyu extract the articles. OCR would not have worked here. People assume the hard part of an old magazine is the words. It wasn't. The hard parts were the two things nobody thinks about: reading the layout and figuring out that three unrelated game reviews share one page, and cropping the screenshots out of the scans. I tested Opus 5, Sonnet 5, MiniMax M3, DeepSeek V4 Pro and Kimi K3 on the work and Opus 5 was on top.

The result is live at [gameover.ro](https://www.gameover.ro) and it spans **659 articles, 569,006 Romanian words, and 3,991 screenshots** cut out of the page scans. It took well over a billion tokens of vision work. In this post I'll cover why the layout breaks OCR, how I structured the agents, and what I learned from it.

Here's how the home page listing the issues looks like:

<img width="500" height="428" alt="image" src="https://github.com/user-attachments/assets/1d5ca9ff-88f1-433a-9a7c-4b843472f0f1" />

And here is how the reader looks like when the users read articles. Note the cropped

<img width="500" height="462" alt="image" src="https://github.com/user-attachments/assets/23a89c62-0f80-4e5c-ba45-c4d5556b3b10" />

## Why plain OCR fails on this

There's no text layer and no separately embedded images, so the prose and the screenshots both have to come out of the same picture. Here is a real page from issue 1, March 1996.

<img width="500" height="706" alt="image" src="https://github.com/user-attachments/assets/8c988487-f3b0-4726-a11e-693273740ed8" />

Here's what makes it hard:

- **Three separate articles on one page.** Cyberia, The Horde and Battle Bugs are reviews of three unrelated games, and the only thing dividing them is a change of lettering and a band of colour. No line, no gap, no marker of any kind.
- **Reading order doesn't match the layout.** The Cyberia review runs down the left column and continues at the *top* of the right column, but the *bottom* of that same right column starts the Battle Bugs review, so anything that reads column by column will blend two unrelated games into one article.
- **Rating boxes float inside body text.** There are two on this page, each with five percentages, one belonging to Cyberia and one to Battle Bugs. Both sit mid-column with a review above and a review below them. The magazine's convention is that a box belongs to the review whose text it follows, which is not obvious, and I got it backwards on two articles early on by trusting a summary instead of looking at the page.
- **The text and the screenshots interlock.** Body text wraps around the images, and on some pages it wraps around irregular artwork that has no border at all, so there's no clean rectangle separating one from the other. This is what makes the images as hard to extract as the words.
- **Headlines are drawn, not typeset.** "CYBERIA" is outlined 3D lettering and "BATTLE BUGS" is part of an illustration, so character recognition doesn't read them at all, even though a person reads them instantly.
- **Romanian with diacritics**, heavy 1990s slang, and English game terms quoted inside it. A lot of the hardest passages are thin white type printed over a dark screenshot, which is exactly where old OCR gives up.
- **Articles span pages.** The BioForge walkthrough runs across six pages and Full Throttle across seven, and nothing on page four of a seven-page walkthrough tells you it's still the same article except a small tab in the margin and a sentence that continues from where the previous page stopped.

None of this is really character recognition, it's document segmentation, which is a comprehension problem. Until recently the only thing that could do it was a person looking at the page, and across 878 pages that adds up to years of volunteer work, which is why nobody had done it. A vision model doesn't go character by character, it looks at the page and works out what it's looking at, and that's the change that made this possible.

## Cutting the screenshots out is the same problem

Segmentation isn't only about finding the article text, it's also about finding everything that isn't text. A 1996 review of Cyberia without the Cyberia screenshot is a much worse artifact than the original, so the images have to come out too, and because the text wraps around them there's no border to snap to. There were 3,991 images to extract, and for each one something has to produce pixel coordinates: left, top, right, bottom, plus four corners for the handful of elements the magazine printed at an angle on purpose.

Nothing generates those for you. The loop is to estimate a box, crop it, look at the result, notice that a sliver of the neighbouring text column came along with it, pull that edge in and crop again. Fitting the printed caption inside the box while keeping the next paragraph out of it is fiddly, and it's where most of the errors were.

The trick that helped most was to stop guessing coordinates and instead draw a labelled grid over the region in page coordinates and read the numbers off it. One of the agents came up with this partway through issue 1 and I used it for every issue afterwards.

The bigger lesson is to save the coordinates, not just the cropped image. Issue 1's 138 boxes were tuned by hand and then thrown away, because they only ever existed as arguments on a command line, so the JPEGs survived but the coordinates didn't. That cost me later: partway through the project I decided to raise crop quality across the archive from JPEG q75 to q95, and for issue 16 that was a single command because its boxes were recorded and all 246 crops could be regenerated from the source PDF. Issue 1 can never be upgraded, and it also can't benefit if a better scan ever turns up, because redoing it would mean re-deriving 138 boxes by hand.

## How I structured the agents

A coordinator agent reads every page of an issue itself and writes out a manifest with where each article starts and stops, which pages it covers, its section, its rating, and the offset between the PDF page number and the printed one. This part can't be delegated, because article boundaries only make sense to something that has seen the whole issue, and a subagent given three pages has no way to tell whether page four continues the article or starts a new one.

After that, one article at a time, a subagent:

1. Transcribes the text
2. Works out the crop boxes for its screenshots
3. Cuts them and checks its own crops by looking at them
4. Writes the article as Markdown with images placed between paragraphs
5. Saves the coordinates

I settled on **one article per agent** by measuring rather than by taste. An agent's cost doesn't track the amount of work it does, because every image it has looked at stays in its prompt and gets billed again on every later turn. On one issue I ran twelve long agents of between 56 and 337 turns each:

| agent length | cost per turn |
|---|---|
| under 100 turns | $0.054 |
| over 200 turns | $0.189 (**3.5x**) |

Same model and same work, and the only variable is how much history each turn is dragging along. Splitting those 43 articles into short single-article agents works out at roughly a quarter of the cost. Put another way, a screenshot read at turn 20 of a 250-turn agent gets paid for about 230 more times.

## The limit was the quota, not the money

I wasn't paying API rates for any of this. I started on the $20 Pro plan, found it far too limited, and moved to the **$100/month Max 5x plan**, with the Anthropic models running through the cloud API. That plan gives you a quota on a rolling five-hour window, and how much you actually get in a given window varies, so the orchestrator and its subagents would run until they hit the ceiling and then everything still in progress would die. Two design decisions came directly out of that.

### 1. Run subagents one at a time, never in parallel

This is the opposite of what you'd normally do with agents, but my results were completely one-sided:

| agents launched at once | outcome |
|---|---|
| 6 | killed mid-run |
| 9 | 7 killed, after 30 of 41 articles |
| 5 | all 5 killed at 35 min, 6 of 40 articles saved |
| 1 (sequential) | finished, every time |

Running in parallel doesn't change what an issue costs in total, it only changes how much work you lose when the quota runs out. A cut-off takes whatever is in flight, so with one agent you lose at most one half-finished article and with nine you lose nine. Going slowly also lets earlier usage fall out of the rolling window, so a paced run can get through an issue that a parallel run can't.

The trade-off is real, though. Running in series kept the orchestrator's context clean and made the work survivable, but it didn't make anything faster, and an issue takes as many turns as it has articles.

### 2. Write everything to disk immediately

Each article is saved as soon as it's done, and each crop's coordinates are appended the moment that crop is accepted rather than batched at the end. This is what makes an interruption cheap. On one issue the quota killed an agent that was still reading and it cost nothing at all, because everything it had finished was already saved. On another the agent died while reviewing its crops, but it had already written both the crops and their coordinates, so those crops could be verified against the recorded boxes and kept.

There's a related rule I had to learn, which is to never kill an agent that's still working. Its tokens are already spent, and killing it before it saves anything turns that spend into nothing.

The day-to-day version of this was less tidy. When a window ran out I'd wait it out and then use Claude's remote control to restart the run from my phone, driving the session on my Mac Mini, so a good part of this archive was resumed from the couch.

## What it cost

The token numbers are in the session logs, so I can be exact about those. Over four days at the end of July, on one machine, across the main sessions and 17 subagent transcripts:

| | |
|---|---|
| assistant messages | 5,746 |
| output tokens | 2,665,399 |
| cache writes | 179,576,528 |
| cache reads | 807,205,561 |
| **total** | **989,521,074** |

That's just under a billion tokens in four days, and the shape of it is the interesting part: 2.6 million tokens out against 987 million in, a ratio of about **1 to 371**, with 82% of the total being cache reads. That's the re-billing effect showing up in the totals. On one issue I audited closely, images were 83% of all cached input, and 1,179 image reads that would have cost 1.14M tokens to read once ended up billed at around **331M tokens** as repeated cache reads, a 240x multiplier.

At Claude Opus 5 list prices those four days would have come to about **$2,021**, and that window covered finishing one issue, most of another, all of a third, and most of the website build. What I actually paid that month was $100, and across the whole project it was roughly **$200 of subscription for 16 issues**, about $12.50 an issue.

## Why Opus won

Opus is the only model that could hold this job together, and the gap wasn't subtle. The task is unusual in that it's a vision problem, a long-horizon agentic problem, and a fidelity problem all at the same time, where a confident guess is worse than admitting failure.

Sonnet 5 looks cheaper per token and was not cheaper in practice. It produced more tokens for the same articles and the quality was lower, so the discount disappeared into the extra output and I ended up paying about the same for worse transcriptions. Kimi K3 behaved much the same way, and the rest were not close.

One failure mode is worth mentioning because it cost me two whole agent runs. Two Opus agents on issue 12 stopped before transcribing anything and asked what the legal basis was for reproducing a copyrighted 1998 magazine in full. Both were right to ask, and both burned their tokens, 73k and 50k, without producing a single article.

The fix took one paragraph. *Game Over* is an orphan work: still in copyright, but the publisher is gone, the website is gone, and there's no findable rights holder. The archive is non-commercial, credits the magazine and its writers, and will remove anything a verifiable rights holder objects to. That statement was already written down, but it was sitting in a file the agent had to go and find, and an agent can stop before it finds it. Once the paragraph went into every extraction prompt directly, the problem disappeared. If your agents need permission to do their job, put the permission in the prompt.

## Preserving the mistakes too

The transcription follows the page even when the page is wrong. *Game Over* was loosely proof-read, so the archive keeps its typos and its factual errors, and one caption reads "Beyond The Call Od Duty" because that's what was printed. One 1999 review has eight typos in a single article and all eight are still there. Gaps get marked rather than filled in, so where a passage genuinely can't be read it's marked unreadable instead of being reconstructed into something plausible. The archive also keeps a defect register, about 24,500 words of it, listing every irrecoverable scan defect, every printed error and every uncertain reading.

Two recovery tricks came out of this that I'd use again.

The first is the conjugate leaf. Some issues were scanned sheet by sheet and then split into pages slightly off centre, which chops one to three characters off the inner margin, but those characters aren't lost. On a saddle-stitched magazine of N pages, page *n* was printed on the same sheet of paper as page *N+1-n*, so the missing characters are still there, hanging off the opposite edge of that other page's scan. One issue was losing characters on about 47 of 75 lines on a single page, and every one of them was sitting in a different file. Eleven pages in that issue came back this way.

The second is that *Game Over* had a website between 1999 and 2001 which republished a lot of the same articles as plain ASCII, so I pulled it out of the Wayback Machine and diffed it against the transcriptions. It's a detector rather than an authority, because it's a different edition whose editor silently fixed typos, so a disagreement only means go and look at the scan. It caught the best mistake of the whole project: a 1999 article titled "Hitman" turns out not to be the Eidos game at all, but a Romanian indie project that happened to share the name, written up by an author who went and visited the developers. Since game titles are what link articles together across all 16 issues, that error would have quietly filed a Romanian hobby project into the Eidos franchise.

## The domain cost more than the AI

`gameover.ro` was sitting with a squatter, and the negotiation took several weeks. They opened above **$4,000** and I got it down to **$650**, which came to $672.75 on the invoice once the 3.5% card fee went on, plus RoTLD registry fees of 12 EUR to transfer it and 108 EUR to renew it for nine years.

So the domain cost more than three times the AI processing, which says something about which parts of this were actually scarce.

## Credit where it's due

The scans came from [Când Apare Revista](https://arhiva.candaparerevista.ro/), a Romanian community that has spent years finding physical copies of 1990s and 2000s Romanian gaming and computing magazines and scanning them page by page, and this project builds on that work. What scanning can't do, because it needs thousands of hours of human attention, is turn the pictures back into text, and that's the part this archive adds.

## What I learned

The thing that made this possible isn't better OCR, it's that a model can look at a magazine page and understand how it's put together: that this is three articles and not one, that this column continues that one, that this rating box goes with the review above it, and that this headline is a game title even though it's drawn as artwork. That capability replaced years of volunteer work and it's only recently become usable.

Transcribing the words was only half the job. Cutting out 3,991 screenshots and recording the coordinates so they can be cut again took as much effort as the text, and I'd treat those coordinates as the durable artifact from day one. I didn't, and issue 1 is permanently stuck at lower quality because of it.

Running agents against a hard quota is a different problem from running them against a budget. When the thing that can stop you is a rolling window that kills everything in progress, you stop optimising for speed and start optimising for how much finished work survives being cut off, which in practice means short agents doing one small thing each, saving results the moment they exist, and running strictly one at a time. It's slower, and it's the only version that ever finished an issue.

The archive is at [gameover.ro](https://www.gameover.ro). It's in Romanian, because the magazine was.
