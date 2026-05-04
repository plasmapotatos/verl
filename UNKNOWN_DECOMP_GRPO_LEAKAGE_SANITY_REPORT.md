# Unknown Decomp GRPO: Leakage + Sanity Sweep

- Generated at: `2026-04-12 20:40:36 `
- Run directory: `/work/hdd/bbsg/echen10/build_repos/verl/outputs/rl/unknown_decomp_grpo/unknown_decomp_grpo`
- Eval files scanned: `18` + base eval files `2`
- Rollout files scanned: `760`

## Executive Summary

- Held eval improved from `151/276` (54.71%) at step `0` to `170/276` (61.59%) at step `760`.
- Train-eval subset improved from `135/247` (54.66%) at step `0` to `172/247` (69.64%) at step `760`.
- Split-ID checks pass: train/held disjoint = `True`, train_eval subset of train = `True`, union(train,held)=unknown_train = `True`.
- Rollout overlap with held eval questions: exact `0`, normalized `0`.
- Rollout overlap with train-eval subset questions: exact `247`, normalized `247` (expected non-zero).

## 1) Split Integrity Checks

- `unknown_train_sample_ids`: `2752`
- `unknown_rl_train_frac0.9_sample_ids`: `2476`
- `unknown_rl_held_frac0.1_sample_ids`: `276`
- `unknown_rl_train_frac0.9_0.1_sample_ids`: `247`
- train ∩ held: `0`
- train_eval ⊂ train: `True`
- train_eval ∩ held: `0`
- train ∪ held == unknown_train: `True`

## 2) Eval Artifact Integrity

### Dataset: `unknown_rl_held_frac0.1`
- Steps present: `[0, 100, 200, 300, 400, 500, 600, 700, 760]`
- Same sample IDs across steps: `True`
- Sample ID count: `276`
- Missing sample IDs in rows: `0`
- Per-ID question mismatch count across steps: `0`
- Membership check vs split IDs: eval_size `276`, expected_size `276`, eval-minus-expected `0`, expected-minus-eval `0`

### Dataset: `unknown_rl_train_frac0.9_0.1`
- Steps present: `[0, 100, 200, 300, 400, 500, 600, 700, 760]`
- Same sample IDs across steps: `True`
- Sample ID count: `247`
- Missing sample IDs in rows: `0`
- Per-ID question mismatch count across steps: `0`
- Membership check vs split IDs: eval_size `247`, expected_size `247`, eval-minus-expected `0`, expected-minus-eval `0`

## 3) Metric Trajectories

### `unknown_rl_held_frac0.1`

| step | correct | total | attempted | not_attempted | accuracy | accuracy_attempted | attempt_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 151 | 276 | 265 | 11 | 54.71% | 56.98% | 96.01% |
| 100 | 160 | 276 | 270 | 6 | 57.97% | 59.26% | 97.83% |
| 200 | 171 | 276 | 273 | 3 | 61.96% | 62.64% | 98.91% |
| 300 | 169 | 276 | 272 | 4 | 61.23% | 62.13% | 98.55% |
| 400 | 167 | 276 | 271 | 5 | 60.51% | 61.62% | 98.19% |
| 500 | 172 | 276 | 270 | 6 | 62.32% | 63.70% | 97.83% |
| 600 | 168 | 276 | 272 | 4 | 60.87% | 61.76% | 98.55% |
| 700 | 171 | 276 | 270 | 6 | 61.96% | 63.33% | 97.83% |
| 760 | 170 | 276 | 270 | 6 | 61.59% | 62.96% | 97.83% |

### `unknown_rl_train_frac0.9_0.1`

| step | correct | total | attempted | not_attempted | accuracy | accuracy_attempted | attempt_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 135 | 247 | 235 | 12 | 54.66% | 57.45% | 95.14% |
| 100 | 155 | 247 | 240 | 7 | 62.75% | 64.58% | 97.17% |
| 200 | 163 | 247 | 238 | 9 | 65.99% | 68.49% | 96.36% |
| 300 | 166 | 247 | 239 | 8 | 67.21% | 69.46% | 96.76% |
| 400 | 167 | 247 | 241 | 6 | 67.61% | 69.29% | 97.57% |
| 500 | 169 | 247 | 240 | 7 | 68.42% | 70.42% | 97.17% |
| 600 | 170 | 247 | 239 | 8 | 68.83% | 71.13% | 96.76% |
| 700 | 172 | 247 | 241 | 6 | 69.64% | 71.37% | 97.57% |
| 760 | 172 | 247 | 241 | 6 | 69.64% | 71.37% | 97.57% |

## 4) Per-Question Transitions (Wrong->Right etc.)

### `unknown_rl_held_frac0.1`
- Total questions tracked: `276`
- Wrong/NotAttempted -> Correct by final step: `23`
- Correct -> NonCorrect by final step: `4`
- Oscillating (>=2 status transitions): `14`

#### Improved Examples (showing up to 12)
- sample_id `6` | `incorrect` -> `correct`
  question: What were the month and year when Obama told Christianity Today, "I am a Christian, and I am a devout Christian. I believe in the redemptive death and resurrection of Jesus Christ"?
  gold: January 2008
  start_resp: August 2010
  final_resp: January 2008
- sample_id `111` | `incorrect` -> `correct`
  question: On what day, month, and year was former Bangladeshi Inspector General Benazir Ahmed born?
  gold: 1 October 1963
  start_resp: 2 October 1968
  final_resp: 1 October 1963
- sample_id `709` | `not_attempted` -> `correct`
  question: What month, day, and year did Brenda Gayle Hayes, wife of W. V. Grant, die?
  gold: October 6, 2020
  start_resp: October 2020
  final_resp: October 6, 2020
- sample_id `945` | `incorrect` -> `correct`
  question: In which year was the municipality of Buriticá, Antioquia, Colombia, founded?
  gold: 1614
  start_resp: 1840
  final_resp: 1614.
- sample_id `1013` | `incorrect` -> `correct`
  question: How many children did Swiss painter Johann Caspar Füssli have with his wife, Elisabeth?
  gold: 18
  start_resp: 13
  final_resp: 18
- sample_id `1214` | `not_attempted` -> `correct`
  question: What day, month, and year was the album "Platinum" by Miranda Lambert certified platinum by the Recording Industry Association of America?
  gold: February 1, 2016
  start_resp: February 2016
  final_resp: February 1, 2016
- sample_id `1240` | `incorrect` -> `correct`
  question: What year was the municipality of Barbosa, Antioquia, Colombia, founded?
  gold: 1795
  start_resp: 1576
  final_resp: 1795
- sample_id `1629` | `incorrect` -> `correct`
  question: When did BAe and the American aircraft manufacturer McDonnell Douglas sign a memorandum of understanding regarding the McDonnell Douglas AV-8B Harrier II? Example answer: mm-yyyy
  gold: 08-1981
  start_resp: August 19, 1974
  final_resp: August 19, 1981
- sample_id `2188` | `incorrect` -> `correct`
  question: What is the durability of the Istarelle spear from Demon's Souls (2009)?
  gold: 800
  start_resp: 80
  final_resp: 800.
- sample_id `2394` | `incorrect` -> `correct`
  question: What day, month, and year was Rudi Dekkers born?
  gold: July 27, 1956.
  start_resp: July 20, 1956
  final_resp: July 27, 1956
- sample_id `2470` | `not_attempted` -> `correct`
  question: On what month, day, and year did the Computer History Museum launch the "Make Software: Change the World!" exhibit?
  gold: January 28, 2017
  start_resp: January 2017
  final_resp: January 28, 2017
- sample_id `2663` | `incorrect` -> `correct`
  question: In which year was the municipality of Pachavita, Boyacá, Colombia, founded?
  gold: 1716
  start_resp: 1853
  final_resp: 1716

#### Regressed Examples (showing up to 8)
- sample_id `82` | `correct` -> `incorrect`
  question: What year was the municipality of San Francisco, Antioquia, Colombia, founded?
  gold: 1830
  start_resp: 1830
  final_resp: 1880
- sample_id `289` | `correct` -> `incorrect`
  question: In Episode 7, Season 2 of the American version of "The Circle," who leaves the players a message on how to play the game Glammequins?
  gold: Jonathan Van Ness
  start_resp: Jonathan Van Ness
  final_resp: DeLeesa St. Agathe
- sample_id `719` | `correct` -> `incorrect`
  question: What year was the municipality of Jardín, Antioquia, Colombia, founded?
  gold: 1863
  start_resp: 1863
  final_resp: 1616
- sample_id `2458` | `correct` -> `incorrect`
  question: What day, month, and year was Lucy Letby born?
  gold: 4 January 1990.
  start_resp: 4 January 1990
  final_resp: 4 January 1990, 14 September 1994, 28 March 1991

### `unknown_rl_train_frac0.9_0.1`
- Total questions tracked: `247`
- Wrong/NotAttempted -> Correct by final step: `37`
- Correct -> NonCorrect by final step: `0`
- Oscillating (>=2 status transitions): `5`

#### Improved Examples (showing up to 12)
- sample_id `37` | `incorrect` -> `correct`
  question: What is the first and last name of the woman whom the British linguist Bernard Comrie married in 1985?
  gold: Akiko Kumahira
  start_resp: Anita
  final_resp: Akiko Kumahira
- sample_id `167` | `incorrect` -> `correct`
  question: What month, day, and year did Southeastern Louisiana University retire Robin Roberts' jersey?
  gold: 5 February 2011
  start_resp: February 2023
  final_resp: February 5, 2011
- sample_id `197` | `incorrect` -> `correct`
  question: On what day, month, and year did Randy Johnston (model) die?
  gold: October 11, 2008
  start_resp: December 2008
  final_resp: October 11, 2008
- sample_id `339` | `not_attempted` -> `correct`
  question: On what month, day, and year did Google run its first live-action video doodle?
  gold: April 15, 2011
  start_resp: April 2011
  final_resp: April 15, 2011
- sample_id `413` | `incorrect` -> `correct`
  question: What day, month, and year was Kris Cuppens born?
  gold: May 22, 1962
  start_resp: May 20, 1962
  final_resp: May 22, 1962
- sample_id `487` | `not_attempted` -> `correct`
  question: Which was the first Indian album to have more than 1 billion streams on Spotify?
  gold: Moosetape
  start_resp: The first Indian album
  final_resp: 'Moosetape'
- sample_id `579` | `incorrect` -> `correct`
  question: How heavy is my Ricoh GR III body only in grams?
  gold: 227 grams
  start_resp: 91 g
  final_resp: 227 grams
- sample_id `690` | `incorrect` -> `correct`
  question: In which year was a replacement pink donut with sprinkles sculpture unveiled in Springfield, New Zealand following arson?
  gold: 2012
  start_resp: 2010.
  final_resp: 2012.
- sample_id `713` | `incorrect` -> `correct`
  question: What ministerial title did Ana Figueroa hold while representing Chile at the United Nations from 1950 to 1952?
  gold: Minister plenipotentiary
  start_resp: Minister of the Supreme Court of Justice
  final_resp: Minister plenipotentiary
- sample_id `748` | `incorrect` -> `correct`
  question: What were the month, day, and year Whig politician James Vernon the Younger was born?
  gold: June 15, 1677
  start_resp: March 1675, day 1, and year 1678.
  final_resp: June 15, 1677
- sample_id `916` | `incorrect` -> `correct`
  question: On what day, month, and year did the Philippines ratify the Southeast Asian Nuclear-Weapon-Free Zone Treaty?
  gold: 21 June 2001
  start_resp: 28 March 2001.
  final_resp: 21 June 2001.
- sample_id `1103` | `incorrect` -> `correct`
  question: Name the person appointed as Vice-Chancellor of Jamia Millia Islamia in the year 1978.
  gold: Anwar Jamal Kidwai
  start_resp: Masud Husain Khan
  final_resp: Anwar Jamal Kidwai

#### Regressed Examples (showing up to 8)
- None

## 5) Rollout Leakage Check (Question Overlap)

- num_rollout_files: `760`
- num_rollout_lines: `389120`
- rollout_json_parse_failures: `0`
- unique_rollout_questions_exact: `2475`
- unique_rollout_questions_norm: `2475`
- held_questions_size: `276`
- train_eval_questions_size: `247`
- held_overlap_exact_count: `0`
- held_overlap_norm_count: `0`
- train_eval_overlap_exact_count: `247`
- train_eval_overlap_norm_count: `247`

- No held-question overlap found in rollouts under normalized matching.

## 6) Raw Generation Sanity

### `unknown_rl_held_frac0.1`
- Step `0`: n_rows `276`, status_counts `{'incorrect': 114, 'correct': 151, 'not_attempted': 11}`
  - empty_response_count `0`
  - echo_question_count `0`
  - contains_url_count `0`
  - multiline_3plus_count `0`
  - dont_know_like_count `0`
  - resp_char_len_avg `14.96`, median `12.00`, p95 `42`
  - resp_word_len_avg `2.54`, median `2.00`
  - top1_repeat_fraction `0.0109`
  - top repeated normalized responses (up to 8):
    - `3`x: `10`
    - `3`x: `university of california, berkeley`
    - `2`x: `2007`
    - `2`x: `1840`
    - `2`x: `1988`
    - `2`x: `13`
    - `2`x: `1853`
    - `2`x: `1995`
- Step `760`: n_rows `276`, status_counts `{'correct': 170, 'incorrect': 100, 'not_attempted': 6}`
  - empty_response_count `0`
  - echo_question_count `0`
  - contains_url_count `0`
  - multiline_3plus_count `0`
  - dont_know_like_count `0`
  - resp_char_len_avg `15.31`, median `13.00`, p95 `42`
  - resp_word_len_avg `2.60`, median `2.00`
  - top1_repeat_fraction `0.0109`
  - top repeated normalized responses (up to 8):
    - `3`x: `university of california, berkeley`
    - `3`x: `two`
    - `2`x: `2007`
    - `2`x: `1988`
    - `2`x: `10`
    - `2`x: `1995`
    - `2`x: `33`
    - `2`x: `6`

### `unknown_rl_train_frac0.9_0.1`
- Step `0`: n_rows `247`, status_counts `{'incorrect': 100, 'correct': 135, 'not_attempted': 12}`
  - empty_response_count `0`
  - echo_question_count `0`
  - contains_url_count `0`
  - multiline_3plus_count `0`
  - dont_know_like_count `0`
  - resp_char_len_avg `14.46`, median `13.00`, p95 `34`
  - resp_word_len_avg `2.48`, median `2.00`
  - top1_repeat_fraction `0.0081`
  - top repeated normalized responses (up to 8):
    - `2`x: `june 2008`
    - `2`x: `kpix-tv`
    - `2`x: `1964`
    - `2`x: `1968`
    - `2`x: `1934`
    - `2`x: `2002`
    - `2`x: `1991`
    - `1`x: `anita`
- Step `760`: n_rows `247`, status_counts `{'correct': 172, 'incorrect': 69, 'not_attempted': 6}`
  - empty_response_count `0`
  - echo_question_count `0`
  - contains_url_count `0`
  - multiline_3plus_count `0`
  - dont_know_like_count `0`
  - resp_char_len_avg `15.18`, median `13.00`, p95 `35`
  - resp_word_len_avg `2.61`, median `2.00`
  - top1_repeat_fraction `0.0121`
  - top repeated normalized responses (up to 8):
    - `3`x: `1964`
    - `2`x: `kpix-tv`
    - `2`x: `1968`
    - `2`x: `2002`
    - `2`x: `1991`
    - `1`x: `akiko kumahira`
    - `1`x: `january 8, 2019`
    - `1`x: `102510`

## 7) Caveats + Interpretation

- This sweep can verify split ID integrity, eval artifact consistency, and whether held eval questions appeared in RL rollouts. It cannot independently prove there was no contamination upstream of this run (for example, during SFT pretraining) without full data lineage metadata.
- The run directory name `unknown_decomp_grpo` has been reused across different experiments historically; old external logs in `/work/hdd/bbsg/echen10/unknown_decomp_grpo_staggered_*.out/.err` correspond to a different dataset config (`unknown_train_frac0.08/0.02`), not the `unknown_rl_*` eval artifacts analyzed here.
- Given no held-question overlap detected in rollouts and split checks passing, the observed held improvement is most consistent with genuine generalization from RL on the 90% train split plus advantages of the decompose-initialized checkpoint.
