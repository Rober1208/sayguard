# Blind judging of the daily report's recommended actions

The 177 actions in `../report_actions_judged.json` (20 synthetic days; three ways of asking Qwen2.5-1.5B for
the actions: `free_text`, `structured`, `menu`) were shuffled and split into 3 batches (`payload_blind.json`,
which holds only the action text and each day's real problems per camera). Each batch went to 2 independent
judge agents that never saw which approach wrote an action; `key.json` maps action ids back to approaches and
`verdicts.json` holds every verdict. The aggregate (both judges agreeing) is the `independent_judges` block in
`../report_actions_judged.json`; the judges agreed on "wrong problem" for 176 of 177 actions.

Re-running `scripts/eval_report.py` writes a new `../report_actions.json` with freshly sampled actions; it does
not re-run the judges, so the judged run is kept here as `report_actions_judged.json`.

Judge prompt (identical for all 6 judges, batch index 0-2):

> You are an independent reviewer of safety recommendations written for a construction-site daily report.
> Open ONLY this file: payload_blind.json (a JSON list of 3 batches). Judge batch index N (0-based) only.
> The batch has "batch": a list of actions {id, day, lang, action} and "truth": for each day, every camera and
> the problems it really had that day (with counts). Some actions start with a bold camera name and a problem
> label, some are plain sentences. Some are in Chinese.
> For EVERY action in the batch, decide:
> - names_camera: does the action name (or start with) a specific camera from that day's truth?
> - wrong_problem: true if the action ties a camera to a problem that camera did NOT have that day, or tells
>   people to fix a different kind of problem than the camera's real ones (e.g. a camera that only had crowding
>   gets "make everyone wear hard hats"; "reduce false alarms" for a camera with no false alarms). Cameras not
>   in the truth list count as wrong. If it names no camera, judge whether the problem it addresses happened
>   anywhere that day.
> - sensible: is it a concrete, reasonable safety action a site supervisor could carry out tomorrow, that fits
>   the problem?
> - why: one short sentence.
> Be strict and independent; do not assume any action is correct. Return one verdict per action id.
