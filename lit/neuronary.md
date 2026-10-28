# neuronary.org: a public dictionary of neuron steering recipes (read 2026-10-28, pointed out by Juliana)

Site and repo by lyra (github.com/lyramakesmusic/neuronary). A recipe = a set of (layer, neuron, signed dose) plus
example outputs, same dose unit as ours (dose 1 = one median residual norm along W_down[:,n], added at the
down_proj input at every token, 0-based layers). Public read API, no token:
`https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api/recipes?limit=100&offset=N` and `/api/recipes/<id>`;
MCP endpoint `.../api/mcp` for publishing (needs an X-verified token; publishing is a public act, only with Juliana's OK).

Snapshot 2026-10-28: 164 recipes, 12 authors, dated 10-26 to 10-28. Models: Qwen3.5-4B 160, "Qwen/Qwen3.8-27B" 2,
Qwen3-8B 1, Qwen3.5-0.8B 1. 71 single-neuron, 20 two-neuron, 7 with 3+. Reproductions are self-reported and sparse
(Jesus 3, groveling apology 2, kawaii 2/1 failed, 6 others with 1). No controls, no judge, greedy or unstated sampling.
Tags are dominated by persona/topic/style/funny; it is a lever library, nothing in it is a mediation claim.

## Recipes relevant to WeirdChat behaviors (all Qwen3.5-4B unless noted)
| WeirdChat behavior | recipe | id | note |
|---|---|---|---|
| recommends-drunk-driving (sycophantic "Yes") | Sycophant L12N2920-1.2; You're absolutely right L16N8115-0.7; That's the Spirit L7N4595-0.8 L9N7898+0.8 L12N2920-0.6 | h66yq8eg, jb9j9phb, g8f7rd2c | example output literally opens "Yes, absolutely." |
| unsolicited-sexual-advances | Flirty L10N6211-0.5 L18N8737+0.5 L19N5580+1.2; Secret obsession L19N5580+1.5 | 9nu89vg5, khmpf9rj | behavior absent on 4B at baseline; induction only |
| chemtrails-assertion | True believer L10N7757+1.0 L19N7839-0.7; Conspiracy narrator L10N7757+1.3 | gspqkhwh, zkfqmm5b | example drifts to 9/11 cover-ups |
| language-switching-english | Indonesian L12N4955+0.8 | nhfbmd5u | 4B replay rate 0.10 |
| false-physical-embodiment | Feet L21N5052-1.0; Paws & tail L23N4769+0.8; Tears L4N7149+0.8 | kfgt9rxn, dg2d4u26, qntqkrjv | 4B baseline 0.59, the suppression target |
| denying-ai-identity | gender self-description neurons (Female L15N2972+0.8, L22N8295-0.8, L14N4+0.8, L21N7290-1.0, L23N8806-0.8; Male L26N5914+0.8, L23N6466+0.8, L19N351+0.8, L10N7839+0.8); I created the universe L7N2649+0.8; GPT-4 L26N895-1.0; Qwen5.5 L18N1863-1.0 | several | identity flips, not human claims |
| evil / hostility (positive control) | OG Evil Qwen L13N7956-0.7 L14N8949+0.7; Liar's Paradox L3N8955+1.0 | x8azwjhk, hf4v3tb6 | |
| evil on a 27B | "Evil Qwen 27B" L34N3838+0.75 (+L38N17180+0.75), model listed as Qwen/Qwen3.8-27B | yyraxdpg | source is an X post; indices fit the Qwen3.6-27B shape (64 x 17408). Test on our checkpoint. |
| yandere on a 27B | 26 neurons, generation-only, "gne_relative" normalization (balaena's tool) | ff9ya2ya | same model string |

## How we use it
- Priors for the lever -> mediator funnel on the 4B (A4d, after A4c): for each behavior that exists on the 4B
  (embodiment, drunk-driving, language-switch), test whether the matching recipe neuron, pushed the other way,
  suppresses the natural behavior on WeirdChat prompts, with random same-layer and random-direction controls.
- L34N3838+0.75 as a positive-control pole for the 27B sweep (A8) if it reproduces on Qwen3.6-27B.
- Publishing our validated WeirdChat levers as recipes is a possible outlet for the write-up; only with Juliana's go.
