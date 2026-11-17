# Actual training results

**Selected: r4_train_longer**, 437,760 parameters, 2 blocks, 128-wide embeddings, 64-byte context. Baseline-to-best validation cross-entropy decreased 36.6%.

All six experiments were actually trained from scratch. The following table uses each run's best validation checkpoint; `best step` can differ from total updates.

| Round | Change | Updates / best step | Parameters | Train loss | Val loss | Val bits/byte | Val byte PPL | Val top-1 | Training seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r0_baseline | Baseline: one small block, 200 updates | 200 / 200 | 68,544 | 2.403 | 2.558 | 3.690 | 12.910 | 29.2% | 0.6 |
| r1_more_training | Increase updates from 200 to 600 | 600 / 600 | 68,544 | 1.989 | 2.139 | 3.086 | 8.494 | 40.3% | 1.8 |
| r2_longer_context | Increase context from 32 to 64 bytes | 600 / 600 | 70,592 | 1.946 | 2.177 | 3.141 | 8.824 | 39.5% | 2.6 |
| r3_more_capacity | Increase width to 128 and depth to two blocks | 600 / 600 | 437,760 | 1.636 | 1.889 | 2.725 | 6.610 | 46.4% | 8.3 |
| r4_train_longer | Increase updates from 600 to 1800 | 1800 / 1800 | 437,760 | 1.078 | 1.622 | 2.339 | 5.061 | 54.8% | 24.8 |
| r5_scheduled_lr | Add 100-step warmup and cosine learning-rate decay | 1800 / 1800 | 437,760 | 1.158 | 1.641 | 2.368 | 5.162 | 53.5% | 24.6 |

## What improved, and what did not

- More training helped the small model (R1); increasing context alone did not (R2).
- More capacity helped (R3), and training that model longer helped again (R4).
- Warmup/cosine decay (R5) was slightly worse than R4 at this budget. Retain the validation-selected R4, not simply the last run.
- This satisfies baseline plus five measured improvement **attempts**, not five guaranteed or monotonically successful gains.

## Final held-out test

The test file was first loaded after all rounds and validation-based selection. Every test target after the first 128 bytes was scored once. Test scores did not determine the selected checkpoint or any training settings.

| Model | Targets | Loss (nats/byte) | Bits/byte | Byte perplexity | Top-1 next-byte accuracy |
|---|---:|---:|---:|---:|---:|
| Selected GPT | 16,300 | 1.628 | 2.349 | 5.094 | 54.8% |
| Train-only smoothed unigram | 16,300 | 3.165 | 4.566 | 23.679 | 18.8% |

Unigram probabilities use training byte counts with add-one smoothing. A uniformly random predictor has loss ln(256) = 5.545 and 8 bits/byte. Perplexity is exp(loss), bits/byte is loss/ln(2). These are **not** GPT-2 BPE metrics.

## Data, compute, and reproducibility

- Corpus: [Through the looking-glass, and what Alice found there](https://archive.org/download/throughlookinggl00carruoft/throughlookinggl00carruoft_djvu.txt), Archive item `throughlookinggl00carruoft`, edition `[1899?]`, status `NOT_IN_COPYRIGHT` (US).
- Cleaned corpus: 165,551 bytes. Splits: {"train": 132440, "validation": 16427, "test": 16428}; 128-byte boundary gaps.
- Cleaned SHA-256: `f7928c0cf3588e0c5721ffb8f98529a651e5d585f9e7154c175cd8a3e2551e7f`.
- Every run: seed 42, batch 32, AdamW, LR 0.001, weight decay 0.01, gradient clipping at norm 1. R5 warms up over 100 updates and decays to 0.0001.
- Validation: 1,024 fixed distinct next-byte target offsets, seed 2026; each model conditions on its configured context. The scored target set is identical, not the context length or compute budget.
- Checkpoint selection every 200 updates; training diagnostics use 512 fixed training targets. Curves and per-round JSON are in `artifacts/`.
- Environment: `{"python": "3.13.14", "torch": "2.13.0", "platform": "macOS-27.2-arm64-arm-64bit-Mach-O", "device": "cpu", "threads": 4}`.
- Measured benchmark (40 steps, 2-block/128-width/64-context, batch 32): CPU 0.0141 seconds/step; Apple MPS 0.0645 seconds/step. CPU was chosen for the recorded runs. Training times exclude sample generation and final test.

| Round | Training bytes presented (including repeated windows) | Approx. corpus passes |
|---|---:|---:|
| r0_baseline | 204,800 | 1.5 |
| r1_more_training | 614,400 | 4.6 |
| r2_longer_context | 1,228,800 | 9.3 |
| r3_more_capacity | 1,228,800 | 9.3 |
| r4_train_longer | 3,686,400 | 27.8 |
| r5_scheduled_lr | 3,686,400 | 27.8 |

## Uncurated fixed-seed samples from every round

All prompts and 240 generated-byte budgets are identical across rounds. Temperature 0.8, top-k 40, seeds 123/124/125. These are recorded model outputs, not edited prose. Samples can terminate mid-word.

The lexicon hit rate counts generated ASCII word runs that appear in training. It is a rough spelling proxy, rewards short/common words, and can increase with memorization. Repeated word-trigram fraction is a crude repetition flag. Neither measures grammar, factuality, or narrative coherence.

### r0_baseline

Prompt 1; seed 123; lexicon hit 48.3%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
Alice said, " ! 'The s say the bre an tin we aly s d he " thanitond f anal t thang w le arid oug s lound ghm I ty werer d  lyouint t selphd thit hind y ou, he wn s "hinisThan ce thed the ig th thinous ce s The lduid s the thed tous tbed tove the m poug 
```

Prompt 2; seed 124; lexicon hit 46.4%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The Queen t talis aikin s tht they t topu shin ashid y hothe Hin f tofis s shicto s he oucang as t wald, we oon o t he thiwhe, t ooul, r wakelis t on hanled und, koling  fth than quguced ad anghed ay t it thawouooullld tid t thius, w wan we tout t st
```

Prompt 3; seed 125; lexicon hit 48.3%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The little y hthe wher ay and the ame ke Que ol ooas ce oul thin her hen w thex: "


" ove " ca t ceedned t whengut,an here wi, t'tn n'g
'to he t d she tou s" thsat the he heveng s ! " s " " he-athey theshing l tond wouoke th t the  ind cedou antt t t
```

### r1_more_training

Prompt 1; seed 123; lexicon hit 54.7%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
Alice said, " she I gring ate br she hand a they sto he " I was oned. " and it and wy mas to ought lhe hegrmme tham and do knoting itse puple it his, you forr ones ear it hand and edree sim to thisue ble she saidd was eand herspe shebed to wer hemppst l
```

Prompt 2; seed 124; lexicon hit 48.1%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The Queen tro wer : une sat thelly u to ther," and thy kothe wing too if said at aste boucte sar toe lddo ea wnas to to could, it oulver the wishe of has and if the her the lot very uump anke it anay und it awnd ould ousid she iupte thand eet the lit
```

Prompt 3; seed 125; lexicon hit 54.7%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The little for a cull way," wen the cor ourd on oof  cherte thing, " hen in it : "

" It thelly kne tone it wecnt.


" It if iu to nin's 'to hers domy woong "


"It the he hryou the be so rouge sae you ont he lllong anto bing an he finge on't and sthe
```

### r2_longer_context

Prompt 1; seed 123; lexicon hit 43.4%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
Alice said, " son the sany the bruid had the aly she said il a ompeaf alally but on y mas id ough sthe he dmit tham a ppriter : Hon't you don ith ith you manthen sailly. Thand to hedr a sid the sifore crea mes ittille ea veny said itbectto in have pof l
```

Prompt 2; seed 124; lexicon hit 52.9%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The Queen troully piking at thouly u to the up as it yourthe nin for? is said at asthe oucan qusaioutly, weas tho the said a ist oo lonr the wisth of hatllly nof you you lthe : and dobed a bruthe say und itt.


"Eillght it she iupten was we so the mu
```

Prompt 3; seed 125; lexicon hit 55.8%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The little for a out she and she ta coke Queen. " I vereraled in hermppoowng exkittugh ove : an the yon dot wacnly lan thiseng, then sary't she sid she tor said said he as hy she say." said Ale-sasty a ond he b ton hat ombing at he you and hisand sthe
```

### r3_more_capacity

Prompt 1; seed 123; lexicon hit 58.7%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
Alice said, " said Alice. Hall drumpt turthen, you do bectiles oment. " I " wehat no loose doong stoo nothim I that's puck it : Ho they phing it his, you mant wass ! " Alice to It got theim to this ore wen The imumpted Alice said. " becaling the not to 
```

Prompt 2; seed 124; lexicon hit 52.1%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The Queen trough tought says ! " aid to jest," said ther to his if to ifirst imtomsten ouch, quicke ald to fight ! " said Alice too loor the wishe only quace if the has tot loook a goot asher hiss alll

to bawousE to old the was aye thand ever up lad
```

Prompt 3; seed 125; lexicon hit 68.8%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The little for a began ! "

" Alice couldn't slooking are linting, " Alice to Alice oliove : conte a deedd awacn'ut, and if the twon thro's she say. you'd it helse for the hras tremever but he ! "

" You Alice. Mongs byoce the fight is a I his wenly i
```

### r4_train_longer

Prompt 1; seed 123; lexicon hit 88.9%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
Alice said, " Sof I could stand way that's got in the was just in surpal so did nothing sordon, in then Sheed me all and of you me with phim sigh it, you may hands ear it had called round his a sighed, went he led was great to to shower, and she was go 
```

Prompt 2; seed 124; lexicon hit 89.6%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The Queen proud at further.

" I amputy never more, and at her middle in she brooks.

"The passs ofet But I conten it stood, is it seemed a whispered hands out first the for of her gust are all just tumble always, larging a tree, things feet of her d
```

Prompt 3; seed 125; lexicon hit 79.2%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The little for a little path rough the wood, of course, twiculle her her in it, and here heaped to herself, with us, and it was toon sigh' wa all dried to it, and there he haventer saying a littled banking his hands, but behind the King creally, was a
```

### r5_scheduled_lr

Prompt 1; seed 123; lexicon hit 75.0%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
Alice said, " I don't sany you have a turn it, you know."

That misself all obout not make to the floud, because asken," she said, insectly this high one for a must hrough and to her the inventides of crwarm sile?" said Alice.

" You've a birthd.

"I sh
```

Prompt 2; seed 124; lexicon hit 91.7%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The Queen cross at fun one other air to them, and it crying and if they is she to stuon the mean off loud, and so that the began openerring when you know."

And the Unicorn lot seemelbody her himself under always, looking at this that, so ever the mu
```

Prompt 3; seed 125; lexicon hit 83.7%; repeated trigram fraction 0.0%; UTF-8 replacement characters 0.

```text
The little for a little panting to me ! " Alice said in a little her hand of them turning !"

444 Here had a ment.

But the with the sarv'y shoutid ? " the Unicorn, the kedd here smove her. Ahow as you long. " It's a looked her fall on and his stoop o
```

## Is this good enough for students?

**Yes, as a transparent language-model demonstration; no, as a general-purpose text generator.** The selected two-block model beats a train-only unigram predictor on completely held-out target bytes and visibly learns words, dialogue punctuation, names, and short English-like clauses.

The generated text still misuses words, misspells them, repeats fragments, and loses narrative meaning. For example, the uncurated R4 sample starts "The little for a little path rough the wood, of course" but soon becomes ungrammatical. Teach this failure rather than presenting the output as coherent.

The train/validation gap shows overfitting risk. Only one OCR book, one seed, and short context were used. Validation is reused for checkpoint/model selection; the reported validation scores are not independent confirmations. Contiguous splits prevent overlapping windows, not repeated phrases within the same book. No pretrained GPT-2 weights, web crawl, instruction tuning, or assistant alignment are included.

## Sources

- https://github.com/openai/gpt-2/blob/master/src/model.py
- https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- https://archive.org/details/throughlookinggl00carruoft
- https://archive.org/metadata/throughlookinggl00carruoft
