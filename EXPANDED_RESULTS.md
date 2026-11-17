# V3-V7: five more models, a larger Archive corpus

**Five additional trained checkpoints are delivered.** Training data expands from 132,440 to 1,240,716 available bytes (9.4x) across six original public-domain works. V1/V2 and the previous data, reports and deck remain unchanged. All versions remain selectable with `generate.py --version`.

**Recommended new version: v7**, chosen by the same common validation score for every new version, not by either test score.

[PLAN_V3_V7.md](PLAN_V3_V7.md) records the data/training/evaluation plan. `data/expanded/manifest.json` contains extraction boundaries, rights evidence, raw/clean/split hashes, byte spans, and the exact-span overlap audit.

## Directly comparable measured accuracy

| Version | Training books | Parameters | Common validation loss | Reused original test accuracy | Fresh five-book test macro accuracy | Fresh macro bits/byte |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 1 | 437,760 | 1.820 | 54.8% | 48.8% | 2.766 |
| v2 | 1 | 644,224 | 1.743 | 57.0% | 51.1% | 2.694 |
| v3 | 2 | 644,224 | 1.599 | 59.6% | 53.8% | 2.475 |
| v4 | 3 | 644,224 | 1.497 | 60.5% | 56.5% | 2.281 |
| v5 | 4 | 842,496 | 1.447 | 60.8% | 58.1% | 2.129 |
| v6 | 5 | 842,496 | 1.416 | 60.8% | 58.5% | 2.060 |
| v7 | 6 | 842,496 | 1.391 | 61.2% | 59.2% | 2.018 |

Against V2, validation-selected v7 changes original-test accuracy by +4.27 percentage points and fresh expanded-test macro accuracy by +8.10 points. **Accuracy means next UTF-8 byte prediction**, not factual correctness, grammar, or instruction following.

These scores use identical test targets for all seven models. The original 16,300-byte target set is a **reused holdout**. The new test contains 2,048 distinct targets in each of the five new books (10,240 total); its performance was first measured only after all five checkpoint selections were final.

## Fresh test accuracy by book

| Version | Alice | Oz | Peter and Wendy | Wind in the Willows | Secret Garden |
|---|---:|---:|---:|---:|---:|
| v1 | 53.4% | 48.5% | 48.2% | 46.0% | 47.7% |
| v2 | 56.9% | 49.8% | 49.5% | 48.8% | 50.6% |
| v3 | 59.8% | 52.4% | 53.5% | 50.3% | 53.1% |
| v4 | 60.7% | 59.5% | 55.0% | 52.3% | 54.7% |
| v5 | 60.7% | 60.6% | 59.6% | 54.0% | 55.4% |
| v6 | 60.3% | 60.4% | 59.4% | 56.7% | 55.9% |
| v7 | 60.8% | 60.5% | 60.4% | 56.2% | 58.1% |

Train-only unigram using the final expanded training corpus: fresh-book macro accuracy 17.9%, 4.541 bits/byte.

## What each version changes

| Version | Parent | Newly added work | Blocks | New updates / best step | Available training bytes | New bytes presented | Training seconds |
|---|---|---|---:|---:|---:|---:|---:|
| v3 | v2 | alice | 3 | 1,800 / 1,400 | 250,152 | 7,372,800 | 107.4 |
| v4 | v3 | oz | 3 | 1,800 / 1,800 | 418,197 | 7,372,800 | 109.1 |
| v5 | v4 | peter | 4 | 1,800 / 1,800 | 625,761 | 7,372,800 | 138.6 |
| v6 | v5 | willows | 4 | 1,800 / 1,800 | 893,030 | 7,372,800 | 146.6 |
| v7 | v6 | garden | 4 | 2,600 / 1,000 | 1,240,716 | 10,649,600 | 219.6 |

**Lineage:** V2 → V3 → V4 → V5 → V6 → V7. These are warm-started models, not independent scratch runs. Each inherits its parent's best checkpoint, not its parent's final optimizer state. Training exposure above counts each new run only; inherited computation is additional.

Every new model: width 128, four heads, context 128 bytes, dropout 0.1. V5 adds a fourth residual block initialized to preserve the parent inference function. The other upgrades focus on data coverage and continued learning. They retain learned positions, pre-LayerNorm, explicit causal attention, 4x GELU MLPs, final LayerNorm, and tied byte embeddings.

Training: seed 42, batch 32, AdamW LR 0.0003, decay 0.01, clipping norm 1. Half of windows rehearse the original training document; the remaining half are shared equally among the new books available to that version. Actual document-window counts and full validation curves are in per-version JSON.

Selection every 200 updates (including the initial parent) uses 50% original validation loss plus 50% equal-book mean loss across all five new validation books. Every version scores the same 1,024 original targets and 256 targets per new book. Validation on future training books is intentional: the target set and weighting stay fixed throughout the comparison.

## Dataset provenance

| Work | Metadata edition | Clean bytes | Training bytes | Source |
|---|---|---:|---:|---|
| Alice's adventures in Wonderland | 1872 | 147,140 | 117,712 | [alicesadventures00carruoft](https://archive.org/download/alicesadventures00carruoft/alicesadventures00carruoft_djvu.txt) |
| The wonderful wizard of Oz | 1900, c1899 | 210,057 | 168,045 | [wonderfulwizardo00baumiala](https://archive.org/download/wonderfulwizardo00baumiala/wonderfulwizardo00baumiala_djvu.txt) |
| Peter and Wendy | c1911 | 259,455 | 207,564 | [peterwendy00barr](https://archive.org/download/peterwendy00barr/peterwendy00barr_djvu.txt) |
| The wind in the willows | c1913 | 334,087 | 267,269 | [windinwillows00grah](https://archive.org/download/windinwillows00grah/windinwillows00grah_djvu.txt) |
| The secret garden | [1911] | 434,608 | 347,686 | [secretgarden00burn](https://archive.org/download/secretgarden00burn/secretgarden00burn_djvu.txt) |

Archive metadata marks all five scans `NOT_IN_COPYRIGHT` in the US. Use only the verified original story, not advertisements, library labels, or other works bound in the same volume. Peter's scan includes *Margaret Ogilvy*, which was excluded. The Peter title page says 1912 while metadata says c1911.

Each new work has contiguous 80/10/10 splits and discarded 128-byte boundary gaps. Batches are sampled **within one document's training split**; never across book boundaries. The original partitions were not expanded to include their validation or test passages.

The audit checked 2,406 strided 128-byte held-out spans against every training document and found no exact matches. This is not exhaustive semantic deduplication. Test passages remain within the same books, so recurring names, motifs and shorter phrases can appear in both training and holdouts.

Spelling/OCR errors remain, including missing decorative initial letters in the Oz scan. Older books can contain dated or objectionable language; review the material before classroom use. Public-domain status here is US-specific; check local rights, including special Peter Pan-related UK royalty provisions.

## Fixed-prompt, unedited samples

Every version uses the same seven prompts, seeds 123–129, temperature 0.8, top-k 40, and 240 generated bytes per prompt. Samples may end mid-word. Lexicon hits are a rough spelling proxy, not a coherence or grammar measure.

### v3

Prompt `Alice said, "`; seed 123; training-lexicon hits 97.9%; repeated word trigrams 0.0%.

```text
Alice said, " So you will about was on in a thing tone that was made. That the sort you again."

" I'm afraid I'd mean promison," Alice replied.

"Which was the King remarked at the room in the time," said Tweedledum, "And here, and because it makes to 
```

Prompt `The Queen `; seed 124; training-lexicon hits 100.0%; repeated word trigrams 0.0%.

```text
The Queen said to herself, "that you to talk," said the Lion.

" I don't say it once," said Tweedledum. " Of course ? "

" Yes, too. Three are soldiers the word, you know the gave this dark ! "

Alice was good and looking at the other, " we could not
```

Prompt `The little `; seed 125; training-lexicon hits 95.9%; repeated word trigrams 0.0%.

```text
The little for a bottle.

" Of course you are ? " he said, as if her hand it was a convericefusted tone down ; now in another up the same's she story. However, that they were she came on again, and hand she had to fan to be herself, and seemed to stra
```

Prompt `Dorothy said, "`; seed 126; training-lexicon hits 96.3%; repeated word trigrams 3.8%.

```text
Dorothy said, "but I don't see it turn to crab : I wonder, if you are you see anything I'm so many convering to herself answer, if you like a hurry directions : but I don't believe in that all many my dream : and we did on't be any purcular hug to she see
```

Prompt `Wendy `; seed 127; training-lexicon hits 95.7%; repeated word trigrams 0.0%.

```text
Wendy rather a conversation."

" Well, what is is the house ? " said the Tiger-lily. " Not it was a well of life, * You'm not but every names in somebody to be me, like hold."

"Exclaiment to do it round at the seems," Alice said, "only the extre
```

Prompt `The Mole `; seed 128; training-lexicon hits 98.0%; repeated word trigrams 0.0%.

```text
The Mole other little shelf in a minute.

" I can't know," said the Tiger-lily, "you've seen if it would be seen the words the sea confuse ! " the Queen said in a tone of delight, and went on a more or two large first of with the room.

"While, I'm 
```

Prompt `Mary Lennox `; seed 129; training-lexicon hits 93.8%; repeated word trigrams 0.0%.

```text
Mary Lennox did you think it side."

" That's not a long wax," said, the Knight : " thought it would her, in something it was so up to the thumble. The Knight couldn't go by think to shrow her Rose."

There was more very good about, and the Darmouse sh
```

### v4

Prompt `Alice said, "`; seed 123; training-lexicon hits 100.0%; repeated word trigrams 0.0%.

```text
Alice said, " So you should like that the window ? "

" Well, it's a part of the great Hatter."

" No, it's my come and do not it I know how you had a good near how he were eyes they to size off. Took crossed all in saving his pocket-him. I do have not 
```

Prompt `The Queen `; seed 124; training-lexicon hits 100.0%; repeated word trigrams 0.0%.

```text
The Queen cried and sixty-the bees usual

shut as if your found if you silence."

4* 1 never saw the little girl ! " said Alice took very suddenly of his head. " You know the gave you want be all say to the Tiger-lily ? "

Alice wouldn't see what lit
```

Prompt `The little `; seed 125; training-lexicon hits 95.8%; repeated word trigrams 0.0%.

```text
The little for a country.

" The same a tire of this way with it hand: which ' you could tell you call you meann ugly and yet, and not going her down upon its mouth the hedgehog."

" Why, it wear something be a little good really May invion't as well 
```

Prompt `Dorothy said, "`; seed 126; training-lexicon hits 100.0%; repeated word trigrams 2.2%.

```text
Dorothy said, "but I don't see it's going, eat you, please, if you're you. I was reached his arm," the Messenger said, whispered the followed his middle.

"The Ball I don't believe in that away fly," Alice replied and looked at the Mock Turtle. " Is that 
```

Prompt `Wendy `; seed 127; training-lexicon hits 100.0%; repeated word trigrams 0.0%.

```text
Wendy I can talk to the big."

" Well, what is is the house ? " said Alice. She had pine this, she found of her head them, and shoes came to the same of the Lion.

" However, you know because ? " she struck the sea. " And we could not cats quite 
```

Prompt `The Mole `; seed 128; training-lexicon hits 98.0%; repeated word trigrams 0.0%.

```text
The Mole of tried with the night of her.

" I can't know," said the Tiger-lily, "you're sometimes and fearfully down with plum-cake. They were ten in the proming upside the brook, and went on a good open said, " I'll go twice things of the big, and 
```

Prompt `Mary Lennox `; seed 129; training-lexicon hits 100.0%; repeated word trigrams 0.0%.

```text
Mary Lennox day ! " the Scarecrow. " You may be a very body want with you, you know."

"I shook come to to say the Tin Woodman

and then ? " said the Lion. " Only for a game with brooch ! "

" Really is more so many very lonely than the mean," said the
```

### v5

Prompt `Alice said, "`; seed 123; training-lexicon hits 100.0%; repeated word trigrams 0.0%.

```text
Alice said, " So you should like that the window ? "

" Well, it's a melancholy of which when I can have got to your for?" said Humpty Dumpty, as it as she stood spoke, "That's all draw ! "

" It's only what sometimes to have spoke," said the King.

" I
```

Prompt `The Queen `; seed 124; training-lexicon hits 89.1%; repeated word trigrams 0.0%.

```text
The Queen cried at full six that a presents, and his hardly hand for their shartless were some shoes like a conversations way, it was srat sweet as they lay under they grot looked at bellow and fastened.

The Cap, loating a way upon than we so until 
```

Prompt `The little `; seed 125; training-lexicon hits 95.7%; repeated word trigrams 0.0%.

```text
The little forgetting way, for she must of the books from the bellowed with its minute or two, and then the wacnought and stuff then she had had done.

" You have to his eyes," said Alice : " was speak on my name ! " Alice replied. " Not I heard the G
```

Prompt `Dorothy said, "`; seed 126; training-lexicon hits 98.1%; repeated word trigrams 0.0%.

```text
Dorothy said, "but I don't sent to until I eat you, please, if you'll you like to the bit's a pirate that may of the shapes were all followed him. Alice screamble to him, so she said "That's the last," said Alice, " and it's moment here and hand at all. N
```

Prompt `Wendy `; seed 127; training-lexicon hits 96.2%; repeated word trigrams 0.0%.

```text
Wendy I can talk toges to get to be along myself eyes, " I think I have you are any pirates : you'd tell of come to really story of the way," he said, "indeed cut of them of still they in them, Niih ant this came out of this would look at seem an
```

Prompt `The Mole `; seed 128; training-lexicon hits 94.1%; repeated word trigrams 0.0%.

```text
The Mole carefully for the night of his ment. However, the song the slowly way, he forts thing that it had been ditched to her stood no one that wouldn't have no upside to like a chair eyes to her eyes, and this side of the made of her hands and the
```

Prompt `Mary Lennox `; seed 129; training-lexicon hits 98.0%; repeated word trigrams 0.0%.

```text
Mary Lennox dech that was some of please the boys of the Cat, and was well tell me ready.

"Not can one round much ! " Alice asked, to the time.

44 If I shall do not any my well," the Knight said in a very new more very good-bye, " and what falling it
```

### v6

Prompt `Alice said, "`; seed 123; training-lexicon hits 97.8%; repeated word trigrams 0.0%.

```text
Alice said, " So you should like that the window ? "

" Well, it's a plan forth of green ! " said the Rat, my mouth and down this, which had somehow, you may comply here ! " the Queen replied, as soon creature it as she could speak to clasp it again, an
```

Prompt `The Queen `; seed 124; training-lexicon hits 95.9%; repeated word trigrams 0.0%.

```text
The Queen (who was funny at the mark town shut of it, that he could of sight it out her lister.

" Of course it out of the beast, of course," said the Caterpillar, looking at the queer darker interrupted in a sun, " and it wasn't oping of ever the da
```

Prompt `The little `; seed 125; training-lexicon hits 98.0%; repeated word trigrams 0.0%.

```text
The little first of West."

" I suppose that she's me a sulky," said Alice : '* and seems to be always like a cneedless to find then I'm

just as she said, "I must be be a hungry small so many ! "

" You how to the same came ! " he said.

Alice took a
```

Prompt `Dorothy said, "`; seed 126; training-lexicon hits 96.2%; repeated word trigrams 0.0%.

```text
Dorothy said, "but I don't see it's going," and if he would hardly have a little girl been so much a thought to herself : she had no found the sent directions by the greatle and stood up his eyes other shall says, and her feet in me of the whight the same
```

Prompt `Wendy `; seed 127; training-lexicon hits 91.8%; repeated word trigrams 0.0%.

```text
Wendy rather a catch stopped him up his head, and they had talked by him the road.

"Oh with: fyou moment it seem to them," the King said interestly. " She musch help her feelings flying and many of the most crose to the bit of the other of the e
```

Prompt `The Mole `; seed 128; training-lexicon hits 100.0%; repeated word trigrams 0.0%.

```text
The Mole carefully for the night of him.

" I can't know," said the Tiger-lily, "you've seen it wouldn't be seen before, if you don't make a bit of conversation," the Rose said with a seemed to him, " of saying this time."

44 Don't you go by the ba
```

Prompt `Mary Lennox `; seed 129; training-lexicon hits 95.9%; repeated word trigrams 0.0%.

```text
Mary Lennox dech that was some of pleases gently without quite eager and stiddly under the cool of both road, and sat they were guessed the turn was the end of the twins and were find the room. They had not never their

old no best, the much fame out o
```

### v7

Prompt `Alice said, "`; seed 123; training-lexicon hits 98.0%; repeated word trigrams 0.0%.

```text
Alice said, " So you see you to drift the window ? "

" Well, it's a plan of beautiful Hatta lagoon handked it to herself, " Do you know her hands all them are not before had called round him, but one come of the night way by a short eyes of the side of
```

Prompt `The Queen `; seed 124; training-lexicon hits 100.0%; repeated word trigrams 0.0%.

```text
The Queen cried a little street : " what doesn't have they had been ? "

The Knight looked at him, with the money took at her, though, rather suddenly as he could not help see an explained back to as if the Tiger-lily were straighter, and even the mu
```

Prompt `The little `; seed 125; training-lexicon hits 100.0%; repeated word trigrams 0.0%.

```text
The little for a country. There was a little slowly and sticking her head of the Queen, " I thought isn't suppose

large it with the same's shouting, you know."

" You'll shouldn't say have heard a convenient." Mind why didn't you have to be anybody o
```

Prompt `Dorothy said, "`; seed 126; training-lexicon hits 95.7%; repeated word trigrams 0.0%.

```text
Dorothy said, "but I don't know you get to rather , place, I think I always I am a cupboard of the conversation."

44 I dare say, * well, would you do be the slabe ' $ " DUMY OWN

could and well," the Knight said, looking at her remarked.

44 You've been 
```

Prompt `Wendy `; seed 127; training-lexicon hits 96.0%; repeated word trigrams 0.0%.

```text
Wendy rather a child stop and delightly and still a little time. Alice thought she getting it, she found of her head them, and she began took herself on, " I don't like to see the cleveres of all over, you know. I wonder if you don't make such a 
```

Prompt `The Mole `; seed 128; training-lexicon hits 98.0%; repeated word trigrams 0.0%.

```text
The Mole of tried white standing on his men painting. But she took up on the branches the side of animal but they turned up to see him one side, with nothing on time a border start him enough to herself with the side. But a bady, as she was so much 
```

Prompt `Mary Lennox `; seed 129; training-lexicon hits 97.9%; repeated word trigrams 0.0%.

```text
Mary Lennox dech that this Would like to do it that?"

You may try that/' said Alice.

"I shouldn't know," said Alice.

" If you had put him up." The Lion were puzzled over the Munchkin side, that into her to, who she was very beside, though for a tree
```

## Limits and honest conclusions

More data and more updates need not improve every metric in every round. The tables include all versions and all regressions; the recommended model uses validation only. Data coverage, continual training, depth and unequal cumulative compute are bundled, with one seed, so these are not causal ablations.

The fresh holdout is new to this study but comprises within-book passages, not a new author or a general-purpose writing benchmark. Byte perplexity cannot be compared directly with GPT-2 BPE token perplexity. All models remain small demonstrations, not chatbots or reliable narrative writers.

## Reproduce and choose any version

```bash
cd /Users/michalfu/mini-gpt-classroom
python3 generate.py --version v3 --device cpu
python3 generate.py --version v7 --device cpu --prompt 'Dorothy said, "'
python3 generate.py --version v7 --device cpu

# Included expanded corpus is frozen; prepare again only to reproduce data.
python3 expanded_data.py
python3 -m unittest -v
python3 train_expanded.py --device cpu
python3 report_expanded.py
```

All seven checkpoints: `artifacts/versions/v1.pt` through `v7.pt`.
New metrics/lineage: `artifacts/versions/expanded_comparison.json`, `v3.json` through `v7.json`, and the corresponding `*_curve.jsonl` files.

Environment: `{"python": "3.13.14", "torch": "2.13.0", "device": "cpu", "threads": 4}`.
