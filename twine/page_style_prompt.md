# Twine ML Topic Page — Style & Content Guide

Use this prompt to generate full Twine/SugarCube passages for ML topics, matching the style and structure of the PCA page.

---

## Structure (in order, all sections required)

```
<<include "Header">>

! {FULL_TITLE}

<img src="assets/{SVG_STEM}.svg" alt="[descriptive alt text]" style="width:100%;max-width:620px;display:block;margin:16px auto 24px auto;">

IMPORTANT: You MUST create the SVG file before writing the passage. Every topic requires its own SVG diagram.

**SVG creation steps:**
1. Design a clean, minimal SVG diagram that illustrates the core concept of the topic.
2. Save it to `assets/{SVG_STEM}.svg`.
3. Then write the passage with the `<img>` tag pointing to it.

**SVG style guidelines** (match the existing assets):
- Viewbox: `viewBox="0 0 800 400"` (or similar landscape ratio)
- White or transparent background
- Simple shapes: circles, rectangles, arrows, lines — no photos or raster content
- Label key elements with `<text>` elements
- Use a limited color palette (2–4 colors max)
- Keep it readable at 620px wide

[Opening paragraph: 3–5 sentences. Concrete real-world analogy. No jargon yet. Hook the reader.]

---

!! The Core Insight
[Intuitive, geometric, or narrative explanation. Build the mental model before any math. 3–5 paragraphs.]

---

!! [Second conceptual section — topic-specific descriptive title]
[Deepen the intuition. Explain *why* something works, not just that it does. 2–4 paragraphs.]

---

!! What the Math Is Actually Doing
[Introduce notation and equations. Explain each symbol in plain terms alongside the math. Use $inline$ and $$display$$ LaTeX. 3–5 paragraphs.]

---

!! [Practical section — e.g. "Choosing Hyperparameters", "Reading the Output", "When to Use This"]
[How to actually use this in practice. Rules of thumb, tradeoffs, common thresholds. 2–4 paragraphs.]

---

!! When {SHORT_TITLE} Breaks Down
[Exactly 3 specific failure modes. For each: what goes wrong, why, what to use instead.]

---

!! Technical Details
[Dense reference-style section. Cover: algorithm steps, complexity, 2–3 notable variants/extensions, key theorems or results. Use ''bold'' for term introductions.]

---

!! Quiz

[7–10 numbered multiple-choice questions. Mix conceptual ("explain why") and technical ("walk through the algorithm").
Each question has exactly 4 options (A–D), one correct. Use this EXACT widget pattern:]

''N. Question text?''
<<button "A) First option">><<replace "#{ID_PREFIX}-qN">>'✗ Wrong. Brief explanation of why A is incorrect.'<</replace>><<script>>typesetMath();<</script>><</button>>
<<button "B) Second option">><<replace "#{ID_PREFIX}-qN">>'✓ Correct! Full explanation of why B is right (2–4 sentences).'<</replace>><<script>>typesetMath();<</script>><</button>>
<<button "C) Third option">><<replace "#{ID_PREFIX}-qN">>'✗ Wrong. Brief explanation of why C is incorrect.'<</replace>><<script>>typesetMath();<</script>><</button>>
<<button "D) Fourth option">><<replace "#{ID_PREFIX}-qN">>'✗ Wrong. Brief explanation of why D is incorrect.'<</replace>><<script>>typesetMath();<</script>><</button>>
<span id="{ID_PREFIX}-qN"></span>

[The correct answer should vary across questions — don't always put it as the same letter. Distractors should be plausible, not obviously wrong.]

<<include "Footer">>
```

---

## Writing Style Rules

- **Analogy first, math second.** Every concept gets a concrete real-world framing before equations appear.
- **Explain the "why."** Don't just say what something is — say why it matters, what problem it solves.
- **Short punchy sentences** mixed with longer explanations. Use "This is the whole idea." type summary sentences after a key insight.
- **American spelling** — visualization, standardize, centered, behavior, optimize.
- `''double apostrophes''` for **bold** key terms when first introduced.
- `//double slashes//` for *italics/emphasis*.
- `$...$` for inline LaTeX, `$$...$$` for display equations on their own line.
- `[[Link Text|Passage Name]]` for internal Twine links when referencing related topics.
- The **Technical Details** section is deliberately denser — compressed, reference-style, covers algorithm, complexity, and 2–3 notable variants.
- **Quiz** questions are multiple choice (A–D), one correct answer. Wrong answers show a brief explanation of why they're wrong. The correct answer gives a full 2–4 sentence explanation. Distractors must be plausible.
- Output the raw passage text only — no commentary outside the passage.

---

## Substitution Variables

| Variable | Description | Example |
|---|---|---|
| `{FULL_TITLE}` | Full display name for H1 | `Linear Regression` |
| `{SVG_STEM}` | SVG filename without extension | `linear_regression` |
| `{SHORT_TITLE}` | Short name for "When X Breaks Down" | `Linear Regression` |
| `{ID_PREFIX}` | Lowercase, no spaces, for quiz span IDs | `lr` |

---

## Available Topics & SVG Stems

| Passage Name (in HTML) | SVG Stem | ID Prefix |
|---|---|---|
| Linear Regression | linear_regression | lr |
| Logistic Regression | logistic_regression | logreg |
| Loss Functions | loss_functions | loss |
| Regularization | regularization | reg |
| Random Forest | random_forest | rf |
| CNN | cnn | cnn |
| LSTM | lstm | lstm |
| Attention | attention | attn |
| Bayesian Methods | bayesian | bayes |
| GAN | gan | gan |
| Embeddings | embeddings | emb |
| Cross Validation | cross_validation | cv |
| Transfer Learning | transfer_learning | tl |
| Backpropogation and Chain Rule | backprop | bp |
| KMeans | kmeans | km |
