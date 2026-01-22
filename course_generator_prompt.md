# Course Website Generator Prompt

Use this prompt to generate a complete, professional course website with modular content for any subject. The framework is designed to be institution-agnostic and fully customizable.

---

## PROMPT

```
I need you to create a complete static course website. Generate all files for a professional, minimalist course site with the following specifications:

## COURSE INFORMATION

- **Course Code:** [YOUR_CODE]
- **Course Title:** [YOUR_TITLE]
- **Institution Name:** [YOUR_INSTITUTION]
- **Number of Modules:** [NUMBER]
- **Password for Site Access:** [PASSWORD] (or "none" for no password protection)

## MODULE THEMES

For each module, provide:
1. Module number
2. Module title
3. Short description (1 sentence)
4. 3-6 learning objectives
5. 5-8 key terms with definitions
6. 3-5 quiz questions with answers and explanations
7. 2-4 discussion topics
8. 2-5 recommended readings (titles, authors, sources)
9. List of prerequisite concepts students should know

### Module List:
[Paste your module list here in this format]

Module 1: [Title]
- Description: [One sentence]
- Key concepts: [comma-separated list]

Module 2: [Title]
- Description: [One sentence]
- Key concepts: [comma-separated list]

[...continue for all modules...]

## ADDITIONAL PAGES (optional)

Include any of the following supplementary pages:
- [ ] How to Read Research Papers guide
- [ ] Capstone/Final Project Best Practices
- [ ] Curated External Resources page (with search/filter)
- [ ] Office Hours / Contact Information
- [ ] Course Syllabus
- [ ] FAQ

## DESIGN PREFERENCES

- **Primary Color:** [hex code, e.g., #1a365d for deep navy]
- **Accent Color:** [hex code, e.g., #2b6cb0 for professional blue]
- **Font:** [e.g., Inter, Roboto, or system fonts]
- **Style:** Professional, minimalist, academic

---

## FILE STRUCTURE TO GENERATE

Please generate the following structure:

```
/
├── index.html                 # Main course homepage with module grid
├── assets/
│   ├── styles.css            # Global stylesheet with CSS variables
│   └── quiz.js               # Interactive quiz functionality
├── [supplementary pages].html # Any additional pages requested
└── week_XX/                   # One folder per module
    ├── index.html            # Module landing page with learning objectives
    ├── glossary.html         # Key terms and definitions
    ├── quiz.html             # Knowledge check with instant feedback
    ├── readings.html         # Recommended readings list
    ├── discussion.html       # Discussion topics for class
    ├── lecture.py            # Code examples (if applicable)
    ├── README.md             # Module summary in markdown
    ├── exercises/            # Practice exercises folder
    ├── audio_script.md       # Source text for audio: pure spoken text only
    ├── slides.tex            # LaTeX source for slides (Zen theme)
    ├── slides.json           # Timing and transcript data for player
    └── slides/               # Directory containing generated slide images (PNG)
        ├── slide-01.png
        └── ...
```

---

## TECHNICAL REQUIREMENTS

1. **Pure HTML/CSS/JS** - No build tools or frameworks required
2. **Mobile-responsive** - Works on all screen sizes
3. **Accessible** - Proper semantic HTML, ARIA labels where needed
4. **Self-contained** - Only external dependency is Google Fonts
5. **Git-ready** - Include .gitignore for common patterns

### Quiz System Features:
- Multiple choice questions with immediate feedback
- Detailed explanations for correct AND incorrect answers
- Visual indicators (green/red) for answer states
- Reset functionality to retake quizzes

### CSS Architecture:
- Use CSS custom properties (variables) for theming
- Mobile-first responsive design
- Consistent spacing scale (xs, sm, md, lg, xl, 2xl)
- Print-friendly styles

### Password Protection (if enabled):
- Simple localStorage-based authentication
- Single password for entire site
- Persists across browser sessions
- Clear visual login interface

---

## CONTENT QUALITY GUIDELINES

### Glossary Entries:
- Each term should have a 3-5 sentence definition
- Explain WHY the concept matters, not just WHAT it is
- Reference related concepts from the same or other modules
- Use concrete examples where helpful

### Quiz Questions:
- Test conceptual understanding, not memorization
- Include plausible distractors that address common misconceptions
- Provide educational explanations for ALL answer options
- Mix difficulty levels within each module

### Discussion Topics:
- Open-ended questions that encourage critical thinking
- Connect module concepts to real-world applications
- Include ethical, practical, and theoretical dimensions
- Suitable for small group or full-class discussion

### Learning Objectives:
- Use action verbs (Define, Identify, Apply, Evaluate, etc.)
- Be specific and measurable
- Cover different levels of Bloom's taxonomy
- 4-6 objectives per module is ideal

---

## AUDIO & SLIDE GUIDELINES

### Audio Transcripts (audio_script.md):
- **CRITICAL:** Scripts must be **pure spoken text only**. Do NOT include:
  - Markdown headers (`# Title`, `## Section`)
  - Metadata blocks (`**Course:**`, `**Duration:**`, etc.)
  - Section markers or timestamps
  - Speaker labels (`**[Narrator]:**`)
- These elements will be read aloud by the TTS engine if included.
- Each paragraph must be on a single line (no hard line breaks within a paragraph) for correct HTML display.
- Use explicit `<break time="1.0s" />` tags for pauses.

### Synced Player Features:
1. **Synced Playback:** Listen for `<audio>` `timeupdate` events (~4Hz). Loop through `SLIDES_DATA` to find the slide where `currentTime` falls between `startTime` and `endTime`.
2. **CORS & Data Loading:** Embed `slides.json` directly into `index.html` as `window.SLIDES_DATA` to allow local file playback (avoiding CORS issues).
3. **Image Preloading:** Attach `onload` event listeners *before* setting `img.src` to prevent flicker.
4. **Transcript Auto-Scroll:** Highlight active segments and automatically scroll them into view.
5. **Responsive Layout:** Flex-column layout stacking Slide, Audio Controls, and Transcript vertically.

---

## EXAMPLE MODULE INPUT

Module 1: Introduction and Framing
- Description: Introduces foundational concepts including functions, models, and the importance of generalization.
- Key concepts: Function, Model, Generalization, Overfitting, Train/Test Split, Baseline

Module 2: Linear Regression and Gradient Descent  
- Description: Covers loss functions and optimization methods including batch, stochastic, and mini-batch gradient descent.
- Key concepts: Loss function, Mean Squared Error, Gradient, Learning rate, Convergence

[...etc...]

---

## OUTPUT INSTRUCTIONS

1. Generate ALL files completely - do not use placeholders like "add content here"
2. Ensure consistent branding/naming across all pages
3. All internal links should work (use relative paths)
4. Include the full quiz.js with all interactive functionality
5. Make the CSS comprehensive with all component styles
6. Footer on every page should link back to course home
```

---

## TIPS FOR BEST RESULTS

1. **Be Specific with Module Content:** The more detail you provide about each module's key concepts, the better the generated glossaries and quizzes will be.

2. **Review Generated Quizzes:** AI-generated quiz answers should be verified for accuracy before deployment.

3. **Customize After Generation:** The generated site is a starting point. Add your own lecture code, exercises, and readings.

4. **Brand Consistency:** Choose colors that match your institution's brand guidelines.

5. **Iterate by Module:** For large courses, you can generate modules in batches and combine them.

---

## QUICK START EXAMPLE

Here's a minimal example to get started:

```
Create a course website with:

Course Code: CS101
Course Title: Introduction to Programming
Institution: State University
Modules: 8
Password: none

Module 1: Variables and Data Types
- Description: Learn about storing and manipulating data in programs.
- Key concepts: Variables, integers, floats, strings, booleans, type conversion

Module 2: Control Flow
- Description: Make programs that make decisions and repeat actions.
- Key concepts: If statements, else, elif, loops, while, for, break, continue

[...etc...]

Include: How to Read Research Papers guide, Curated Resources page
Primary Color: #2563eb
Accent Color: #3b82f6
Font: Inter
```

---

## ADVANCED CONTENT WORKFLOWS

### 1. Audio Generation (ElevenLabs)
1. Write the script in `audio_script.md` (pure text).
2. Store API key in `.env` file (`ELEVENLABS_API_KEY=...`).
3. Run `generate_audio.py --input week_XX/audio_script.md --output week_XX/audio.mp3`.
   - Uses `eleven_multilingual_v2` model for high fidelity.
   - **Do NOT** use turbo/flash models.

### 2. Slide Creation (LaTeX Beamer)
1. Create `slides.tex` in the module folder using the "Zen" theme (minimalist black/white).
2. Compile to PDF: `pdflatex slides.tex`.
3. Convert PDF to PNG: `pdftoppm -png slides.pdf slide`.

### 3. Synchronization Data
1. Create `slides.json` mapping images to audio timestamps:
   ```json
   [
     {
       "id": "slide-01",
       "image": "slides/slide-01.png",
       "startTime": 0,
       "endTime": 15,
       "text": "Transcript segment..."
     }
   ]
   ```
2. Embed this data in `index.html` as `window.SLIDES_DATA`.

### 4. Player Integration
In `index.html`, add:
```html
<div class="player-layout">
  <div class="player-slide-container">
    <img id="slide-image" class="player-slide" src="" alt="Slide">
  </div>
  <div class="player-transcript" id="transcript"></div>
</div>
```

---

## MAINTENANCE NOTES

- **Adding Resources:** The curated resources page uses a JSON array in the HTML. Add new items to the `resources` array.
- **Updating Quizzes:** Each question is a div with `data-correct` and `data-explanations` attributes.
- **Changing Colors:** All colors are defined as CSS variables in `:root` - change once, update everywhere.
- **Adding Modules:** Copy an existing `week_XX` folder, rename, and update content and navigation links.

---

*This prompt template was derived from the DATASCI 207: Applied Machine Learning course site framework.*
