# Odosseus deck in LaTeX

A Beamer copy of `../odysseus-presentation.html`: same colours, fonts, logo and layout, 16:9.

| file | what it is |
|---|---|
| `odosseus.tex` | the slides: edit the text here |
| `beamerthemeodosseus.sty` | the theme: colours, fonts, footer, logo and every building block below |
| `latexmkrc` | build settings: LuaLaTeX (needed for the fonts), output `odosseus.pdf` |
| `fonts/` | Inter, Space Grotesk, Outfit (TTF from Google Fonts via Fontsource) with their licences `OFL-*.txt` (SIL Open Font License: keep them next to the fonts) |
| `videos/` | copies of the visualization videos the stills on slide 5 link to (not in git; rebuild with `visualization/run_viz.py`) |
| `images/` | pictures: the failure-case slide, and the A2RL car (`a2rl_constructor_car_side.png`, `…_transparent.png`, from slide 2 of the organizers' INTRO deck) |

## Build

```bash
cd Presentation/latex
latexmk              # -> odosseus.pdf, ~5 s
latexmk -C           # remove build files
```

**In VS Code:**
1. Install LaTeX Workshop: `code --install-extension James-Yu.latex-workshop`.
2. Open `odosseus.tex` and run *Build LaTeX project* with the recipe **latexmk (latexmkrc)**. The default recipe passes `-pdf`, which forces pdfLaTeX and fails on the fonts.
3. *View LaTeX PDF* opens the PDF in a tab. It rebuilds on every save. Cmd+click in the PDF jumps to the source line.

## Units

The HTML slides are 1600 × 900 px; this deck is 160 × 90 mm, so **1 px = 0.1 mm**. Lengths are written as in the
HTML, e.g. `58\px`. Font sizes are `\pxfont{<px>}{<line height>}`, the CSS `font: 58px/1.05`.

## Building blocks (`odosseus.tex`)

| slide element | macro |
|---|---|
| logo slide | `\logoslide`; set before it: `\logocaption{…}` (caption, centred on the slide), `\logoside{images/…}` (image beside the logo, empty = logo alone), `\logocredits{…}` (credit line at the bottom edge), each with `\renewcommand` |
| a content slide (footer with page number) | `\begin{odslide} … \end{odslide}` |
| kicker line "— 02 OUR SOLUTION" | `\kicker{02}{Our solution}` |
| title, orange part | `\slidetitle{Car state predicts, \highlight{the map corrects}}` |
| sentence under the title | `\slidesub{…}` |
| row of steps with arrows | `\begin{pipeline}[<height px>] \pipestep[<width factor>]{plain\|hl\|new}{label}{title}{text} \pipearrow … \end{pipeline}` |
| stat row (left-border colours cycle orange, amber, sea, navy) | `\begin{stats} \stat{0.21}{m}{label\\second line} … \end{stats}` |
| small note, bold part in navy | `\slidenote{\textbf{Trade-off:} …}` |
| two cards with bullet points | `\begin{cards}[<height px>] \card{no\|ok}{title}{\point{…}\point{…}} \card{…} \end{cards}` |
| navy insight bar | `\insight{Key insight}{text with \textbf{amber part}}` |
| failure cases (3 columns) | `\begin{fails} \failcase{<image or {}>}{<CAM label>}{title}{text}{error tag}{fix} … \end{fails}` |
| clickable video stills (2 per row) | `\begin{videos} \videocard{images/<poster>.jpg}{<link>}{LABEL}{caption} … \end{videos}` |
| summary slide on Constructor blue | `\summaryslide{06}{Thank you}{\sumword{Predict,}{description} \sumword{…}{…} \sumword{…}{…}}` |
| amber placeholder | `\ph{GPU model}` |
| arrow | `\ra` (the font subset has no → glyph) |

- **Images:** put the file in `images/` and give `images/<file>` as the first argument of `\failcase`. Camera
  frames (1506 × 728) fit exactly; `{}` keeps the "Image goes here" placeholder.
- **Colours:** at the top of `beamerthemeodosseus.sty`, under the same names as the HTML (`ink`, `signal`, `amber`,
  `sea`, …). Use them anywhere with `\color{signal}`.
- **Sizes that must add up:**
  - a pipeline row is 4 steps (width factors 1, 1, 1.2, 1.2) plus 3 arrows = 1408 px. With other factors, change
    `\pipefr` in the theme.
  - cards are 689 px each, failure columns 448 px.
- **The footer text** is `\footertext` in the theme; redefine it in `odosseus.tex` with `\renewcommand\footertext{…}`.

## Image rights

- **The car on the logo slide** is © Constructor 2026, from slide 2 of the challenge INTRO deck. The challenge
  material is licensed for the National AI Hackathon 2026 only, with no redistribution. So:
  - the credit line on the logo slide names the source;
  - the image file is in `.gitignore`, so it stays out of the public repository. The PDF for the jury includes it.
  - Without the file the slide still builds and shows a short "not found" note.
- **The Odosseus logo** was generated with ChatGPT (OpenAI); the logo slide's credit line says so.

## Videos on slide 5

PDF viewers no longer play video inside the PDF (that relied on Flash), so each video is a clickable still:
- `run:videos/<file>.mp4` (the default) opens the local copy next to the PDF in your video player. This works in
  Adobe Acrobat and Skim, after a confirmation prompt; macOS Preview ignores such links.
- For the jury, upload the videos (e.g. YouTube unlisted, Google Drive) and put the `https://…` link in
  `\videocard{…}{https://…}{…}{…}`. That works in every viewer.
- The stills are frames of the videos: `ffmpeg -ss 20 -i videos/<file>.mp4 -frames:v 1 -vf scale=1280:-2 images/<poster>.jpg`.
