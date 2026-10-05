# Project page

This is a static HTML/CSS/JavaScript page with no build step, external fonts,
analytics, or runtime service. Serve it from the repository root for local review:

```bash
python -m http.server 8000 --directory website --bind 127.0.0.1
```

Open `http://127.0.0.1:8000`. Check both a narrow mobile viewport and a desktop
viewport, the four application tabs (including arrow keys, Home, and End), the
formulation and supplementary-table disclosures, and the BibTeX copy button. All application sections
remain visible without JavaScript. Copying requires browser clipboard support.

Local in-app browser checks on 2026-10-05 covered 1440px desktop and 390px/320px
mobile widths, keyboard tab navigation, all three disclosures, BibTeX copying,
and all four application sections with JavaScript disabled. Tables and long
formulas scroll within their own containers; no page-wide horizontal overflow
or captured console errors were observed. Temporary browser overrides were
reset after testing. These checks do not establish a deployed Pages site.

The page follows the paper's argument: motivation, posterior learning and
readouts, ODE/SDE sampling, generation across domains, posterior diagnostics,
covariance and stochasticity ablations, then sampling cost. Domain tabs pair
the paper's baseline tables with their corresponding figures. The covariance
summary has a full comparison below it, and stochasticity is visible in the
main page. An independent source audit matched all 48 numerical rows in the
13 HTML tables to the manuscript, including the five-row covariance summary.

## Publication

The public repository is `merry7cherry/posterior-flow-matching`, initialized
from a reviewed source snapshot without private development history. The
private `merry7cherry/pfm` research repository is preserved. Original PFM code
and the project-page implementation use MIT; upstream components retain their
own licenses as documented in `LICENSE` and `THIRD_PARTY_NOTICES.md`.

The project-page destination is
https://merry7cherry.github.io/posterior-flow-matching/ . Paper buttons use
`assets/paper.pdf` so they work under the repository's Pages path. Keep the
preprint PDF link until an actual arXiv identifier is available, then update
paper links and citation metadata together. Code and running-guide links point
to the public repository.

The `Publish project page` workflow uploads only this directory to GitHub Pages.
The repository's Pages publishing source must be set to **GitHub Actions**. It
runs for website changes pushed to `main`, or by manual dispatch on `main`.
Preparing this workflow does not itself publish the website or change repository
visibility.

## Paper and assets

`assets/provenance.json` records the manuscript revision, source figure paths,
hashes, rendering settings, and the paper download checksum. WebP figures are
rendered from the authors' existing paper PDFs at 2,200 pixels wide; their data,
labels, and scientific comparisons are unchanged. `method.webp` and
`teaser.webp` are complementary crops of the method overview. No template code
or external media is embedded.

`assets/paper.pdf` is the 42-page preprint compiled from the unmodified tracked
manuscript source. It is approximately 27.3 MiB and is downloaded only when a
paper link is opened. It retains the original vector figures and embedded image
quality. Page images total approximately 1.2 MiB.

When updating the paper, regenerate figures from the reviewed source, update
the provenance manifest and displayed metrics together, and verify the PDF
visually. Update the webpage BibTeX, repository README citation, and
`CITATION.cff` together when a verified arXiv identifier or publication record is
available. Do not infer conference acceptance from the manuscript template.

The image code provides training, checkpoint save/load/resume, sampling and
evaluation interfaces; pretrained checkpoints are not provided. Default
configurations are adjustable examples, not exact recipes for the paper's
numbers. Link other task repositories only after their public destinations and
contents are verified.
