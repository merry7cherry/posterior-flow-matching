# Project page

`index.html`, `style.css` and `script.js` serve the shared PFM project page.
`assets/` contains the paper PDF and figures. No build step is needed.

From the repository root, preview locally:

```bash
python -m http.server 8000 --directory website --bind 127.0.0.1
```

Open <http://127.0.0.1:8000>. The main Code button opens the repository; each
application panel links to its task directory: `images/`, `pointcloud/`,
`molecules/` or `pusht/`. Keep those descriptions aligned with the task READMEs.

The `Publish project page` workflow uploads this directory to
<https://merry7cherry.github.io/posterior-flow-matching/> when website changes
reach `main`, or on manual dispatch. GitHub Pages uses **GitHub Actions** as
its publishing source.

Update figures, displayed results and `assets/paper.pdf` together from the paper.
Paper links use the preprint PDF until an arXiv identifier is available; update
the links, webpage BibTeX and root `CITATION.cff` together when it is available.
