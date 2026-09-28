# infra — the non-Python parts of delivery

Docker images, the Potree viewer pages and the screenshot harness. They live inside the
`geovap-deliver` distribution rather than in a top-level tools directory because they are part of
what that distribution delivers: `geovap publish` builds an octree with the PotreeConverter image
and serves it with the viewer image, and neither works if they were left behind.

    containers/   compose.yml + the three images (PotreeConverter, nginx viewer, PDAL shell)
    viewer/       the Potree pages: index, the single-cloud view, consolidated, clusters
    shots/        Playwright driver for the automated screenshot checks (stage `visual`)

Everything is parameterised by the same three roots the Python side resolves a dataset from —
`$GEOVAP_DATA`, `$GEOVAP_WORKSPACE`, `$GEOVAP_PUBLISH` — so the containers and `geovap run` cannot
disagree about where the data is. `compose.yml` fails with a named error if one is unset rather than
silently mounting the wrong thing. Copy `containers/.env.example` to `.env` to set them per machine.

## The Potree API guard

`containers/viewer.Dockerfile` greps the built Potree bundle for `degToRad(-course + 90)` and
`repeat.x = -1` and **fails the image build** if either is missing. Our panorama export and the
consolidated viewer page are hand-tuned against that specific Images360 implementation: if a newer
Potree revision changes the sphere orientation formula or the mirrored-texture default, the viewer
does not error — it just renders the panoramas at the wrong yaw, or flipped. Failing the build is
the only place that mismatch is cheap to notice. Do not remove the guard; update the pages and then
the guard together.
