# ELK-only input/output readability diagnosis (2026-09-25)

## Conclusion

The current problem is real, but it is **not** evidence that Graph IR or the
model topology is missing input/output edges. The problem is the Layout IR
projection and the way the single ELK compound solve exposes it:

1. The root is laid out `DOWN` with three virtual compounds (`input`, `main`,
   `auxiliary`).
2. Virtual lanes are not rendered as labeled visual bands.
3. Expanded descendants make a lane extremely tall. Cross-lane boundary ports
   then cause top-level nodes in one semantic flow to occupy very different
   vertical positions.
4. The visible diagram therefore loses the expected left-to-right spine:
   input → embedding/fusion/encoder → decoder → final norm → lm head.
5. Auxiliary MTP/DSpark content can dominate the overall canvas height and make
   the main output path look disconnected.

This is a presentation/layout failure, not a conclusion that the external
architecture gallery is authoritative or that the Graph IR should be changed.

## Local evidence

The production ELK-only graph still contains the expected top-level dataflow.
For DeepSeek-V4.1-Flash, the expanded graph has edges including:

```text
image/video input → Vision Tower → Multi-modal Projector
text IDs → embed tokens
embed/projector/text IDs → text/vision fusion
fusion → Causal Encoder → Decoder → final norm → lm head
Decoder → DSpark → lm head
```

However, after full expansion, the layout projection placed these nodes in
different vertical bands. Representative local coordinates reached tens of
thousands of pixels for expanded descendants, while the main nodes were not
visually aligned on one spine. The browser showed no routing error, but the
semantic reading order was poor.

Qwen3.8-Flash-Next and GLM-5.2 show the same class of symptom: the graph has
the expected edges, but the expanded auxiliary/deep compounds visually compete
with the main input/output path.

## External-evidence boundary

The public architecture gallery is useful terminology and cross-check material,
but it is not being used as topology truth. The current online retrieval did
not produce a sufficiently reliable set of official implementation pages for
this diagnosis, so no structural change is justified from the gallery alone.
The repository's config, implementation/source references, checkpoint truth,
and independent graph tests remain authoritative.

## Experiments not accepted

Two uncommitted experiments were reverted after inspection:

- broadening lane classification using display names;
- setting root `elk.hierarchyHandling=INCLUDE_CHILDREN`.

They did not restore a stable input/output spine and one changed the ELK-only
POC's placement assertions. Production behavior was not left in either
experimental state.

## Required next POC before implementation

Compare two ELK-native Layout IR designs on the four representative models:

1. **Partitioned stage spine**: keep real nodes at the model level, assign
   input/backbone/auxiliary stage partitions, and let ELK layer/routeline the
   whole graph without lane compounds swallowing expanded subtree height.
2. **Labeled stage compounds**: retain compounds but make input/backbone/auxiliary
   bands explicit layout objects with bounded stage anchors and visible labels;
   route deep children inside their own compound without allowing them to
   determine the top-level spine.

The winning design must pass mechanism assertions and Chrome SVG geometry
checks before production code is changed again.
