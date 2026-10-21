# Albion playing table

The source design system is the portfolio's `docs/ALBION.md`: sand
`#D8CDAA`, deep blue `#052D46`, mist blue `#A4D9E5`, Cormorant Garamond
and Inter. Fonts are bundled locally with their licenses.

The board is the primary composition. No marketing headline, persistent setup
sidebar, editorial panels, or divider lines. Only piece counts, a short turn cue,
and icon controls appear during play. New game, rules, and move history open
on demand. Icons have accessible names, focus states, and hover/focus tooltips.
The fixed overhead board preserves square selection and orientation; the unique
visual viewpoint comes from Albion's literary, engraved world.

## Language and learning pages

Use precise, restrained language. Name algorithms and operations directly:
“Random”, “Uniform sampling”, “Evaluation”. Avoid promotional headlines,
taglines, rhetorical introductions, personification, and motivational copy.
Keep explanations short without omitting facts needed to understand the method.
Put secondary explanations and comparisons behind descriptive disclosures.

The learning page follows the same hierarchy through space and typography.
Do not add a marketing hero, bordered lesson cards, status pills, or hard divider
lines. Distinguish available opponents from planned models with plain labels.

## Board decoration

The board also carries light paper grain, an engraved compass-rose throne mark,
and four translucent oak-and-acorn corner ornaments. A wordless landscape footer
extends the world with standing stones, a lake, mist, a distant castle, and a
crescent moon. Decoration is excluded from the accessibility tree and cannot
intercept pointer events. There are no new separators, gameplay rules, or controls.

## Original piece artwork

Created with the built-in image-generation tool using the portfolio's
`public/images/albion-landscape.webp` as a style reference. These are original
transparent illustrations, not glossy checker pieces. The files are retained
unchanged from generation:

- `src/hnef/web/assets/attacker.png`: dark raven.
- `src/hnef/web/assets/defender.png`: pale white hart.
- `src/hnef/web/assets/king.png`: oak-crowned king.
- `src/hnef/web/assets/board-flourish.png`: oak-and-acorn corner engraving.
- `src/hnef/web/assets/albion-footer.png`: moonlit Arthurian lakeshore.

All three are bundled in the Python distribution and served locally. Square
occupancy, legal moves, and outcomes continue to come from the Python engine.
The raven/hart identities are visual treatments, not changes to the rules.

## Generation prompts

Each prompt began with:

> Use case: stylized-concept. Asset type: transparent browser board-game piece illustration.

Shared art direction:

> Albion Hnef game pieces. Palette strictly deep Prussian navy #052D46, parchment sand #D8CDAA and pale mist blue #A4D9E5. Antique Arthurian wood engraving, elegant hand-carved collectible silhouette, intricate negative-space incised line work, subtly worn printmaking edges. Sophisticated literary fantasy book-cover illustration. NOT photoreal 3D, NOT glossy plastic, NOT cartoon mascot. Upright front three-quarter viewpoint, self-contained strong silhouette readable at 60 pixels. Single isolated piece centered filling 85% of a square canvas, with even padding, no scenery, no lettering, no border, no watermark, genuinely transparent alpha background. Small engraved oval plinth under the figure. Style reference is the attached Albion landscape: use its ink-engraved visual vocabulary, not the landscape composition.

Attacker subject:

> A raven knight: proud raven in profile, folded feathered wings forming a heraldic shield, angular beak, short broad silhouette. Dominant DEEP NAVY ink with thin parchment and mist-blue engraved feather hatching. The attacker piece, imposing and dark.

Defender subject:

> A white hart guardian: elegant seated stag, antlers held high, heraldic front three-quarter pose, folded legs and a shield-shaped chest. Dominant PALE MIST BLUE and PARCHMENT with fine deep navy engraved contours; clearly lighter than the raven. Low broad silhouette, finely etched fur, a small sprig of oak at its feet. The defender piece.

King subject:

> The antler-crowned king: a regal bearded Arthurian king enthroned, oak-leaf crown rising in three unmistakable points, a sword held vertically, long flowing cloak. Dominant PARCHMENT with deep navy engraved outlines and pale mist blue inner cloak details. Taller than a normal pawn but compact iconic silhouette. Exquisite miniature carved sculptural portrait, solemn and luminous. The royal king piece.

## Flourish and footer prompts

Both illustrations were created with the built-in image-generation tool, using
the portfolio landscape and the generated raven piece as style references.
Final PNGs retain the generated alpha; CSS controls their scale and subtlety.

Board corner:

> Use case: stylized-concept. Asset type: transparent decorative botanical corner for a minimal board-game UI. Make ONE original finely engraved corner ornament: an old twisting oak twig sweeps upward along the LEFT edge and outward along the BOTTOM edge of a square image, forming a loose open L. Small elegant lobed oak leaves, two acorns, fern curls, extremely delicate incised woodcut hatching. Leave the upper-right and center 75 percent of the square EMPTY and genuinely transparent. The twig curves organically; NO straight border, no frame. Palette muted deep Prussian blue #052D46 with subtle pale mist blue #A4D9E5 hatching. Literature and Arthurian woodland wonder, sophisticated antique book engraving matching the attached landscape and raven engraving. Spare, exquisite and airy. No text, no creatures, no plinth, no board, no paper background. 1024px square transparent PNG. Useful as a modest corner flourish, readable at 90px.

Footer:

> Use case: stylized-concept. Asset type: original very wide transparent footer illustration for minimal Albion Hnef board-game website. An exquisitely engraved dreamlike Arthurian lakeshore, horizontal panoramic 3:1 composition. At far LEFT a small cluster of standing stones and sparse oak foliage; at far RIGHT a tiny distant castle on a low ridge, beside a graceful leaning oak. Between them a broad calm lake with delicate incised horizontal reflection lines, a few wisps of pale blue mist, and far distant rolling hills. A very small crescent moon and just three tiny stars hover above the distant castle. Empty transparent space above and most of the middle, illustration concentrated along the lower third, asymmetric, open and airy. A sense of splendid, quiet wonder. Antique woodcut etched linework matching the reference Albion landscape. Strict palette deep Prussian navy #052D46, parchment sand #D8CDAA and pale mist blue #A4D9E5. Delicate light-weight lines, no large solid ink masses, no enclosing frame, no text or labels, no board or UI, no characters. GENUINELY transparent alpha background, soft organic edges fading to transparency. Deliver an elegant landscape-shaped transparent PNG, approximately 3:1 aspect ratio.
