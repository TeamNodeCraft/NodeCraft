# Third-party notices

## SolidCraft

The `nodecraft-egui` branch consumes unmodified `solidcraft-geom`,
`solidcraft-render` and optionally `solidcraft-gpu` from
https://github.com/tangency-labs/solidcraft.

SolidCraft is licensed under either the MIT License or the Apache License,
Version 2.0, at the user's option. See the upstream `LICENSE-MIT` and
`LICENSE-APACHE` files for the full terms.

The sketch line-type icons under
`vendor/iconify-design/sketch/line-normal.svg`, `line-construction.svg`,
`line-centerline.svg` and `line-infinite.svg` are adapted from SolidCraft's
`assets/icons/ui/design/svg/model/` icons under the same license.

## SolidCraft transform gizmo

The transform gizmo in `rust/apps/nodecraft-egui/src/viewer_gizmo.rs` is
adapted from SolidCraft's `crates/ui-egui/src/gizmo.rs`.

SolidCraft is Copyright (c) 2026 SolidCraft contributors and is licensed under
either the MIT License or the Apache License, Version 2.0, at your option.

## CHOC

NodeCraft uses ideas and code adapted from CHOC
(https://github.com/Tracktion/choc), Tracktion Corporation's header-only C++
utility library.

The `nodecraft-search` crate currently contains Rust ports of CHOC's
`choc::text::WildcardPattern` and `choc::text::getLevenshteinDistance`. Further
CHOC utilities may be ported to Rust as needed.

CHOC is Copyright (c) 2022 Tracktion Corporation and is offered under the ISC
license:

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted, provided that the above copyright notice
and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.

## Inter

The egui front end bundles the Inter variable font from the Inter project
(https://github.com/rsms/inter) at
`rust/apps/nodecraft-egui/assets/InterVariable.ttf`, matching the Inter family
used by the Swift/WebView front end.

Copyright (c) 2016 The Inter Project Authors.

Inter is licensed under the SIL Open Font License, Version 1.1. The full license
text is included at `rust/apps/nodecraft-egui/assets/Inter-LICENSE.txt`.


## Icon assets

Icons under `vendor/fluent/`, `vendor/icons/` and `vendor/iconify-design/` are
bundled as local SVG assets.

| Files | Source | License | Copyright |
|---|---|---|---|
| `vendor/fluent/` | [Microsoft Fluent UI System Icons](https://github.com/microsoft/fluentui-system-icons) | MIT | Microsoft Corporation |
| andrsrxn icons under `vendor/iconify-design/` | [andrsrxn/icons](https://github.com/andrsrxn/icons) | MIT | 2026 Andrés Raxón (andrsrxn) |
| Hugeicons under `vendor/iconify-design/` | [Hugeicons](https://hugeicons.com/) | MIT | Hugeicons |
| Phosphor icons under `vendor/iconify-design/` | [Phosphor Icons](https://github.com/phosphor-icons/core) | MIT | 2023 Phosphor Icons |
| Vadivam icons under `vendor/iconify-design/` | [Vadivam](https://github.com/praveenjuge/vadivam) | MIT | Praveen Juge |
| Iconoir icons under `vendor/iconify-design/` | [Iconoir](https://github.com/iconoir-icons/iconoir) | MIT | 2021 Luca Burgio |
| Framework7 icons under `vendor/iconify-design/` | [Framework7 Icons](https://github.com/framework7io/framework7-icons) | MIT | 2016 Vladimir Kharlampidi |
| Material Design Icons under `vendor/iconify-design/` | [Material Design Icons](https://github.com/Templarian/MaterialDesign) | Pictogrammers Free License | Pictogrammers |
| `vendor/icons/`, `vendor/iconify-design/viewport/` | NodeCraft originals | project license | NodeCraft contributors |

The MIT-licensed sets share this notice:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The Pictogrammers Free License for Material Design Icons is available at
<https://github.com/Templarian/MaterialDesign/blob/master/LICENSE>.

## Lyon, Glyphon, cosmic-text

The node editor uses `lyon` for path tessellation and `glyphon` /
`cosmic-text` for shaped GPU text.

- Lyon: MIT OR Apache-2.0
- Glyphon: MIT OR Apache-2.0 OR Zlib
- cosmic-text: MIT OR Apache-2.0

## mimalloc

The NodeCraft egui binary uses mimalloc as its global allocator
(https://github.com/microsoft/mimalloc).

Copyright (c) 2018-2021 Microsoft Corporation

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## SolidCore test fixture

`crates/nodecraft-layers/tests/data/box_10x6x4.json.gz` is copied from the
solidcore repository (`crates/solidcore-tess/tests/data/box_10x6x4.json.gz`,
revision `bff1569a4c3347ed9af67faab71578cadf84fde2`) and used only as a test
fixture for the record tessellation adapter.

solidcore is licensed under the Apache License, Version 2.0. Copyright Tangency
Labs. See https://github.com/tangency-labs/solidcore for the full license.

