# Third-party notices

## FluidCAD

Portions of the sketch interaction logic in `sketch-fluid.js` and `viewer.js` are
adapted from FluidCAD (https://github.com/Fluid-CAD/FluidCAD).

MIT License

Copyright (c) 2026 Marwan Aouida

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

## SolidCraft

The `nodecraft-egui` branch consumes unmodified `solidcraft-geom`,
`solidcraft-render` and optionally `solidcraft-gpu` from
https://github.com/tangency-labs/solidcraft.

SolidCraft is licensed under either the MIT License or the Apache License,
Version 2.0, at the user's option. See the upstream `LICENSE-MIT` and
`LICENSE-APACHE` files for the full terms.

## SolidCraft transform gizmo

The transform gizmo in `rust/apps/nodecraft-egui/src/viewer_gizmo.rs` is
adapted from SolidCraft's `crates/ui-egui/src/gizmo.rs`.

SolidCraft is Copyright (c) 2026 SolidCraft contributors and is licensed under
either the MIT License or the Apache License, Version 2.0, at your option.

## Inter

The egui front end bundles the Inter variable font from the Inter project
(https://github.com/rsms/inter) at
`rust/apps/nodecraft-egui/assets/InterVariable.ttf`, matching the Inter family
used by the Swift/WebView front end.

Copyright (c) 2016 The Inter Project Authors.

Inter is licensed under the SIL Open Font License, Version 1.1. The full license
text is included at `rust/apps/nodecraft-egui/assets/Inter-LICENSE.txt`.


## Lyon, Glyphon, cosmic-text

The node editor uses `lyon` for path tessellation and `glyphon` /
`cosmic-text` for shaped GPU text.

- Lyon: MIT OR Apache-2.0
- Glyphon: MIT OR Apache-2.0 OR Zlib
- cosmic-text: MIT OR Apache-2.0
