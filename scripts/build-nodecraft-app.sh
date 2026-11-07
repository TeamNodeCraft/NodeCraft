#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
cargo build --release -p nodecraft
APP="$ROOT/build/NodeCraft.app"
CONTENTS="$APP/Contents"
MACOS="$CONTENTS/MacOS"
rm -rf "$APP"
mkdir -p "$MACOS"
cp "$ROOT/macos/Info.plist" "$CONTENTS/Info.plist"
/usr/libexec/PlistBuddy -c 'Delete :CFBundleDisplayName' "$CONTENTS/Info.plist" 2>/dev/null || true
/usr/libexec/PlistBuddy -c 'Add :CFBundleDisplayName string NodeCraft' "$CONTENTS/Info.plist" 2>/dev/null || true
printf 'APPL????' > "$CONTENTS/PkgInfo"
cp "$ROOT/rust/target/release/nodecraft" "$MACOS/NodeCraft"
chmod +x "$MACOS/NodeCraft"
if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$APP"
fi
echo "$APP"
