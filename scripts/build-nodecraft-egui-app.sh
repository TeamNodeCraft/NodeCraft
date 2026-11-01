#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
cargo build --release -p nodecraft-egui
APP="$ROOT/build/nodecraft-egui.app"
CONTENTS="$APP/Contents"
MACOS="$CONTENTS/MacOS"
rm -rf "$APP"
mkdir -p "$MACOS"
cp "$ROOT/macos/Info.plist" "$CONTENTS/Info.plist"
sed -i '' 's/<string>NodeCraft<\/string>/<string>nodecraft-egui<\/string>/g' "$CONTENTS/Info.plist"
/usr/libexec/PlistBuddy -c 'Delete :CFBundleDisplayName' "$CONTENTS/Info.plist" 2>/dev/null || true
/usr/libexec/PlistBuddy -c 'Add :CFBundleDisplayName string NodeCraft' "$CONTENTS/Info.plist" 2>/dev/null || true
printf 'APPL????' > "$CONTENTS/PkgInfo"
cp "$ROOT/rust/target/release/nodecraft-egui" "$MACOS/nodecraft-egui"
chmod +x "$MACOS/nodecraft-egui"
if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$APP"
fi
echo "$APP"
