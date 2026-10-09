#!/bin/sh
# Recreates the two corpus cases a zip file cannot carry. Run once, from
# anywhere. Linux/macOS only -- on Windows chmod has no effect at all.
set -e
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

mkdir -p "$here/mc3-corpus/archive"
chmod 000 "$here/mc3-corpus/vendor/internal_audit.txt"

echo "OK - mc3-corpus/archive/ exists and is empty"
echo "OK - mc3-corpus/vendor/internal_audit.txt is now mode 000"
echo
echo "If you are root, chmod 000 will NOT stop you reading that file."
echo "At evaluation we drop DAC_OVERRIDE, so it genuinely raises. Test with:"
echo "  docker run --cap-drop DAC_OVERRIDE ..."
