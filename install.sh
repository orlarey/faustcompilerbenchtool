#!/bin/bash

# Installer script for the Faust Compiler Benchmark Tools.
#
# Adding a tool means adding it to TOOLS below — nothing else. The copy,
# the chmod and the version stamp follow from that one list.

set -u

# Directories to install files
INSTALL_DIR="/usr/local/bin"
SHARE_DIR="/usr/local/share/fctool"
mkdir -p "$SHARE_DIR"

TOOLS="fcbenchtool fcmultibench fcplottool fccomparetool fcanalyzetool
       fcdebugtool fcasmtool fcspilltool fcautotool fcgentool multifaust
       fcversion fcexplorer.py fcbenchgraph.py fcanalyze.py fcoptimize.py
       fcspillgraph.py"

# Copy files to installation directory

for t in $TOOLS; do
    cp "$t" "$INSTALL_DIR/$t" || exit 1
    chmod 755 "$INSTALL_DIR/$t"
done

# Create symbolic links for Python scripts
ln -sf "$INSTALL_DIR/fcbenchgraph.py" "$INSTALL_DIR/fcbenchgraph"
ln -sf "$INSTALL_DIR/fcanalyze.py" "$INSTALL_DIR/fcanalyze"
ln -sf "$INSTALL_DIR/fcoptimize.py" "$INSTALL_DIR/fcoptimize"
ln -sf "$INSTALL_DIR/fcspillgraph.py" "$INSTALL_DIR/fcspillgraph"

cp *_footer.cpp "$SHARE_DIR"
cp *_header.cpp "$SHARE_DIR"

# The version stamp. An installed copy lives outside any checkout and has
# no way to name itself, so record here which commit it came from; that is
# what `fcversion` reads and what every tool answers to --version. A
# working tree with modified tracked files is stamped `-dirty`: an
# installation made from it is not reproducible from its sha alone.
SHA=$(git -C "$(dirname "$0")" rev-parse --short HEAD 2>/dev/null || echo unknown)
if [ "$SHA" != unknown ]; then
    git -C "$(dirname "$0")" diff --quiet HEAD 2>/dev/null || SHA="$SHA-dirty"
    WHEN=$(git -C "$(dirname "$0")" log -1 --format=%cI HEAD 2>/dev/null)
else
    WHEN=unknown
fi
printf '%s committed %s installed %s\n' \
    "$SHA" "$WHEN" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$SHARE_DIR/VERSION"
echo "installed $SHA into $INSTALL_DIR"

# Exit with success status

exit 0
