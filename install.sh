#!/bin/bash

# Installer script for fcbenchtool

# Directories to install files
INSTALL_DIR="/usr/local/bin"
SHARE_DIR="/usr/local/share/fctool"
mkdir -p "$SHARE_DIR"

# Copy files to installation directory

cp fcbenchtool fcplottool fccomparetool fcanalyzetool fcdebugtool fcasmtool fcexplorer.py fcbenchgraph.py fcanalyze.py fcoptimize.py "$INSTALL_DIR"

chmod 755 "$INSTALL_DIR/fcbenchtool"
chmod 755 "$INSTALL_DIR/fcplottool"
chmod 755 "$INSTALL_DIR/fccomparetool"
chmod 755 "$INSTALL_DIR/fcanalyzetool"
chmod 755 "$INSTALL_DIR/fcdebugtool"
chmod 755 "$INSTALL_DIR/fcasmtool"
chmod 755 "$INSTALL_DIR/fcexplorer.py"
chmod 755 "$INSTALL_DIR/fcbenchgraph.py"
chmod 755 "$INSTALL_DIR/fcanalyze.py"
chmod 755 "$INSTALL_DIR/fcoptimize.py"

# Create symbolic links for Python scripts
ln -sf "$INSTALL_DIR/fcbenchgraph.py" "$INSTALL_DIR/fcbenchgraph"
ln -sf "$INSTALL_DIR/fcanalyze.py" "$INSTALL_DIR/fcanalyze"
ln -sf "$INSTALL_DIR/fcoptimize.py" "$INSTALL_DIR/fcoptimize"

cp *_footer.cpp "$SHARE_DIR"
cp *_header.cpp "$SHARE_DIR"

# Exit with success status

exit 0