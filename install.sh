#!/bin/bash

# Installer script for fcbenchtool

# Directories to install files
INSTALL_DIR="/usr/local/bin"
SHARE_DIR="/usr/local/share/fctool"
mkdir -p "$SHARE_DIR"

# Copy files to installation directory

cp fcbenchtool fcplottool fccomparetool fcanalyzetool fcdebugtool fcasmtool fcexplorer.py fcbenchgraph.py fcanalyze.py "$INSTALL_DIR"

chmod +x "$INSTALL_DIR/fcbenchtool"
chmod +x "$INSTALL_DIR/fcplottool"
chmod +x "$INSTALL_DIR/fccomparetool"
chmod +x "$INSTALL_DIR/fcanalyzetool"
chmod +x "$INSTALL_DIR/fcdebugtool"
chmod +x "$INSTALL_DIR/fcasmtool"
chmod +x "$INSTALL_DIR/fcexplorer.py"
chmod +x "$INSTALL_DIR/fcbenchgraph.py"
chmod +x "$INSTALL_DIR/fcanalyze.py"

# Create symbolic links for Python scripts
ln -sf "$INSTALL_DIR/fcbenchgraph.py" "$INSTALL_DIR/fcbenchgraph"
ln -sf "$INSTALL_DIR/fcanalyze.py" "$INSTALL_DIR/fcanalyze"

cp *_footer.cpp "$SHARE_DIR"
cp *_header.cpp "$SHARE_DIR"

# Exit with success status

exit 0