#!/bin/bash

# Installer script for fcbenchtool

# Directories to install files
INSTALL_DIR="/usr/local/bin"
SHARE_DIR="/usr/local/share/fctool"
mkdir -p "$SHARE_DIR"

# Copy files to installation directory

cp fcbenchtool fcplottool fccomparetool "$INSTALL_DIR"

chmod +x "$INSTALL_DIR/fcbenchtool"
chmod +x "$INSTALL_DIR/fcplottool"
chmod +x "$INSTALL_DIR/fccomparetool"

cp *_footer.cpp "$SHARE_DIR"
cp *_header.cpp "$SHARE_DIR"

# Exit with success status

exit 0