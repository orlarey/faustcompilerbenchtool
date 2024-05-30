#!/bin/bash

# Installer script for fcbenchtool

# Directories to install files
INSTALL_DIR="/usr/local/bin"
SHARE_DIR="/usr/local/share/fcbenchtool"
mkdir -p "$SHARE_DIR"



# Copy files to installation directory
cp -r fcbenchtool "$INSTALL_DIR"
chmod +x "$INSTALL_DIR/fcbenchtool"
cp bencharch_*.cpp "$SHARE_DIR"

# Exit with success status
exit 0