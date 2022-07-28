"""
Use this package to define application-wide constants or other definitions.
"""
import os

# Avoid using relative paths. Instead, whenever you create a path use this as the root for that path.
# This will help evade relative path errors when a file or package gets moved.
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
