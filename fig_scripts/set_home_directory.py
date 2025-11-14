import os
import sys

def get_project_root_homedir_in_sys_path(target_folder="inter_areal_predictability_github"):
    """
    Iterate through sys.path entries, and for each, walk up its parent directories
    until finding one that contains target_folder. Returns the first found project root,
    or None if not found.
    """
    for entry in sys.path:
        # If entry is empty (""), it refers to the current working directory.
        if not entry:
            entry = os.getcwd()
        entry = os.path.abspath(entry)
        candidate = entry
        # Walk up the directory tree
        while candidate != os.path.dirname(candidate):
            if os.path.isdir(os.path.join(candidate, target_folder)):
                return candidate, os.path.join(candidate, target_folder)+ '/'
            candidate = os.path.dirname(candidate)
    return None
