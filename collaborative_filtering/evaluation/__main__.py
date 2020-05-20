import sys
import site
import os

cwd = sys.path[0]
cwd = os.path.abspath(cwd)
collaborative_filtering_dir = os.path.join(cwd, "..")
site.addsitedir(collaborative_filtering_dir)
