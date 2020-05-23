import sys
import os
import site

cwd = os.path.abspath(sys.path[0])
collaborative_filtering_dir = os.path.join(cwd, "..", "collaborative_filtering")
site.addsitedir(collaborative_filtering_dir)
