from typing import Tuple

import git

def get_active_branch() -> str:
    repo = git.Repo(
        path=__file__,
        search_parent_directories=True)
    return repo.active_branch.name

def get_repo_status() -> Tuple[str, bool]:
    repo = git.Repo(
        path=__file__,
        search_parent_directories=True)
    return repo.head.commit.hexsha, repo.is_dirty()