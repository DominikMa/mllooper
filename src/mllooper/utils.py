from typing import Any, Tuple, Optional
import urllib.parse
import urllib.request

urlsplit = urllib.parse.urlsplit
urlunsplit = urllib.parse.urlunsplit


def full_name(o: Any) -> str:
    return f"{o.__module__}.{o.__class__.__qualname__}"


def vcs_get_url_rev_and_auth(url: str) -> Tuple[str, Optional[str], Tuple]:
    """pip._internal.vcs.versioncontrol"""
    scheme, netloc, path, query, frag = urllib.parse.urlsplit(url)
    if "+" not in scheme:
        raise ValueError(
            "Sorry, {!r} is a malformed VCS url. "
            "The format is <vcs>+<protocol>://<url>, "
            "e.g. svn+http://myrepo/svn/MyApp#egg=MyApp".format(url)
        )
    # Remove the vcs prefix.
    scheme = scheme.split("+", 1)[1]
    netloc, user_pass = netloc, (None, None)
    rev = None
    if "@" in path:
        path, rev = path.rsplit("@", 1)
        if not rev:
            raise Exception(
                "The URL {!r} has an empty revision (after @) "
                "which is not supported. Include a revision after @ "
                "or remove @ from the URL.".format(url)
            )
    url = urllib.parse.urlunsplit((scheme, netloc, path, query, ""))
    return url, rev, user_pass


def git_get_url_rev_and_auth(url: str) -> Tuple[str, Optional[str], Tuple]:
    """from pip._internal.git"""
    scheme, netloc, path, query, fragment = urlsplit(url)
    if scheme.endswith("file"):
        initial_slashes = path[: -len(path.lstrip("/"))]
        newpath = initial_slashes + urllib.request.url2pathname(path).replace(
            "\\", "/"
        ).lstrip("/")
        after_plus = scheme.find("+") + 1
        url = scheme[:after_plus] + urlunsplit(
            (scheme[after_plus:], netloc, newpath, query, fragment),
        )

    if "://" not in url:
        assert "file:" not in url
        url = url.replace("git+", "git+ssh://")
        url, rev, user_pass = vcs_get_url_rev_and_auth(url)
        url = url.replace("ssh://", "")
    else:
        url, rev, user_pass = vcs_get_url_rev_and_auth(url)

    return url, rev, user_pass
