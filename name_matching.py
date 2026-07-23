"""
name_matching.py
Shared name-normalization utilities for matching players across sources
that use different name formats:

  - baseball-reference (bbref): "First Last"     e.g. "Julio Rodriguez"
  - Statcast / Savant:          "Last, First"     e.g. "Rodríguez, Julio"

Previously this normalization logic (accent stripping, "last name + first
initial" key building) was copy-pasted independently in recommender.py and
debug_tables.py. It now lives in one place so a fix here (e.g. handling a
new name format quirk) applies everywhere at once.

All matching is done on a "last name + first initial" key. This is
resilient to accents (Muñoz -> munoz) and to the two source formats
disagreeing on ordering, but is NOT resilient to two players sharing both
a last name and first initial. That's an accepted tradeoff at this
project's scale (one team's roster + trade targets, not the whole league).

Usage:
    from name_matching import key_from_first_last, key_from_last_first, last_name_only

    key_from_first_last("Julio Rodriguez")      -> "rodriguez_j"
    key_from_last_first("Rodríguez, Julio")     -> "rodriguez_j"
    last_name_only("Andrés Muñoz")              -> "munoz"
    last_name_only("Ferrer, Jose A.")           -> "ferrer"
"""

import unicodedata


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalize(s: str) -> str:
    """Lowercase + strip accents/whitespace. Building block for the key fns."""
    return _strip_accents(str(s).lower().strip())


def _clean_last(last: str) -> str:
    """
    Strip roster-page annotations like '(60-day IL)' and trailing
    whitespace/punctuation that show up in bbref roster/injury tables.
    """
    return last.split("(")[0].strip().rstrip(",")


_NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def _strip_suffix(tokens: list) -> list:
    """
    Drops a trailing generational suffix token (Jr., Sr., II, III, IV) so
    it isn't mistaken for a last name -- e.g. "Vladimir Guerrero Jr." would
    otherwise key as "jr_vladimir" instead of "guerrero_vladimir".
    """
    if len(tokens) >= 2 and normalize(tokens[-1]).rstrip(".") in _NAME_SUFFIXES:
        return tokens[:-1]
    return tokens


def key_from_first_last(name: str) -> str:
    """
    Build a 'lastname_firstname' key from bbref-style "First Last" names.
    Falls back to a normalized full-name key if there's no clear split
    (e.g. a single-word name). Roster annotations like "(60-day IL)" are
    stripped before splitting, since splitting first would otherwise treat
    "IL)" as the last name.

    Uses the full first name/nickname token (e.g. "julio", "j.p.") rather
    than just a first initial. A bare initial isn't precise enough in
    practice -- e.g. Julio Rodriguez, Jesus Rodriguez, and Johnathan
    Rodriguez all share both a last name AND a first initial, and did
    cause real false-match bugs in an earlier version of this function.
    Middle names/initials are intentionally ignored (only the first
    whitespace-separated token of the given name is used), since bbref
    and Statcast don't consistently include them in the same format.
    """
    name  = _clean_last(str(name))
    parts = _strip_suffix(name.split())
    if len(parts) < 2:
        return normalize(name)
    last  = normalize(parts[-1])
    first = normalize(parts[0])
    return f"{last}_{first}" if first else last


def key_from_last_first(name: str) -> str:
    """
    Build a 'lastname_firstname' key from Statcast-style "Last, First"
    names. Falls back to a normalized full-name key if there's no comma.
    See key_from_first_last() for why this uses a full first-name token
    rather than just an initial.
    """
    name  = _clean_last(str(name))
    parts = name.split(",")
    if len(parts) < 2:
        return normalize(name)
    last_tokens = _strip_suffix(parts[0].split())
    last        = normalize(" ".join(last_tokens)) if last_tokens else normalize(parts[0])
    first_part  = parts[1].strip().split()
    first       = normalize(first_part[0]) if first_part else ""
    return f"{last}_{first}" if first else last


def last_name_only(name: str) -> str:
    """
    Normalized last name with no first initial, used for roster-membership
    sets (e.g. 'is this player currently on the 40-man?'). Handles both
    "First Last" and "Last, First" input, and strips roster annotations
    like "(60-day IL)" -- these must be stripped BEFORE splitting on
    whitespace, or a name like "Cooper Criswell (60-day IL)" incorrectly
    yields "IL)" as the last token instead of "Criswell".
    """
    name = _clean_last(str(name))  # strip "(...)" annotations first
    if "," in name:
        last_tokens = _strip_suffix(name.split(",")[0].split())
        last = " ".join(last_tokens) if last_tokens else name.split(",")[0]
    else:
        parts = _strip_suffix(name.split())
        last  = parts[-1] if parts else name
    return normalize(last)


def key_last_from_matchkey(key: str) -> str:
    """Given a 'lastname_firstinitial' key, return just the lastname part."""
    return key.split("_")[0]
