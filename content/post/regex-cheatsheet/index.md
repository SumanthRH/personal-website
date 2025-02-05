---
title: Python Regex Cheatsheet
subtitle: A cheatsheet for regex in Python with the `re` module.
date: 2025-01-16T05:37:15.089Z
summary: A cheatsheet for regex in Python with the `re` module.
draft: false
featured: false
commentable: true
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
# Python Regex Cheatsheet

This is a cheatsheet for regex in Python with the `re` module. I plan to update this as I go, but the essentials are here. 

## Table of Contents
- [Basic Patterns](#basic-patterns)
   - [Character Classes](#character-classes)
   - [Anchors](#anchors)
   - [Quantifiers](#quantifiers)
   - [Groups and References](#groups-and-references)
- [`re` Functions](#re-functions)
- [Modifiers / Function Flags](#modifiers--function-flags)
- [Example Regex Patterns](#example-regex-patterns)
- [Examples](#examples)
- [Notes](#notes)

## Basic Patterns
### Character Classes
- `.` - Matches any character except newline
- `\w` - Matches word characters (letters, digits, underscores)
- `\W` - Matches non-word characters
- `\d` - Matches digits (0-9)
- `\D` - Matches non-digits
- `\s` - Matches whitespace (space, tab, newline)
- `\S` - Matches non-whitespace
- `[abc]` - Matches any character in the set
- `[^abc]` - Matches any character not in the set
- `[a-z]` - Matches any character in the range
- `[A-Z]` - Matches any uppercase letter
- `[0-9]` - Matches any digit

### Anchors

- `^` - Start of string/line
- `$` - End of string/line
- `\b` - Word boundary
- `\B` - Not a word boundary

### Quantifiers

- `*` - Match 0 or more times
- `+` - Match 1 or more times
- `?` - Match 0 or 1 time
- `{n}` - Match exactly n times
- `{n,}` - Match n or more times
- `{n,m}` - Match between n and m times
- `*?` - Match 0 or more times (non-greedy)
- `+?` - Match 1 or more times (non-greedy)

### Groups and References

- `(...)` - Capturing group
- `(?:...)` - Non-capturing group
- `(?P<name>...)` - Named capturing group
- `(?P=name)` - Reference named group
- `\1, \2, etc.` - Reference numbered group



# `re` Functions
```python
import re

# Search for pattern in string
re.search(pattern, string)  # Returns Match object or None

# Find all non-overlapping matches
re.findall(pattern, string)  # Returns list of strings

# Split string by pattern
re.split(pattern, string)  # Returns list of strings

# Replace pattern with repl string
re.sub(pattern, repl, string)  # Returns modified string

# Compile pattern for reuse
regex = re.compile(pattern)
```


## Modifiers / Function Flags

```python
# At the start of the pattern:
(?i)     # Case-insensitive matching
(?m)     # Multiline mode (^ and $ match start/end of each line)
(?s)     # Dot matches newline
(?x)     # Verbose mode (allows comments and whitespace)

# As function flags:
re.IGNORECASE  # or re.I - Case-insensitive matching
re.MULTILINE   # or re.M - Multi-line matching
re.DOTALL      # or re.S - Dot matches any char including newline
re.VERBOSE     # or re.X - Allow comments and whitespace
re.ASCII       # or re.A - ASCII-only matching
re.UNICODE     # or re.U - Unicode matching (default in Python 3)
```


## Examples

1. Answer sanitization: Say you're extracting the final answer to a math problem from a larger text. The text has ".........The final answer is \\boxed{123}....". You want to extract the 123. Sometimes the answer has a dollar sign, sometimes it doesn't. 

```python  
# regex pattern extracts the answer as a named group. The ?P<answer> is the named group.
# \d+ matches one or more digits.
pattern = r"The final answer is \\boxed{(?P<answer>\d+)}[\.]*"

# replace dollar sign with nothing
new_text = re.sub(r"\$", "", text) # or use text.replace("$", "")

# Use the regex to extract the answer
answer = re.search(pattern, new_text).group('answer')
```

## Notes
- Use raw strings (`r"pattern"`) for regex patterns to avoid escape character issues
- Use `re.compile()` for patterns you'll use multiple times. 