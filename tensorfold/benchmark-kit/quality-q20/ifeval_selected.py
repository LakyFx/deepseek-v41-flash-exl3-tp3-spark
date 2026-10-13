# Copyright 2023 The Google Research Authors. Licensed under Apache-2.0.
# See google-LICENSE. Exact selected classes; original source pinned in SOURCE-MANIFEST.json.
import json,re,random

class Instruction:
  """An instruction template."""

  def __init__(self, instruction_id):
    self.id = instruction_id

  def build_description(self, **kwargs):
    raise NotImplementedError("`build_description` not implemented.")

  def get_instruction_args(self):
    raise NotImplementedError("`get_instruction_args` not implemented.")

  def get_instruction_args_keys(self):
    raise NotImplementedError("`get_instruction_args_keys` not implemented.")

  def check_following(self, value):
    raise NotImplementedError("`check_following` not implemented.")

class BulletListChecker(Instruction):
  """Checks the bullet list in the prompt."""

  def build_description(self, *, num_bullets = None):  # pyrefly: ignore[bad-override]
    """Build the instruction description.

    Args:
      num_bullets: An integer specifying the exact number of bullet lists
        that is required to appear in the response.

    Returns:
      A string representing the instruction description.
    """
    self._num_bullets = num_bullets
    if self._num_bullets is None or self._num_bullets < 0:
      self._num_bullets = random.randint(1, _NUM_BULLETS)
    self._description_pattern = (
        "Your answer must contain exactly {num_bullets} bullet points. " +
        "Use the markdown bullet points such as:\n" +
        "* This is point 1. \n" +
        "* This is point 2")
    return self._description_pattern.format(
        num_bullets=self._num_bullets)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_bullets": self._num_bullets}  # pyrefly: ignore[bad-return]

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_bullets"]

  def check_following(self, value):
    r"""Check if the number of bullet lists meets the requirement.

    Args:
      value: A string representing the response. The response is expected to
        contain some bullet lists that start with `\*`.

    Returns:
      True if the actual number of bullet lists in the response meets the
      requirement.
    """
    bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
    bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
    num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
    return num_bullet_lists == self._num_bullets

class HighlightSectionChecker(Instruction):
  """Checks the highlighted section."""

  def build_description(self, *, num_highlights = None):  # pyrefly: ignore[bad-override]
    """Build the instruction description.

    Args:
      num_highlights: An integer specifying the minimum number of highlighted
        sections.

    Returns:
      A string representing the instruction description.
    """
    self._num_highlights = num_highlights
    if self._num_highlights is None or self._num_highlights < 0:
      self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

    self._description_pattern = (
        "Highlight at least {num_highlights} sections in your answer with " +
        "markdown, i.e. *highlighted section*.")

    return self._description_pattern.format(num_highlights=self._num_highlights)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_highlights": self._num_highlights}  # pyrefly: ignore[bad-return]

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_highlights"]

  def check_following(self, value):
    """Checks if the number of highlighted sections meets the requirement.

    Args:
      value: a string repesenting the response. The response is expected to
        contain highlighted sections in the format of *highlighted*.

    Returns:
      True if the actual number of highlighted sections in the format of
      *highlighed sections* meets the minimum requirement; otherwise False.
    """
    num_highlights = 0
    highlights = re.findall(r"\*[^\n\*]*\*", value)
    double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
    for highlight in highlights:
      if highlight.strip("*").strip():
        num_highlights += 1
    for highlight in double_highlights:
      if highlight.removeprefix("**").removesuffix("**").strip():
        num_highlights += 1

    return num_highlights >= self._num_highlights

class JsonFormat(Instruction):
  """Check the Json format."""

  def build_description(self):  # pyrefly: ignore[bad-override]
    self._description_pattern = (
        "Entire output should be wrapped in JSON format. You can use markdown"
        " ticks such as ```."
    )
    return self._description_pattern

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    value = (
        value.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
      json.loads(value)
    except ValueError as _:
      return False
    return True

class TitleChecker(Instruction):
  """Checks the response for a title."""

  def build_description(self):  # pyrefly: ignore[bad-override]
    """Build the instruction description."""
    self._description_pattern = (
        "Your answer must contain a title, wrapped in double angular brackets,"
        " such as <<poem of joy>>."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response contains a title."""
    pattern = r"<<[^\n]+>>"
    re_pattern = re.compile(pattern)
    titles = re.findall(re_pattern, value)

    for title in titles:
      if title.lstrip("<").rstrip(">").strip():
        return True
    return False

class CommaChecker(Instruction):
  """Checks the response for no commas."""

  def build_description(self):  # pyrefly: ignore[bad-override]
    """Build the instruction description."""
    self._description_pattern = (
        "In your entire response, refrain from the use of any commas."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks that the response does not contain commas."""
    return not re.search(r"\,", value)

class QuotationChecker(Instruction):
  """Checks response is wrapped with double quotation marks."""

  def build_description(self):  # pyrefly: ignore[bad-override]
    """Build the instruction description."""
    self._description_pattern = (
        "Wrap your entire response with double quotation marks."
    )
    return self._description_pattern

  def get_instruction_args(self):
    """Returns the keyword args of build description."""
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response is wrapped with double quotation marks."""
    value = value.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'
