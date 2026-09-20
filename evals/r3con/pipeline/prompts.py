from __future__ import annotations

from typing import Any

import jinja2
import yaml

from evals.r3con.pipeline.settings import settings


def load_prompt(name: str, *, version: str, **context: Any) -> str:
    """Load and render a stage's prompt at the given ``version``.

    Prompts live at ``prompts/<name>/<version>.yaml`` — ``name`` may be slash-
    separated for the inference sub-folder (``inference/codeact`` →
    ``prompts/inference/codeact/<v>.yaml``). The ``version`` is supplied by the run's
    :class:`evals.r3con.pipeline.config.RunConfig` (``config.prompts[name]``) — prompt versions
    are part of the experiment identity, not an env default. Template variables are
    keyword args.

    YAML shape::

        instructions: |-
          ...prose, may contain {{ jinja }} placeholders...
        examples:                 # optional
          - name: <example-name>
            text: |-
              Input: ...
              Output: ...
    """
    path = settings.PROMPTS_DIR / name / f"{version}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No prompt at {path} (stage {name!r}, version {version!r}).")
    data = yaml.safe_load(path.read_text())
    body: str = data["instructions"]
    if examples := data.get("examples"):
        body += "\n\nExamples:\n\n" + "\n\n".join(ex["text"] for ex in examples)
    return jinja2.Template(body).render(**context)
