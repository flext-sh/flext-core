"""Environment reading and interpolation primitives shared through ``u.Cli``."""

from __future__ import annotations

import os
import re

from flext_cli import p, r

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


class FlextCliUtilitiesEnv:
    """Read and interpolate environment variables, exposed on ``u.Cli``."""

    @staticmethod
    def env_read(name: str) -> p.Result[str]:
        """Read one environment variable by ``name`` from the process environment.

        Returns the variable's value, or an empty string when it is unset. Callers
        pass the variable name as data (e.g. resolved from configuration) so no
        consumer hardcodes a specific environment-variable identity. An empty or
        missing variable is a legitimate empty-string state, not a failure; callers
        decide whether an empty value is acceptable.
        """
        return r[str].ok(os.environ.get(name, ""))

    @staticmethod
    def env_expand(template: str) -> p.Result[str]:
        """Interpolate ``${VAR}`` / ``$VAR`` / ``${VAR:-default}`` from the environment.

        Substitutes every environment reference in ``template`` with its
        process-environment value, honouring ``${VAR:-default}`` fallbacks; an
        unset variable without a default resolves to an empty segment. Callers
        pass the template (e.g. a product's standard ``${HOME}/.tool`` path
        resolved from configuration) as data and receive the absolute string, so
        no consumer hardcodes an environment-variable identity.
        """

        def _replace(match: re.Match[str]) -> str:
            token = (
                match.group(1) if match.group(1) is not None else (match.group(2) or "")
            )
            key, _, default = token.partition(":-")
            return os.environ.get(key, default)

        return r[str].ok(_VAR_PATTERN.sub(_replace, template))


__all__: list[str] = ["FlextCliUtilitiesEnv"]
