"""FLX Core - Enterprise Data Integration and Orchestration Platform.

FLX Core is the foundational package of the FLX Meltano Enterprise platform, providing
enterprise-grade data integration capabilities built on top of the open-source Meltano
framework. It extends Meltano with production-ready features essential for enterprise
data operations.

Core capabilities:
- Advanced pipeline orchestration with dependency management
- Distributed execution across multiple workers and environments
- Enterprise authentication and authorization (JWT, OAuth, SAML)
- Comprehensive monitoring and observability (metrics, logs, traces)
- High availability and fault tolerance mechanisms
- Multi-tenant support with resource isolation
- Advanced scheduling with cron and event-based triggers
- Data quality validation and lineage tracking
- Plugin lifecycle management with versioning
- Configuration management with environment promotion

The platform follows Domain-Driven Design (DDD) principles with clear separation
between domain logic, application services, and infrastructure concerns. All
components adhere to the Zero Tolerance architectural principles established
in the project.
"""

from flx_core.__version__ import __version__
from flx_core.application.application import FlxApplication
from flx_core.config import Settings, settings

__author__ = "FLX Team"
__email__ = "team@flx.io"

__all__ = ["FlxApplication", "Settings", "__version__", "settings"]
