"""Batch file-operation helper for FlextTestsFiles."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path

from flext_tests import c, m, p, r, t
from flext_tests._utilities._files._contexts import FlextTestsFilesContextsMixin
from flext_tests._utilities.payload import FlextTestsPayloadUtilities


class FlextTestsFilesBatchMixin(FlextTestsFilesContextsMixin):
    """Batch create/read/delete file operations."""

    def batch_files[TModel: m.BaseModel](
        self,
        items: t.Tests.BatchFiles,
        *,
        directory: Path | None = None,
        operation: c.Tests.Operation = c.Tests.Operation.CREATE,
        model: type[TModel] | None = None,
        on_error: c.Tests.ErrorMode = c.Tests.ErrorMode.COLLECT,
        parallel: bool = False,
    ) -> p.Result[m.Tests.BatchResult]:
        """Batch file operations with error handling.

        Args:
            items: Mapping of file names to content or a sequence of name-content
                pairs.
            directory: Target directory for create operations
            operation: "create", "read", or "delete"
            model: Optional model class for read operations
            on_error: Error handling mode ("stop", "skip", "collect")
            parallel: Run operations in parallel (not implemented yet)

        Returns:
            r[m.Tests.BatchResult] with results and errors

        """
        try:
            params = m.Tests.BatchParams.model_validate({
                "files": items,
                "directory": directory,
                "operation": operation,
                "model": model,
                "on_error": on_error,
                "parallel": parallel,
            })
        except c.EXC_BASIC_TYPE as exc:
            return r[m.Tests.BatchResult].fail(
                f"Invalid parameters for batch operation: {exc}"
            )
        files_dict: MutableMapping[str, t.Tests.TestobjectSerializable] = dict(
            params.files
        )
        error_mode_str = (
            "collect" if params.on_error is c.Tests.ErrorMode.COLLECT else "fail"
        )

        def process_one(
            name_and_content: tuple[str, t.Tests.TestobjectSerializable],
        ) -> p.Result[Path]:
            """Process single file operation."""
            name, content = name_and_content
            path = Path(content) if isinstance(content, (Path, str)) else Path(name)
            result: p.Result[Path]
            match params.operation:
                case c.Tests.Operation.CREATE:
                    try:
                        payload = (
                            {
                                k: FlextTestsPayloadUtilities.to_payload(v)
                                for k, v in content.items()
                            }
                            if isinstance(content, Mapping)
                            else content
                        )
                        result = r[Path].ok(
                            self.create(
                                self._coerce_file_content(payload),
                                name,
                                params.directory,
                            )
                        )
                    except (OSError, TypeError, ValueError, AttributeError) as e:
                        result = r[Path].fail(f"Failed to create {name}: {e}")
                case c.Tests.Operation.READ:
                    read_result = self.read(path, model_cls=None)
                    result = (
                        r[Path].ok(path)
                        if read_result.success
                        else r[Path].fail(read_result.error or f"Failed to read {name}")
                    )
                case c.Tests.Operation.DELETE:
                    try:
                        path.unlink(missing_ok=True)
                        result = r[Path].ok(path)
                    except OSError as e:
                        result = r[Path].fail(f"Failed to delete {name}: {e}")
            return result

        items_list = list(files_dict.items())
        results_dict: MutableMapping[str, p.Result[t.Tests.TestResultValue]] = {}
        failed_dict: t.MutableStrMapping = {}
        rtype = r[t.Tests.TestResultValue]
        for name, _ in items_list:
            op_result = process_one((name, files_dict[name]))
            if op_result.success:
                results_dict[name] = rtype.ok(op_result.value)
                continue
            err_msg = op_result.error or "Unknown error"
            results_dict[name] = rtype.fail(err_msg)
            failed_dict[name] = err_msg
            if error_mode_str == "fail":
                return r[m.Tests.BatchResult].fail(err_msg)

        summary = m.Tests.BatchResult.model_validate({
            "succeeded": len(items_list) - len(failed_dict),
            "failed": len(failed_dict),
            "total": len(items_list),
            "results": dict(results_dict),
            "errors": dict(failed_dict),
        })
        return r[m.Tests.BatchResult].ok(summary)


__all__: list[str] = ["FlextTestsFilesBatchMixin"]
