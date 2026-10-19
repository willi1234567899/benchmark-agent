from __future__ import annotations

from benchmark_agent.config import AppConfig, GuidanceLevelConfig
from benchmark_agent.models import ContextDocument


def load_workload_prompt(
    app_config: AppConfig,
    guidance_level: GuidanceLevelConfig | None = None,
) -> str:
    file_name = "influxdb_l2_task.md"
    if guidance_level is not None and guidance_level.workload_file:
        file_name = guidance_level.workload_file
    path = app_config.paths.workload_dir / file_name
    return path.read_text(encoding="utf-8")


def load_openapi_documents(
    app_config: AppConfig,
    guidance_level: GuidanceLevelConfig,
) -> list[ContextDocument]:
    documents: list[ContextDocument] = []
    for file_name in guidance_level.openapi_files:
        openapi_path = app_config.paths.openapi_dir / file_name
        documents.append(
            ContextDocument(
                title=f"API specification ({file_name})",
                source_path=openapi_path.relative_to(app_config.paths.project_root).as_posix(),
                content=openapi_path.read_text(encoding="utf-8"),
                media_type="application/yaml",
            )
        )
    return documents


def load_use_case_documents(
    app_config: AppConfig,
    guidance_level: GuidanceLevelConfig,
) -> list[ContextDocument]:
    usecases_dir = app_config.paths.inputs_dir / "usecases"
    documents: list[ContextDocument] = []
    for file_name in guidance_level.use_case_files:
        use_case_path = usecases_dir / file_name
        documents.append(
            ContextDocument(
                title=f"Use case ({file_name})",
                source_path=use_case_path.relative_to(app_config.paths.project_root).as_posix(),
                content=use_case_path.read_text(encoding="utf-8"),
                media_type="text/markdown",
            )
        )
    return documents


def load_example_documents(app_config: AppConfig) -> list[ContextDocument]:
    examples_dir = app_config.paths.inputs_dir / "examples"
    documents: list[ContextDocument] = []
    if not examples_dir.is_dir():
        return documents
    for path in sorted(examples_dir.iterdir()):
        if not path.is_file() or path.suffix not in {".feature", ".py"}:
            continue
        media_type = "text/x-python" if path.suffix == ".py" else "text/gherkin"
        documents.append(
            ContextDocument(
                title=f"Multi-shot example ({path.name})",
                source_path=path.relative_to(app_config.paths.project_root).as_posix(),
                content=path.read_text(encoding="utf-8"),
                media_type=media_type,
            )
        )
    return documents


def load_feedback_documents(
    app_config: AppConfig,
    guidance_level: GuidanceLevelConfig,
) -> list[ContextDocument]:
    usecases_dir = app_config.paths.inputs_dir / "usecases"
    documents: list[ContextDocument] = []
    for file_name in guidance_level.feedback_files:
        feedback_path = usecases_dir / file_name
        documents.append(
            ContextDocument(
                title=f"Feedback hints ({file_name})",
                source_path=feedback_path.relative_to(app_config.paths.project_root).as_posix(),
                content=feedback_path.read_text(encoding="utf-8"),
                media_type="text/markdown",
            )
        )
    return documents


def load_reference_documents(
    app_config: AppConfig,
    guidance_level: GuidanceLevelConfig,
) -> list[ContextDocument]:
    reference_root = app_config.paths.project_root / "reference"
    documents: list[ContextDocument] = []
    for file_name in guidance_level.reference_files:
        reference_path = reference_root / file_name
        if reference_path.suffix == ".py":
            media_type = "text/x-python"
        elif reference_path.suffix == ".feature":
            media_type = "text/gherkin"
        else:
            media_type = "text/markdown"
        documents.append(
            ContextDocument(
                title=f"Reference ({file_name})",
                source_path=reference_path.relative_to(app_config.paths.project_root).as_posix(),
                content=reference_path.read_text(encoding="utf-8"),
                media_type=media_type,
            )
        )
    return documents


def format_document_sections(documents: list[ContextDocument]) -> str:
    if not documents:
        return ""
    return "\n\n".join(
        f"### {document.title}\nSource: {document.source_path}\n\n{document.content}"
        for document in documents
    )
