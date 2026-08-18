//! An additive directory of library-owned NLP package surfaces.
//!
//! Focused crates remain the owners of their operations. This crate only
//! namespaces discovery and forwards requests to the corresponding library.

use std::collections::BTreeSet;

use runtime_core::{
    surface_operation, OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation,
    SurfaceRequest, SurfaceResponse,
};

const REGISTRY_OPERATION: &str = "registry.describe";

type SurfaceRunner = fn(SurfaceRequest) -> Result<SurfaceResponse, String>;

struct RegisteredSurface {
    surface: PackageSurface,
    run: SurfaceRunner,
}

/// Reports an invalid package-surface registry before dispatch can occur.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    DuplicateLibrary(String),
    DuplicateOperation(String),
    EmptyLibrary,
    EmptyOperation { library: String },
    ReservedNamespaceDelimiter { field: &'static str, value: String },
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateLibrary(library) => write!(formatter, "duplicate library `{library}`"),
            Self::DuplicateOperation(operation) => {
                write!(formatter, "duplicate namespaced operation `{operation}`")
            }
            Self::EmptyLibrary => formatter.write_str("package surface has an empty library name"),
            Self::EmptyOperation { library } => {
                write!(
                    formatter,
                    "package surface `{library}` has an empty operation id"
                )
            }
            Self::ReservedNamespaceDelimiter { field, value } => {
                write!(
                    formatter,
                    "{field} `{value}` contains the reserved `/` delimiter"
                )
            }
        }
    }
}

impl std::error::Error for RegistryError {}

/// Returns the namespace used by registry operation IDs.
pub fn namespace_operation(library: &str, operation: &str) -> String {
    format!("{library}/{operation}")
}

/// Validates that package names and namespaced operation IDs are unique.
pub fn validate_package_surfaces<'a>(
    surfaces: impl IntoIterator<Item = &'a PackageSurface>,
) -> Result<(), RegistryError> {
    let mut libraries = BTreeSet::new();
    let mut operations = BTreeSet::new();

    for surface in surfaces {
        if surface.library.is_empty() {
            return Err(RegistryError::EmptyLibrary);
        }
        if surface.library.contains('/') {
            return Err(RegistryError::ReservedNamespaceDelimiter {
                field: "library",
                value: surface.library.clone(),
            });
        }
        if !libraries.insert(surface.library.as_str()) {
            return Err(RegistryError::DuplicateLibrary(surface.library.clone()));
        }
        for operation in &surface.operations {
            if operation.id.as_str().is_empty() {
                return Err(RegistryError::EmptyOperation {
                    library: surface.library.clone(),
                });
            }
            if operation.id.as_str().contains('/') {
                return Err(RegistryError::ReservedNamespaceDelimiter {
                    field: "operation",
                    value: operation.id.as_str().to_owned(),
                });
            }
            let namespaced = namespace_operation(&surface.library, operation.id.as_str());
            if !operations.insert(namespaced.clone()) {
                return Err(RegistryError::DuplicateOperation(namespaced));
            }
        }
    }
    Ok(())
}

/// Returns the additive registry surface with deterministic, namespaced IDs.
pub fn package_surface() -> PackageSurface {
    let libraries = registered_surfaces();
    validate_registered_surfaces(&libraries)
        .expect("the statically registered NLP package surfaces must be unique");

    let mut operations = Vec::with_capacity(
        1 + libraries
            .iter()
            .map(|entry| entry.surface.operations.len())
            .sum::<usize>(),
    );
    operations.push(surface_operation(
        REGISTRY_OPERATION,
        "Inspect registered NLP package surfaces",
        "Lists library-owned NLP operations exposed through this additive registry.",
        serde_json::json!({"includeOperations": true}),
    ));
    for entry in libraries {
        operations.extend(
            entry
                .surface
                .operations
                .into_iter()
                .map(|operation| namespaced_operation(entry.surface.library.as_str(), operation)),
        );
    }

    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations,
    }
}

/// Runs a namespaced operation through the library that owns it.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let requested_operation = request.operation.clone();
    let libraries = registered_surfaces();
    validate_registered_surfaces(&libraries).map_err(|error| error.to_string())?;

    if requested_operation.as_str() == REGISTRY_OPERATION {
        return Ok(SurfaceResponse {
            operation: requested_operation,
            value: serde_json::json!({
                "library": env!("CARGO_PKG_NAME"),
                "version": env!("CARGO_PKG_VERSION"),
                "libraries": libraries.iter().map(|entry| &entry.surface.library).collect::<Vec<_>>(),
                "operations": package_surface()
                    .operations
                    .into_iter()
                    .map(|operation| operation.id)
                    .collect::<Vec<_>>(),
            }),
            diagnostics: Vec::new(),
            artifacts: Vec::new(),
        });
    }

    let (library, operation) = requested_operation
        .as_str()
        .split_once('/')
        .ok_or_else(|| unsupported_operation_message(requested_operation.as_str()))?;
    let entry = libraries
        .iter()
        .find(|entry| entry.surface.library == library)
        .ok_or_else(|| unsupported_operation_message(requested_operation.as_str()))?;
    if !entry
        .surface
        .operations
        .iter()
        .any(|candidate| candidate.id.as_str() == operation)
    {
        return Err(unsupported_operation_message(requested_operation.as_str()));
    }

    let mut response = (entry.run)(SurfaceRequest {
        operation: OperationId::new(operation),
        input: request.input,
    })?;
    response.operation = requested_operation;
    Ok(response)
}

fn namespaced_operation(library: &str, mut operation: SurfaceOperation) -> SurfaceOperation {
    operation.id = OperationId::new(namespace_operation(library, operation.id.as_str()));
    operation
}

fn validate_registered_surfaces(libraries: &[RegisteredSurface]) -> Result<(), RegistryError> {
    validate_package_surfaces(libraries.iter().map(|entry| &entry.surface))
}

fn unsupported_operation_message(operation: &str) -> String {
    runtime_core::SurfaceError::unsupported_operation(operation, env!("CARGO_PKG_NAME"))
        .to_error_string()
}

fn registered_surfaces() -> Vec<RegisteredSurface> {
    vec![
        registered(
            text_analysis::surface::package_surface,
            text_analysis::surface::run_surface_operation,
        ),
        registered(
            text_classification::surface::package_surface,
            text_classification::surface::run_surface_operation,
        ),
        registered(
            text_core::surface::package_surface,
            text_core::surface::run_surface_operation,
        ),
        registered(
            text_embeddings::surface::package_surface,
            text_embeddings::surface::run_surface_operation,
        ),
        registered(
            text_generation::surface::package_surface,
            text_generation::surface::run_surface_operation,
        ),
        registered(
            text_generation_linguistics::surface::package_surface,
            text_generation_linguistics::surface::run_surface_operation,
        ),
        registered(
            text_index::surface::package_surface,
            text_index::surface::run_surface_operation,
        ),
        registered(
            text_lexical::surface::package_surface,
            text_lexical::surface::run_surface_operation,
        ),
        registered(
            text_linguistics::surface::package_surface,
            text_linguistics::surface::run_surface_operation,
        ),
        registered(
            text_model_runtime::surface::package_surface,
            text_model_runtime::surface::run_surface_operation,
        ),
        registered(
            text_question_answering::surface::package_surface,
            text_question_answering::surface::run_surface_operation,
        ),
        registered(
            text_retrieval::surface::package_surface,
            text_retrieval::surface::run_surface_operation,
        ),
        registered(
            text_transcripts::surface::package_surface,
            text_transcripts::surface::run_surface_operation,
        ),
    ]
}

fn registered(surface: fn() -> PackageSurface, run: SurfaceRunner) -> RegisteredSurface {
    RegisteredSurface {
        surface: surface(),
        run,
    }
}

#[cfg(test)]
mod tests {
    use runtime_core::{surface_operation, PackageSurface, RuntimeCapabilities, SurfaceRequest};

    use super::*;

    #[test]
    fn registry_namespaces_every_library_operation_deterministically() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(surface.library, "moenarch-nlp-package-registry");
        assert_eq!(ids.first(), Some(&REGISTRY_OPERATION));
        assert!(ids.contains(&"moenarch-text-core/text.statistics"));
        assert!(ids.contains(&"moenarch-text-retrieval/retrieval.search"));
        assert_eq!(ids.len(), ids.iter().collect::<BTreeSet<_>>().len());
        assert_eq!(surface, package_surface());
    }

    #[test]
    fn registry_forwards_to_the_library_owned_runner() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("moenarch-text-core/text.statistics"),
            input: serde_json::json!({"text": "one two"}),
        })
        .unwrap();

        assert_eq!(
            response.operation.as_str(),
            "moenarch-text-core/text.statistics"
        );
        assert_eq!(response.value["value"]["wordCount"], 2);
    }

    #[test]
    fn registry_rejects_an_unknown_or_unnamespaced_operation() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("text.statistics"),
            input: serde_json::json!({}),
        })
        .unwrap_err();

        assert!(error.contains("unsupported_operation"));
    }

    #[test]
    fn validation_rejects_a_duplicate_namespaced_operation() {
        let duplicate = PackageSurface {
            library: "duplicate".to_string(),
            version: "0.1.0".to_string(),
            capabilities: RuntimeCapabilities::pure_rust(),
            operations: vec![
                surface_operation("same", "Same", "Same", serde_json::json!({})),
                surface_operation("same", "Same again", "Same", serde_json::json!({})),
            ],
        };
        let error = validate_package_surfaces([&duplicate]).unwrap_err();

        assert_eq!(
            error,
            RegistryError::DuplicateOperation("duplicate/same".to_string())
        );
    }

    #[test]
    fn validation_reserves_the_namespace_delimiter() {
        let invalid = PackageSurface {
            library: "invalid/library".to_string(),
            version: "0.1.0".to_string(),
            capabilities: RuntimeCapabilities::pure_rust(),
            operations: Vec::new(),
        };

        assert!(matches!(
            validate_package_surfaces([&invalid]),
            Err(RegistryError::ReservedNamespaceDelimiter {
                field: "library",
                ..
            })
        ));
    }
}
