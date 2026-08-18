//! Thin CLI adapter for the additive NLP package registry.

use runtime_core::{
    cli::{self, CliAdapterMetadata},
    PackageSurface, SurfaceResponse,
};

pub const LIBRARY_CRATE: &str = "nlp-package-registry";
pub const SURFACE_KIND: &str = "cli";
pub const LIBRARY_IMPORT: &str = "use nlp_package_registry";
pub const SERVER_PACKAGE: &str = "nlp-package-registry-server";
pub const WASM_PACKAGE: &str = "nlp-package-registry-wasm";

const METADATA: CliAdapterMetadata = CliAdapterMetadata {
    library_crate: LIBRARY_CRATE,
    surface_kind: SURFACE_KIND,
    library_import: LIBRARY_IMPORT,
    server_package: SERVER_PACKAGE,
    app_package: "not-provided",
    wasm_package: WASM_PACKAGE,
};

pub fn package_surface() -> PackageSurface {
    nlp_package_registry::package_surface()
}

pub fn package_metadata_json() -> String {
    cli::package_metadata_json(METADATA, package_surface())
}

pub fn command_schema_json() -> String {
    cli::command_schema_json()
}

pub fn run_operation(operation: &str, input: serde_json::Value) -> Result<SurfaceResponse, String> {
    cli::run_wrapped_operation(
        operation,
        input,
        nlp_package_registry::run_surface_operation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_describes_the_aggregate_library() {
        assert!(package_metadata_json().contains(LIBRARY_CRATE));
        assert!(package_surface()
            .operations
            .iter()
            .any(|operation| operation.id.as_str() == "moenarch-text-core/text.statistics"));
    }
}
