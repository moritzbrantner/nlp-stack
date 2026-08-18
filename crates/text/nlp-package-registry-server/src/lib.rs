//! Thin HTTP adapter for the additive NLP package registry.

use std::io;

use runtime_core::{
    server::{self, ServerAdapterMetadata},
    PackageSurface,
};

pub const LIBRARY_CRATE: &str = "nlp-package-registry";
pub const SURFACE_KIND: &str = "api";
pub const LIBRARY_IMPORT: &str = "use nlp_package_registry";
pub const CLI_PACKAGE: &str = "nlp-package-registry-cli";
pub const WASM_PACKAGE: &str = "nlp-package-registry-wasm";

pub type HttpResponse = server::HttpResponse;

const METADATA: ServerAdapterMetadata = ServerAdapterMetadata {
    library_crate: LIBRARY_CRATE,
    surface_kind: SURFACE_KIND,
    library_import: LIBRARY_IMPORT,
    cli_package: CLI_PACKAGE,
    app_package: "not-provided",
    wasm_package: WASM_PACKAGE,
};

pub fn package_surface() -> PackageSurface {
    nlp_package_registry::package_surface()
}

pub fn serve(addr: &str) -> io::Result<()> {
    server::serve(
        addr,
        METADATA,
        package_surface,
        nlp_package_registry::run_surface_operation,
    )
}

pub fn response_for(method: &str, path: &str, body: &str) -> HttpResponse {
    server::response_for(
        method,
        path,
        body,
        METADATA,
        package_surface,
        nlp_package_registry::run_surface_operation,
    )
}

pub fn package_metadata_json() -> String {
    server::package_metadata_json(METADATA, package_surface())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operations_endpoint_lists_namespaced_operations() {
        let response = response_for("GET", "/api/operations", "");
        assert_eq!(response.status_code, 200);
        assert!(response.body.contains("moenarch-text-core/text.statistics"));
    }
}
