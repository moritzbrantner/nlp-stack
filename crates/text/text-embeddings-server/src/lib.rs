use std::io;

use runtime_core::{
    server::{self, ServerAdapterMetadata},
    PackageSurface,
};

/// Wrapped library crate name.
pub const LIBRARY_CRATE: &str = "text-embeddings";
/// Adapter surface kind.
pub const SURFACE_KIND: &str = "api";
/// Rust import path for the wrapped crate.
pub const LIBRARY_IMPORT: &str = "use text_embeddings";
/// Companion CLI package name.
pub const CLI_PACKAGE: &str = "text-embeddings-cli";
/// Companion React app package name.
pub const APP_PACKAGE: &str = "text-embeddings-app";
/// Companion WASM package name.
pub const WASM_PACKAGE: &str = "text-embeddings-wasm";

pub type HttpResponse = server::HttpResponse;

const METADATA: ServerAdapterMetadata = ServerAdapterMetadata {
    library_crate: LIBRARY_CRATE,
    surface_kind: SURFACE_KIND,
    library_import: LIBRARY_IMPORT,
    cli_package: CLI_PACKAGE,
    app_package: APP_PACKAGE,
    wasm_package: WASM_PACKAGE,
};

pub fn package_surface() -> PackageSurface {
    text_embeddings::surface::package_surface()
}

pub fn serve(addr: &str) -> io::Result<()> {
    server::serve(
        addr,
        METADATA,
        package_surface,
        text_embeddings::surface::run_surface_operation,
    )
}

pub fn response_for(method: &str, path: &str, body: &str) -> HttpResponse {
    let response = server::response_for(
        method,
        path,
        body,
        METADATA,
        package_surface,
        text_embeddings::surface::run_surface_operation,
    );
    match (method, path) {
        ("GET", "/health") | ("GET", "/api/package") => with_candle_device_metadata(response),
        _ => response,
    }
}

pub fn package_metadata_json() -> String {
    let mut value: serde_json::Value =
        serde_json::from_str(&server::package_metadata_json(METADATA, package_surface()))
            .expect("server metadata is valid JSON");
    insert_candle_device(&mut value);
    value.to_string()
}

fn with_candle_device_metadata(response: HttpResponse) -> HttpResponse {
    let mut value = match serde_json::from_str::<serde_json::Value>(&response.body) {
        Ok(value) => value,
        Err(_) => return response,
    };
    insert_candle_device(&mut value);
    HttpResponse {
        body: value.to_string(),
        ..response
    }
}

fn insert_candle_device(value: &mut serde_json::Value) {
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "candleDevice".to_string(),
            serde_json::to_value(text_model_runtime::candle_device_preference())
                .expect("Candle device preference serializes"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_endpoint_reports_package() {
        let response = response_for("GET", "/health", "");
        assert_eq!(response.status_code, 200);
        assert!(response.body.contains(LIBRARY_CRATE));
    }
}
