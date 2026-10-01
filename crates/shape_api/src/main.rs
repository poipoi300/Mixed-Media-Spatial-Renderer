//! A minimal server for the viewer, arranging local media into 3D shapes.
//!
//! This exists to show that the viewer's API is not shaped around the
//! server it was first written against. It has no concept of dimensions or
//! axes; it publishes a shape dropdown, a radius slider, a jitter toggle, a
//! seed field and a randomize button, and the viewer renders exactly those
//! because a server describes its own controls.
//!
//! The whole contract is three endpoints:
//!
//! * `GET  /health`         — readiness
//! * `POST /catalog/stream` — NDJSON snapshots; this server sends one
//! * `POST /projection`     — one snapshot for a set of control values
//!
//! ```text
//! cargo run -p shape_api -- --port 8766
//! cargo run -p spatial_viewer -- --api http://127.0.0.1:8766
//! ```

mod media;
mod panel;
mod shapes;

use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use spatial_api::ControlValues;
use tiny_http::{Header, Request, Response, Server};

use panel::{build_snapshot, CatalogSnapshot, Controls, ServerState, RANDOMIZE_CONTROL};

/// One line of the NDJSON catalog stream. Flattening the snapshot keeps the
/// event and the `/projection` body the same shape without splicing JSON
/// text, which cannot survive nested objects.
#[derive(Debug, Serialize)]
struct SnapshotEvent {
    kind: &'static str,
    complete: bool,
    roots: Vec<String>,
    #[serde(flatten)]
    snapshot: CatalogSnapshot,
}

/// The viewer's request body. Every control value arrives on every request,
/// so this server keeps no per-client state.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ProjectionRequest {
    roots: Vec<String>,
    control_values: ControlValues,
    activated: Option<String>,
    limit: usize,
}

impl ProjectionRequest {
    fn limit(&self) -> usize {
        if self.limit == 0 {
            10_000
        } else {
            self.limit
        }
    }
}

fn main() {
    let port = parse_port();
    let address = format!("127.0.0.1:{port}");
    let server = match Server::http(&address) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("Failed to listen on {address}: {error}");
            std::process::exit(1);
        }
    };
    println!("Shape API listening on http://{address}");
    println!("Point the viewer at it:");
    println!("    cargo run --release -p spatial_viewer -- --api http://{address}");

    let state = Mutex::new(ServerState::default());
    for request in server.incoming_requests() {
        if let Err(error) = handle(request, &state) {
            eprintln!("Request failed: {error}");
        }
    }
}

fn parse_port() -> u16 {
    let mut port = 8766;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--port" => {
                let Some(value) = args.next() else {
                    eprintln!("Missing value for --port");
                    std::process::exit(2);
                };
                match value.parse() {
                    Ok(parsed) => port = parsed,
                    Err(error) => {
                        eprintln!("Invalid value for --port: {error}");
                        std::process::exit(2);
                    }
                }
            }
            "--help" | "-h" => {
                println!(
                    "\
Shape API — a minimal server for the spatial viewer

USAGE:
    shape_api [--port <PORT>]

Arranges the images and videos under the folders the viewer's start menu
sends into a selectable 3D shape. Folders are chosen in the viewer, not here.

OPTIONS:
    --port <PORT>  Port to listen on [default: 8766]
    -h, --help     Print help
"
                );
                std::process::exit(0);
            }
            unknown => {
                eprintln!("Unknown argument: {unknown}");
                std::process::exit(2);
            }
        }
    }
    port
}

fn handle(mut request: Request, state: &Mutex<ServerState>) -> std::io::Result<()> {
    let url = request
        .url()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_owned();
    match (request.method().as_str(), url.as_str()) {
        ("GET", "/health") => respond_json(request, r#"{"status":"ok"}"#.to_owned()),
        ("POST", "/projection") => {
            let parsed = read_request(&mut request);
            match parsed {
                Ok(body) => respond_json(request, encode(&project(state, &body))),
                Err(message) => respond_json(request, error_event(&message)),
            }
        }
        ("POST", "/catalog/stream") => {
            let parsed = read_request(&mut request);
            match parsed {
                // One complete snapshot: this server has nothing to stream
                // progressively, which the NDJSON contract allows.
                Ok(body) => respond_ndjson(request, format!("{}\n", stream_event(state, &body))),
                Err(message) => respond_ndjson(request, format!("{}\n", error_event(&message))),
            }
        }
        _ => request.respond(Response::from_string("Not found").with_status_code(404)),
    }
}

/// One NDJSON stream line: a complete snapshot, since this server has
/// nothing to deliver progressively.
fn stream_event(state: &Mutex<ServerState>, body: &ProjectionRequest) -> String {
    let event = SnapshotEvent {
        kind: "snapshot",
        complete: true,
        roots: body.roots.clone(),
        snapshot: project(state, body),
    };
    serde_json::to_string(&event)
        .unwrap_or_else(|error| error_event(&format!("Failed to encode snapshot: {error}")))
}

/// Runs one request against the shared state.
fn project(state: &Mutex<ServerState>, body: &ProjectionRequest) -> CatalogSnapshot {
    let mut state = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.ensure_media(&body.roots);
    // Pressing Randomize is what makes the next arrangement differ; the seed
    // still determines which arrangement that is.
    if body.activated.as_deref() == Some(RANDOMIZE_CONTROL) {
        state.shuffle_round += 1;
    }
    let controls = Controls::from_values(&body.control_values, body.activated.as_deref());
    build_snapshot(&state, &controls, body.limit())
}

fn encode(snapshot: &CatalogSnapshot) -> String {
    serde_json::to_string(snapshot)
        .unwrap_or_else(|error| error_event(&format!("Failed to encode snapshot: {error}")))
}

fn read_request(request: &mut Request) -> Result<ProjectionRequest, String> {
    let mut body = String::new();
    request
        .as_reader()
        .read_to_string(&mut body)
        .map_err(|error| format!("Could not read request body: {error}"))?;
    if body.trim().is_empty() {
        return Ok(ProjectionRequest::default());
    }
    serde_json::from_str(&body).map_err(|error| format!("Invalid request body: {error}"))
}

fn error_event(message: &str) -> String {
    serde_json::json!({"kind": "error", "message": message}).to_string()
}

fn respond_json(request: Request, body: String) -> std::io::Result<()> {
    request.respond(Response::from_string(body).with_header(content_type("application/json")))
}

fn respond_ndjson(request: Request, body: String) -> std::io::Result<()> {
    request.respond(Response::from_string(body).with_header(content_type("application/x-ndjson")))
}

fn content_type(value: &str) -> Header {
    Header::from_bytes(&b"Content-Type"[..], value.as_bytes())
        .expect("static content type header is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_body_falls_back_to_workable_defaults() {
        let request = ProjectionRequest::default();
        assert_eq!(request.limit(), 10_000);
    }

    #[test]
    fn the_viewers_request_body_parses() {
        let body = r#"{
            "roots": ["E:/images"],
            "control_values": {"shape": "cube", "radius": 30.0, "jitter": true},
            "activated": "randomize",
            "offset": 0,
            "limit": 5000
        }"#;

        let request: ProjectionRequest = serde_json::from_str(body).unwrap();

        assert_eq!(request.roots, vec!["E:/images".to_owned()]);
        assert_eq!(request.activated.as_deref(), Some("randomize"));
        assert_eq!(request.limit(), 5000);
        assert_eq!(request.control_values.len(), 3);
    }

    /// The stream line must be a snapshot event carrying the same panel and
    /// projection the non-streaming endpoint returns, and must parse as the
    /// viewer's own `CatalogStreamEvent`.
    #[test]
    fn a_stream_snapshot_wraps_the_projection_payload() {
        let state = Mutex::new(ServerState::default());
        let body = ProjectionRequest {
            roots: vec!["fixture".to_owned()],
            ..Default::default()
        };

        let line = stream_event(&state, &body);
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();

        assert_eq!(parsed["kind"], "snapshot");
        assert_eq!(parsed["complete"], true);
        assert_eq!(parsed["roots"][0], "fixture");
        assert!(parsed["panel"]["widgets"].is_array());
        assert!(parsed["projection"]["axis_labels"].is_array());

        // The viewer must be able to deserialize exactly this.
        let event: spatial_api::CatalogStreamEvent = serde_json::from_str(&line).unwrap();
        let spatial_api::CatalogStreamEvent::Snapshot {
            panel, complete, ..
        } = event
        else {
            panic!("expected a snapshot event");
        };
        assert!(complete);
        assert_eq!(panel.find("shape").map(|widget| widget.id()), Some("shape"));
    }

    #[test]
    fn each_randomize_press_advances_the_arrangement() {
        let state = Mutex::new(ServerState {
            roots: vec!["fixture".to_owned()],
            media: (0..20)
                .map(|index| crate::media::MediaFile {
                    path: std::path::PathBuf::from(format!("{index}.png")),
                    is_video: false,
                })
                .collect(),
            shuffle_round: 0,
        });
        let press = || {
            let body = ProjectionRequest {
                activated: Some(RANDOMIZE_CONTROL.to_owned()),
                ..Default::default()
            };
            project(&state, &body)
                .projection
                .points
                .iter()
                .map(|point| point.image_id)
                .collect::<Vec<_>>()
        };

        assert_ne!(
            press(),
            press(),
            "pressing again must keep producing new arrangements"
        );
    }
}
