//! Blocking HTTP client for a server that implements the viewer's API.
//!
//! The contract is three endpoints — `GET /health`, `POST /catalog/stream`
//! (NDJSON snapshots) and `POST /projection` (one snapshot) — and the
//! control-panel schema in [`controls`]. Nothing here is specific to any
//! particular server: the viewer sends folder roots plus a map of control
//! values, and receives points plus the panel that produced them.
//!
//! Positions are in cube units: every billboard fills a 1x1x1 cube, and a
//! server lays cubes out touching. A point may also name the group it
//! belongs to (see [`PointGroup`]); the viewer shows a group of several
//! points as one folder the user opens and closes, and lays the groups out
//! itself.

mod controls;

pub use controls::{
    ControlOption, ControlPanel, ControlValue, ControlValues, ControlWidget, StatLine,
};

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::time::Duration;

/// The blocking client applies this to each wait — response headers and every
/// body read — not to the whole request, so a catalog stream may run for any
/// length of time as long as the server keeps sending heartbeats inside it.
const READ_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct SpatialApiClient {
    base_url: String,
    client: Client,
}

#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProjectionPoint {
    pub image_id: usize,
    pub path: String,
    /// Cube center in cube units, with neighbouring cubes touching and
    /// every group laid out open. A viewer that lays groups out itself uses
    /// `group` instead.
    pub position: [f32; 3],
    /// The group this point belongs to; `None` from a server without groups.
    #[serde(default)]
    pub group: Option<PointGroup>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub media_type: String,
    pub duration_seconds: Option<f32>,
    /// Human-readable value backing each axis (e.g. an acquisition date or
    /// seed), aligned to `ProjectionPage::axis_labels`; `None` where an axis
    /// carries no value for this point.
    pub coordinate_labels: [Option<String>; 3],
}

/// Where a point sits among the points it is grouped with, for example every
/// image sharing one coordinate value.
///
/// Groups form a grid: along each axis a group occupies one slot, `index`,
/// and its points pack into a block of cubes centered on the slot. The
/// viewer sizes every slot to what it currently shows there, so a closed
/// folder takes one cube and an open one its whole block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PointGroup {
    /// Names the group across snapshots, which the slot rank cannot do: a
    /// value discovered later shifts the ranks of everything after it.
    pub key: String,
    /// The group's slot along each axis, counting from 0 at the low end.
    pub index: [u32; 3],
    /// The point's cube center relative to the center of its group's block,
    /// in cube units.
    pub offset: [f32; 3],
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProjectionPage {
    /// What each world axis represents, as the server names it. The gizmo and
    /// billboard coordinate labels render these directly, so the viewer needs
    /// no notion of what produced the layout.
    #[serde(default)]
    pub axis_labels: [Option<String>; 3],
    pub offset: usize,
    pub limit: usize,
    pub total: usize,
    pub points: Vec<ProjectionPoint>,
}

/// One request body, shared by `/catalog/stream` and `/projection`.
///
/// Every control value travels on every request: the server holds no
/// per-client state, so a reconnect or a retry cannot desynchronize. The
/// `activated` id says which control the user just interacted with, which is
/// what distinguishes a button press (same values, do something) from a
/// dropdown change.
#[derive(Debug, Clone, Serialize)]
pub struct ProjectionRequest {
    pub roots: Vec<String>,
    pub control_values: ControlValues,
    pub activated: Option<String>,
    pub offset: usize,
    pub limit: usize,
}

impl ProjectionRequest {
    /// A request with no controls set, which is what the viewer sends before
    /// a server has described its panel.
    pub fn new(limit: usize) -> Self {
        Self {
            roots: Vec::new(),
            control_values: ControlValues::new(),
            activated: None,
            offset: 0,
            limit,
        }
    }

    pub fn with_roots(mut self, roots: Vec<String>) -> Self {
        self.roots = roots;
        self
    }

    pub fn with_controls(mut self, values: ControlValues, activated: Option<String>) -> Self {
        self.control_values = values;
        self.activated = activated;
        self
    }
}

/// Each snapshot is a complete view of the records discovered so far, paired
/// with the panel describing the controls that produced it. Coordinates may
/// change between snapshots as more metadata is discovered. A server with
/// nothing to stream may send a single `complete` snapshot.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CatalogStreamEvent {
    Snapshot {
        #[serde(default)]
        panel: ControlPanel,
        /// Boxed because a snapshot dwarfs the error variant, and the enum is
        /// moved once per streamed line.
        projection: Box<ProjectionPage>,
        complete: bool,
        #[serde(default)]
        roots: Vec<String>,
    },
    Error {
        message: String,
    },
    /// Sent while the server is busy with nothing new to report, so a slow
    /// load never trips the read timeout. Receiving one also gives the
    /// callback a chance to cancel an idle stream.
    Heartbeat,
}

impl SpatialApiClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            client: Client::builder()
                .timeout(READ_TIMEOUT)
                .build()
                .expect("HTTP client configuration is valid"),
        }
    }

    pub fn health(&self) -> reqwest::Result<HealthResponse> {
        self.get_json("/health")
    }

    /// Consume NDJSON as it arrives instead of buffering the HTTP body. Returning
    /// false from the callback cancels the request when the viewer drops a load.
    pub fn stream_catalog(
        &self,
        request: &ProjectionRequest,
        mut receive: impl FnMut(CatalogStreamEvent) -> bool,
    ) -> Result<(), String> {
        let response = self
            .client
            .post(format!("{}/catalog/stream", self.base_url))
            .json(request)
            .send()
            .and_then(|response| response.error_for_status())
            .map_err(|error| format!("Catalog stream failed: {error}"))?;
        let mut complete = false;
        for line in BufReader::new(response).lines() {
            let line = line.map_err(|error| format!("Catalog stream interrupted: {error}"))?;
            if line.trim().is_empty() {
                continue;
            }
            let event: CatalogStreamEvent = serde_json::from_str(&line)
                .map_err(|error| format!("Invalid catalog stream event: {error}"))?;
            complete = matches!(&event, CatalogStreamEvent::Snapshot { complete: true, .. });
            let failed = matches!(&event, CatalogStreamEvent::Error { .. });
            if !receive(event) || failed {
                return Ok(());
            }
            if complete {
                break;
            }
        }
        if complete {
            Ok(())
        } else {
            Err("Catalog stream ended before completion.".to_owned())
        }
    }

    /// Reproject without rescanning the roots, for a control change against a
    /// catalog the server already holds.
    pub fn projection(&self, request: &ProjectionRequest) -> reqwest::Result<CatalogSnapshot> {
        self.client
            .post(format!("{}/projection", self.base_url))
            .json(request)
            .send()?
            .error_for_status()?
            .json()
    }

    fn get_json<T>(&self, path: &str) -> reqwest::Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.client
            .get(format!("{}{}", self.base_url, path))
            .send()?
            .error_for_status()?
            .json()
    }
}

/// The non-streaming form of a snapshot, returned by `/projection`.
#[derive(Debug, Default, Deserialize)]
pub struct CatalogSnapshot {
    #[serde(default)]
    pub panel: ControlPanel,
    pub projection: ProjectionPage,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::mpsc,
        time::Duration,
    };

    fn snapshot(complete: bool) -> String {
        serde_json::json!({
            "kind": "snapshot", "roots": ["fixture"], "complete": complete,
            "panel": {"revision": 1, "title": "Axes", "summary": "0 pts", "widgets": []},
            "projection": {"axis_labels": [null,null,null],
                "offset": 0, "limit": 1, "total": 0, "points": []}
        })
        .to_string()
            + "\n"
    }

    fn request() -> ProjectionRequest {
        ProjectionRequest::new(100).with_roots(vec!["fixture".to_owned()])
    }

    #[test]
    fn catalog_stream_delivers_first_event_before_response_finishes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = SpatialApiClient::new(format!("http://{}", listener.local_addr().unwrap()));
        let (first_received, wait_for_first) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut buffer = [0; 8192];
            let _request_bytes = socket.read(&mut buffer).unwrap();
            let first = snapshot(false);
            let final_event = snapshot(true);
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{first}", first.len() + final_event.len()).unwrap();
            socket.flush().unwrap();
            wait_for_first
                .recv_timeout(Duration::from_secs(5))
                .expect("client buffered the stream");
            socket.write_all(final_event.as_bytes()).unwrap();
        });
        let mut events = 0;
        client
            .stream_catalog(&request(), |_| {
                events += 1;
                if events == 1 {
                    first_received.send(()).unwrap();
                }
                true
            })
            .unwrap();
        server.join().unwrap();
        assert_eq!(events, 2);
    }

    #[test]
    fn catalog_stream_passes_heartbeats_through_until_completion() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = SpatialApiClient::new(format!("http://{}", listener.local_addr().unwrap()));
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut buffer = [0; 8192];
            let _request_bytes = socket.read(&mut buffer).unwrap();
            let body = format!(
                "{}{{\"kind\":\"heartbeat\"}}\n{}",
                snapshot(false),
                snapshot(true)
            );
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        });
        let mut kinds = Vec::new();
        client
            .stream_catalog(&request(), |event| {
                kinds.push(match event {
                    CatalogStreamEvent::Snapshot { complete, .. } => {
                        if complete {
                            "complete"
                        } else {
                            "partial"
                        }
                    }
                    CatalogStreamEvent::Error { .. } => "error",
                    CatalogStreamEvent::Heartbeat => "heartbeat",
                });
                true
            })
            .unwrap();
        server.join().unwrap();
        assert_eq!(kinds, ["partial", "heartbeat", "complete"]);
    }

    #[test]
    fn catalog_stream_rejects_missing_completion() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = SpatialApiClient::new(format!("http://{}", listener.local_addr().unwrap()));
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut buffer = [0; 8192];
            let _request_bytes = socket.read(&mut buffer).unwrap();
            let body = snapshot(false);
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        });
        let error = client.stream_catalog(&request(), |_| true).unwrap_err();
        server.join().unwrap();
        assert!(error.contains("before completion"), "{error}");
    }

    /// The whole control contract is carried in the request body, so a server
    /// receives values and the activated id without any query-string encoding.
    #[test]
    fn requests_carry_every_control_value_and_the_activated_control() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = SpatialApiClient::new(format!("http://{}", listener.local_addr().unwrap()));
        let (body_sender, body_receiver) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut buffer = [0; 8192];
            let read = socket.read(&mut buffer).unwrap();
            body_sender
                .send(String::from_utf8_lossy(&buffer[..read]).into_owned())
                .unwrap();
            let body = serde_json::json!({
                "panel": {"revision": 1, "widgets": []},
                "projection": {"axis_labels": ["Seed", null, null],
                    "offset": 0, "limit": 1, "total": 0, "points": []}
            })
            .to_string();
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        });
        let values = ControlValues::from([
            ("shape".to_owned(), ControlValue::Text("cube".to_owned())),
            ("radius".to_owned(), ControlValue::Number(12.5)),
        ]);
        let snapshot = client
            .projection(&request().with_controls(values, Some("randomize".to_owned())))
            .unwrap();
        let request_text = body_receiver.recv_timeout(Duration::from_secs(5)).unwrap();
        server.join().unwrap();

        assert!(
            request_text.starts_with("POST /projection"),
            "{request_text}"
        );
        assert!(request_text.contains(r#""shape":"cube""#), "{request_text}");
        assert!(request_text.contains(r#""radius":12.5"#), "{request_text}");
        assert!(
            request_text.contains(r#""activated":"randomize""#),
            "{request_text}"
        );
        assert_eq!(snapshot.projection.axis_labels[0].as_deref(), Some("Seed"));
    }
}
