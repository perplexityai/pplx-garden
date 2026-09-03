//! Optional terminal dashboard for the serve loop. Runs on its own thread,
//! owns the terminal while it is up, and asks the server to stop when the
//! user quits. Everything it shows comes from [`super::stats`].

use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph, Row, Table};
use tiny_http::Server;

use super::stats::{Event, ROLLING_WINDOW, Stats};

/// Static facts shown in the header.
pub(super) struct Banner {
    pub model: &'static str,
    pub address: String,
    pub max_seq: usize,
    pub gpu_family: i64,
}

/// How long to wait for a key press between redraws.
const TICK: Duration = Duration::from_millis(250);

/// Runs until the user quits or `events` disconnects, which happens when the
/// serve loop ends and drops its sender.
pub(super) fn run(
    banner: Banner,
    events: Receiver<Event>,
    server: Arc<Server>,
) -> Result<()> {
    // Restore the terminal before the default panic output so a panic in the
    // render loop does not leave the shell in raw mode with no echo.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        default_hook(info);
    }));
    let mut terminal = ratatui::try_init()?;
    let started = Instant::now();
    let mut stats = Stats::new();
    let mut quitting = false;
    let outcome = loop {
        let mut disconnected = false;
        loop {
            match events.try_recv() {
                Ok(event) => stats.record(event),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        let uptime = started.elapsed();
        if let Err(error) =
            terminal.draw(|frame| draw(frame, &banner, &stats, uptime, quitting))
        {
            break Err(error.into());
        }
        if disconnected {
            break Ok(());
        }
        match event::poll(TICK) {
            Ok(true) => {
                if let Ok(TermEvent::Key(key)) = event::read()
                    && key.kind == KeyEventKind::Press
                    && is_quit(key.code, key.modifiers)
                    && !quitting
                {
                    quitting = true;
                    // Takes effect at the loop's next `recv`, so a request in
                    // flight still completes.
                    server.unblock();
                }
            }
            Ok(false) => {}
            Err(error) => break Err(error.into()),
        }
    };
    ratatui::restore();
    outcome
}

fn is_quit(code: KeyCode, modifiers: KeyModifiers) -> bool {
    matches!(code, KeyCode::Char('q') | KeyCode::Esc)
        || (code == KeyCode::Char('c') && modifiers.contains(KeyModifiers::CONTROL))
}

fn draw(
    frame: &mut Frame,
    banner: &Banner,
    stats: &Stats,
    uptime: Duration,
    quitting: bool,
) {
    let [header, counters, table] = Layout::vertical([
        Constraint::Length(5),
        Constraint::Length(5),
        Constraint::Min(4),
    ])
    .areas(frame.area());

    let status = if quitting {
        "stopping after the current request".to_string()
    } else if stats.in_flight {
        "request in flight".to_string()
    } else {
        "idle".to_string()
    };
    let header_lines = vec![
        Line::from(format!(
            "lily  {}  http://{}  max-seq {}  uptime {}",
            banner.model,
            banner.address,
            banner.max_seq,
            fmt_uptime(uptime)
        )),
        Line::from(gpu_line(banner.gpu_family)),
        Line::from(format!("{status}    q / Esc / Ctrl-C to quit")),
    ];
    frame.render_widget(
        Paragraph::new(header_lines).block(Block::bordered().title("server")),
        header,
    );

    let ratio = stats
        .cache_hit_ratio()
        .map_or("n/a".to_string(), |r| format!("{:.1}%", r * 100.0));
    let rate =
        stats.rolling_tok_per_s().map_or("n/a".to_string(), |r| format!("{r:.1}"));
    let counter_lines = vec![
        Line::from(format!(
            "requests {}   ok {}   4xx {}   5xx {}   in flight {}",
            stats.total,
            stats.ok,
            stats.client_errors,
            stats.server_errors,
            u8::from(stats.in_flight)
        )),
        Line::from(format!(
            "tokens   prompt {}   cached {} ({ratio})   completion {}",
            stats.prompt_tokens, stats.cached_tokens, stats.completion_tokens
        )),
        Line::from(format!(
            "tok/s over last {ROLLING_WINDOW} completions, wall time incl. prefill: {rate}    \
             queue depth and GPU load: not exposed"
        )),
    ];
    frame.render_widget(
        Paragraph::new(counter_lines).block(Block::bordered().title("totals")),
        counters,
    );

    let rows = stats.recent.iter().rev().map(|r| {
        let (prompt, cached, completion, rate) = match r.tokens {
            Some(t) => (
                t.prompt.to_string(),
                t.cached.to_string(),
                t.completion.to_string(),
                format!(
                    "{:.1}",
                    t.completion as f64 / r.elapsed.as_secs_f64().max(1e-3)
                ),
            ),
            None => ("".into(), "".into(), "".into(), "".into()),
        };
        Row::new(vec![
            r.method.clone(),
            r.path.clone(),
            r.status.to_string(),
            r.elapsed.as_millis().to_string(),
            prompt,
            cached,
            completion,
            rate,
            r.error.clone().unwrap_or_default(),
        ])
    });
    let widths = [
        Constraint::Length(6),
        Constraint::Length(22),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Min(10),
    ];
    let header_row = Row::new(vec![
        "method", "path", "status", "ms", "prompt", "cached", "compl", "tok/s", "error",
    ])
    .style(Style::default().add_modifier(Modifier::BOLD));
    frame.render_widget(
        Table::new(rows, widths)
            .header(header_row)
            .block(Block::bordered().title("recent requests, newest first")),
        table,
    );
}

/// `H:MM:SS`, hours unbounded.
fn fmt_uptime(uptime: Duration) -> String {
    let secs = uptime.as_secs();
    format!("{}:{:02}:{:02}", secs / 3_600, (secs / 60) % 60, secs % 60)
}

fn gpu_line(family: i64) -> String {
    if family >= 10 {
        format!("GPU family {family}: native tensor units")
    } else {
        format!(
            "GPU family {family}: Metal-4 tensor ops emulated, performance not representative"
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/tui.rs"]
mod tests;
