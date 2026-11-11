//! Alert core: arm state, confirmed alerts and a bounded event history.
//!
//! Input is one [`ZoneSignal`] per zone and tick, taken from the ADR-424 fused
//! zones. A single signal never raises an alert; each rule needs a confirming
//! condition that holds over time:
//!
//! - **intrusion** — armed-away, exit delay over, presence held for
//!   `intrusion_persist_ms` in a zone that is not ignored;
//! - **possible fall** — a fall candidate followed by `fall_stillness_ms` of
//!   presence with low motion. The person leaving the zone, or the candidate
//!   ageing past `fall_window_ms`, cancels it;
//! - **sensor offline** — a zone that had a live source has had none for
//!   `offline_after_ms`. A blind sensor is reported, never read as "empty".
//!
//! Every alert carries the evidence grade of the readings behind it
//! (MEASURED / CLAIMED / SYNTHETIC). The thresholds are defaults, not measured
//! values: no false-alarm or miss rate is claimed for any rule.
//!
//! Pure logic: no I/O, no clock. Callers pass `now_ms`. History is in memory
//! and is lost on restart.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

/// Most alerts kept; the oldest closed ones are dropped first.
pub const MAX_ALERTS: usize = 256;
/// Most history events kept.
pub const MAX_EVENTS: usize = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArmMode {
    #[default]
    Disarmed,
    /// People are home: intrusion is off, fall and sensor rules stay on.
    ArmedHome,
    /// Home should be empty: sustained presence is an intrusion.
    ArmedAway,
}

/// One zone's state for one tick.
#[derive(Debug, Clone, PartialEq)]
pub struct ZoneSignal {
    pub room: Option<String>,
    pub zone: String,
    /// `None` when no healthy, fresh source covers the zone.
    pub presence_probability: Option<f64>,
    /// Strongest motion reported by a fresh source, `[0, 1]`; `None` if none reports it.
    pub motion: Option<f64>,
    /// A fresh source reports a fall candidate.
    pub fall: bool,
    /// Weakest evidence grade among the readings used.
    pub evidence: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertKind {
    Intrusion,
    PossibleFall,
    SensorOffline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertStatus {
    Open,
    Acknowledged,
    /// The condition ended by itself (a sensor came back).
    Resolved,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Alert {
    pub id: u64,
    pub kind: AlertKind,
    pub room: Option<String>,
    pub zone: String,
    pub raised_ms: i64,
    pub status: AlertStatus,
    pub closed_ms: Option<i64>,
    pub evidence: &'static str,
    /// Why it fired, as short tags.
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    ModeChanged,
    AlertRaised,
    AlertAcknowledged,
    AlertResolved,
    FallCandidate,
    FallCandidateCancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Event {
    pub id: u64,
    pub timestamp_ms: i64,
    pub kind: EventKind,
    pub zone: Option<String>,
    pub alert_id: Option<u64>,
    pub detail: String,
}

/// Rule thresholds. Defaults are starting points, not measured values.
#[derive(Debug, Clone, PartialEq)]
pub struct AlertConfig {
    /// Time after arming-away before intrusion can fire (time to leave).
    pub exit_delay_ms: i64,
    /// Presence must hold this long before an intrusion alert.
    pub intrusion_persist_ms: i64,
    /// Fused probability at or above this counts as present.
    pub present_at: f64,
    /// Fused probability at or below this counts as absent.
    pub absent_at: f64,
    /// Stillness after a fall candidate needed to confirm it.
    pub fall_stillness_ms: i64,
    /// A fall candidate not confirmed within this window is dropped.
    pub fall_window_ms: i64,
    /// Motion at or below this counts as still.
    pub still_motion_max: f64,
    /// A zone with no live source for this long is reported offline.
    pub offline_after_ms: i64,
    /// After an acknowledgement, the same alert cannot re-fire for this long.
    pub cooldown_ms: i64,
    /// Zones where presence never counts as intrusion (for example a pet's room).
    pub ignore_intrusion_zones: BTreeSet<String>,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            exit_delay_ms: 60_000,
            intrusion_persist_ms: 10_000,
            present_at: 0.7,
            absent_at: 0.3,
            fall_stillness_ms: 30_000,
            fall_window_ms: 120_000,
            still_motion_max: 0.1,
            offline_after_ms: 60_000,
            cooldown_ms: 300_000,
            ignore_intrusion_zones: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AckError {
    #[error("no alert with that id")]
    NotFound,
    #[error("alert is already closed")]
    AlreadyClosed,
}

#[derive(Debug, Clone)]
struct PendingFall {
    started_ms: i64,
    still_since_ms: Option<i64>,
    motion_seen: bool,
    evidence: &'static str,
}

#[derive(Debug, Clone, Default)]
struct ZoneTrack {
    room: Option<String>,
    present_since_ms: Option<i64>,
    /// Set once a live source has covered the zone.
    covered: bool,
    dark_since_ms: Option<i64>,
    /// Previous tick's fall flag, so a held flag opens one candidate.
    fall_prev: bool,
    pending_fall: Option<PendingFall>,
}

/// Of two evidence grades, the one that supports less.
fn weaker(a: &'static str, b: &'static str) -> &'static str {
    fn rank(g: &str) -> u8 {
        match g {
            "MEASURED" => 0,
            "CLAIMED" => 1,
            "SYNTHETIC" => 2,
            _ => 3,
        }
    }
    if rank(b) > rank(a) { b } else { a }
}

#[derive(Debug, Default)]
pub struct AlertCore {
    config: AlertConfig,
    mode: ArmMode,
    mode_since_ms: i64,
    zones: BTreeMap<String, ZoneTrack>,
    alerts: VecDeque<Alert>,
    events: VecDeque<Event>,
    next_alert_id: u64,
    next_event_id: u64,
}

impl AlertCore {
    pub fn new(config: AlertConfig) -> Self {
        Self { config, ..Self::default() }
    }

    pub fn mode(&self) -> ArmMode {
        self.mode
    }

    pub fn alerts(&self) -> impl Iterator<Item = &Alert> {
        self.alerts.iter()
    }

    /// History events with an id greater than `after`, oldest first.
    pub fn events_after(&self, after: u64) -> impl Iterator<Item = &Event> {
        self.events.iter().filter(move |e| e.id > after)
    }

    fn log(&mut self, now_ms: i64, kind: EventKind, zone: Option<&str>, alert_id: Option<u64>, detail: String) {
        self.next_event_id += 1;
        if self.events.len() == MAX_EVENTS {
            self.events.pop_front();
        }
        self.events.push_back(Event {
            id: self.next_event_id,
            timestamp_ms: now_ms,
            kind,
            zone: zone.map(str::to_string),
            alert_id,
            detail,
        });
    }

    /// Change the arm mode. Disarming acknowledges open intrusion alerts and
    /// clears presence timers, so re-arming starts from a clean state.
    pub fn set_mode(&mut self, mode: ArmMode, now_ms: i64) {
        if mode == self.mode {
            return;
        }
        self.mode = mode;
        self.mode_since_ms = now_ms;
        self.log(now_ms, EventKind::ModeChanged, None, None, format!("{mode:?}"));
        for t in self.zones.values_mut() {
            t.present_since_ms = None;
        }
        if mode == ArmMode::Disarmed {
            let open: Vec<u64> = self
                .alerts
                .iter()
                .filter(|a| a.kind == AlertKind::Intrusion && a.status == AlertStatus::Open)
                .map(|a| a.id)
                .collect();
            for id in open {
                let _ = self.acknowledge(id, now_ms);
            }
        }
    }

    pub fn acknowledge(&mut self, id: u64, now_ms: i64) -> Result<(), AckError> {
        let alert = self.alerts.iter_mut().find(|a| a.id == id).ok_or(AckError::NotFound)?;
        if alert.status != AlertStatus::Open {
            return Err(AckError::AlreadyClosed);
        }
        alert.status = AlertStatus::Acknowledged;
        alert.closed_ms = Some(now_ms);
        let zone = alert.zone.clone();
        self.log(now_ms, EventKind::AlertAcknowledged, Some(&zone), Some(id), String::new());
        Ok(())
    }

    /// True while an alert of this kind and zone is open or in cooldown.
    fn suppressed(&self, kind: AlertKind, zone: &str, now_ms: i64) -> bool {
        self.alerts.iter().any(|a| {
            a.kind == kind
                && a.zone == zone
                && match a.status {
                    AlertStatus::Open => true,
                    AlertStatus::Acknowledged => {
                        a.closed_ms.is_some_and(|t| now_ms.saturating_sub(t) < self.config.cooldown_ms)
                    }
                    AlertStatus::Resolved => false,
                }
        })
    }

    fn raise(
        &mut self,
        now_ms: i64,
        kind: AlertKind,
        room: Option<String>,
        zone: &str,
        evidence: &'static str,
        reasons: Vec<String>,
    ) {
        if self.suppressed(kind, zone, now_ms) {
            return;
        }
        if self.alerts.len() == MAX_ALERTS {
            // Drop the oldest closed alert; if all are open, the oldest of those.
            let at = self.alerts.iter().position(|a| a.status != AlertStatus::Open).unwrap_or(0);
            self.alerts.remove(at);
        }
        self.next_alert_id += 1;
        let id = self.next_alert_id;
        self.log(now_ms, EventKind::AlertRaised, Some(zone), Some(id), reasons.join(","));
        self.alerts.push_back(Alert {
            id,
            kind,
            room,
            zone: zone.to_string(),
            raised_ms: now_ms,
            status: AlertStatus::Open,
            closed_ms: None,
            evidence,
            reasons,
        });
    }

    /// Advance every rule by one tick. A zone seen before but missing from
    /// `signals` is treated as having no live source.
    pub fn step(&mut self, now_ms: i64, signals: &[ZoneSignal]) {
        for s in signals {
            let t = self.zones.entry(s.zone.clone()).or_default();
            t.room.clone_from(&s.room);
        }
        let names: Vec<String> = self.zones.keys().cloned().collect();
        for name in names {
            let dark = ZoneSignal {
                room: None,
                zone: name.clone(),
                presence_probability: None,
                motion: None,
                fall: false,
                evidence: "NONE",
            };
            let s = signals.iter().find(|s| s.zone == name).unwrap_or(&dark);
            self.step_coverage(now_ms, &name, s);
            self.step_intrusion(now_ms, &name, s);
            self.step_fall(now_ms, &name, s);
        }
    }

    fn step_coverage(&mut self, now_ms: i64, name: &str, s: &ZoneSignal) {
        let offline_after = self.config.offline_after_ms;
        let Some(t) = self.zones.get_mut(name) else { return };
        if s.presence_probability.is_some() {
            t.covered = true;
            t.dark_since_ms = None;
            let back: Vec<u64> = self
                .alerts
                .iter()
                .filter(|a| a.kind == AlertKind::SensorOffline && a.zone == name && a.status == AlertStatus::Open)
                .map(|a| a.id)
                .collect();
            for id in back {
                if let Some(a) = self.alerts.iter_mut().find(|a| a.id == id) {
                    a.status = AlertStatus::Resolved;
                    a.closed_ms = Some(now_ms);
                }
                self.log(now_ms, EventKind::AlertResolved, Some(name), Some(id), "source back".into());
            }
            return;
        }
        if !t.covered {
            return;
        }
        let since = *t.dark_since_ms.get_or_insert(now_ms);
        if now_ms.saturating_sub(since) >= offline_after {
            let room = t.room.clone();
            self.raise(
                now_ms,
                AlertKind::SensorOffline,
                room,
                name,
                "NONE",
                vec![format!("no_live_source_ms={}", now_ms.saturating_sub(since))],
            );
        }
    }

    fn step_intrusion(&mut self, now_ms: i64, name: &str, s: &ZoneSignal) {
        let c = &self.config;
        let armed = self.mode == ArmMode::ArmedAway
            && now_ms.saturating_sub(self.mode_since_ms) >= c.exit_delay_ms
            && !c.ignore_intrusion_zones.contains(name);
        let present = s.presence_probability.is_some_and(|p| p >= c.present_at);
        let persist = c.intrusion_persist_ms;
        let Some(t) = self.zones.get_mut(name) else { return };
        if !armed || !present {
            // Anything short of confident presence restarts the timer.
            t.present_since_ms = None;
            return;
        }
        let since = *t.present_since_ms.get_or_insert(now_ms);
        let held = now_ms.saturating_sub(since);
        if held >= persist {
            let room = t.room.clone();
            self.raise(
                now_ms,
                AlertKind::Intrusion,
                room,
                name,
                s.evidence,
                vec!["armed_away".into(), format!("presence_held_ms={held}")],
            );
        }
    }

    fn step_fall(&mut self, now_ms: i64, name: &str, s: &ZoneSignal) {
        let c = self.config.clone();
        let Some(t) = self.zones.get_mut(name) else { return };
        let rising = s.fall && !t.fall_prev;
        t.fall_prev = s.fall;
        let mut log: Option<(EventKind, String)> = None;
        let mut confirmed: Option<(&'static str, Vec<String>)> = None;

        if rising && t.pending_fall.is_none() {
            t.pending_fall = Some(PendingFall {
                started_ms: now_ms,
                still_since_ms: None,
                motion_seen: false,
                evidence: s.evidence,
            });
            log = Some((EventKind::FallCandidate, String::new()));
        } else if let Some(p) = t.pending_fall.as_mut() {
            p.evidence = weaker(p.evidence, s.evidence);
            let absent = s.presence_probability.is_some_and(|x| x <= c.absent_at);
            let present = s.presence_probability.is_some_and(|x| x >= c.present_at);
            if absent {
                t.pending_fall = None;
                log = Some((EventKind::FallCandidateCancelled, "left_zone".into()));
            } else if now_ms.saturating_sub(p.started_ms) > c.fall_window_ms {
                t.pending_fall = None;
                log = Some((EventKind::FallCandidateCancelled, "not_confirmed_in_window".into()));
            } else {
                p.motion_seen |= s.motion.is_some();
                // Unknown motion cannot rule stillness out, so it counts as
                // still and the alert says the stillness was not checked.
                let still = present && s.motion.map_or(true, |m| m <= c.still_motion_max);
                if !still {
                    p.still_since_ms = None;
                } else {
                    let since = *p.still_since_ms.get_or_insert(now_ms);
                    let held = now_ms.saturating_sub(since);
                    if held >= c.fall_stillness_ms {
                        let mut reasons = vec!["fall_candidate".into(), format!("still_ms={held}")];
                        if !p.motion_seen {
                            reasons.push("motion_unknown".into());
                        }
                        confirmed = Some((p.evidence, reasons));
                        t.pending_fall = None;
                    }
                }
            }
        }

        let room = t.room.clone();
        if let Some((kind, detail)) = log {
            self.log(now_ms, kind, Some(name), None, detail);
        }
        if let Some((evidence, reasons)) = confirmed {
            self.raise(now_ms, AlertKind::PossibleFall, room, name, evidence, reasons);
        }
    }

    /// Response body of `GET /api/v1/alerts`.
    pub fn status_json(&self, now_ms: i64) -> serde_json::Value {
        let exit_left = match self.mode {
            ArmMode::ArmedAway => {
                (self.config.exit_delay_ms - now_ms.saturating_sub(self.mode_since_ms)).max(0)
            }
            _ => 0,
        };
        let pending: Vec<&String> =
            self.zones.iter().filter(|(_, t)| t.pending_fall.is_some()).map(|(z, _)| z).collect();
        serde_json::json!({
            "timestamp_ms": now_ms,
            "mode": self.mode,
            "mode_since_ms": self.mode_since_ms,
            "exit_delay_remaining_ms": exit_left,
            "thresholds_measured": false,
            "pending_fall_zones": pending,
            "open": self.alerts.iter().filter(|a| a.status == AlertStatus::Open).collect::<Vec<_>>(),
            "closed": self.alerts.iter().filter(|a| a.status != AlertStatus::Open).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: i64 = 1_794_300_000_000;

    fn sig(zone: &str, p: Option<f64>, motion: Option<f64>, fall: bool) -> ZoneSignal {
        ZoneSignal {
            room: None,
            zone: zone.to_string(),
            presence_probability: p,
            motion,
            fall,
            evidence: "SYNTHETIC",
        }
    }

    /// Step once a second from `from` to `to` (seconds after T0) with one signal.
    fn run(core: &mut AlertCore, from: i64, to: i64, s: &ZoneSignal) {
        for sec in from..=to {
            core.step(T0 + sec * 1000, std::slice::from_ref(s));
        }
    }

    fn open(core: &AlertCore, kind: AlertKind) -> usize {
        core.alerts().filter(|a| a.kind == kind && a.status == AlertStatus::Open).count()
    }

    #[test]
    fn presence_while_disarmed_or_armed_home_is_not_an_intrusion() {
        let mut core = AlertCore::default();
        run(&mut core, 0, 300, &sig("hall", Some(0.95), Some(1.0), false));
        core.set_mode(ArmMode::ArmedHome, T0 + 301_000);
        run(&mut core, 301, 600, &sig("hall", Some(0.95), Some(1.0), false));
        assert_eq!(open(&core, AlertKind::Intrusion), 0);
    }

    #[test]
    fn intrusion_needs_exit_delay_then_sustained_presence() {
        let mut core = AlertCore::default();
        core.set_mode(ArmMode::ArmedAway, T0);
        // Present throughout the exit delay: the person leaving is not an intruder.
        run(&mut core, 0, 59, &sig("hall", Some(0.9), None, false));
        assert_eq!(open(&core, AlertKind::Intrusion), 0);
        // Still present after the delay, but not yet for the persist time.
        run(&mut core, 60, 69, &sig("hall", Some(0.9), None, false));
        assert_eq!(open(&core, AlertKind::Intrusion), 0);
        core.step(T0 + 70_000, &[sig("hall", Some(0.9), None, false)]);
        assert_eq!(open(&core, AlertKind::Intrusion), 1);
        let a = core.alerts().next().unwrap();
        assert_eq!(a.evidence, "SYNTHETIC", "the alert keeps the grade of its readings");
    }

    #[test]
    fn a_brief_or_uncertain_blip_does_not_alarm() {
        let mut core = AlertCore::default();
        core.set_mode(ArmMode::ArmedAway, T0);
        run(&mut core, 0, 100, &sig("hall", Some(0.1), None, false));
        run(&mut core, 101, 108, &sig("hall", Some(0.9), None, false));
        // An uncertain reading restarts the timer.
        core.step(T0 + 109_000, &[sig("hall", Some(0.5), None, false)]);
        run(&mut core, 110, 118, &sig("hall", Some(0.9), None, false));
        assert_eq!(open(&core, AlertKind::Intrusion), 0);
    }

    #[test]
    fn ignored_zone_never_raises_intrusion() {
        let mut cfg = AlertConfig::default();
        cfg.ignore_intrusion_zones.insert("cat_room".into());
        let mut core = AlertCore::new(cfg);
        core.set_mode(ArmMode::ArmedAway, T0);
        run(&mut core, 0, 600, &sig("cat_room", Some(0.95), Some(1.0), false));
        assert_eq!(open(&core, AlertKind::Intrusion), 0);
    }

    #[test]
    fn one_alert_per_zone_and_cooldown_after_acknowledge() {
        let mut core = AlertCore::default();
        core.set_mode(ArmMode::ArmedAway, T0);
        let s = sig("hall", Some(0.9), None, false);
        run(&mut core, 0, 200, &s);
        assert_eq!(core.alerts().count(), 1, "a held condition raises once");
        let id = core.alerts().next().unwrap().id;
        core.acknowledge(id, T0 + 200_000).unwrap();
        assert_eq!(core.acknowledge(id, T0 + 200_000), Err(AckError::AlreadyClosed));
        assert_eq!(core.acknowledge(999, T0 + 200_000), Err(AckError::NotFound));
        run(&mut core, 201, 499, &s);
        assert_eq!(core.alerts().count(), 1, "no re-fire inside the cooldown");
        run(&mut core, 500, 501, &s);
        assert_eq!(core.alerts().count(), 2, "still present after the cooldown");
    }

    #[test]
    fn disarming_acknowledges_intrusion_and_resets_timers() {
        let mut core = AlertCore::default();
        core.set_mode(ArmMode::ArmedAway, T0);
        run(&mut core, 0, 80, &sig("hall", Some(0.9), None, false));
        assert_eq!(open(&core, AlertKind::Intrusion), 1);
        core.set_mode(ArmMode::Disarmed, T0 + 81_000);
        assert_eq!(open(&core, AlertKind::Intrusion), 0);
        assert_eq!(core.alerts().next().unwrap().status, AlertStatus::Acknowledged);
    }

    #[test]
    fn fall_candidate_then_stillness_confirms() {
        let mut core = AlertCore::default();
        run(&mut core, 0, 5, &sig("bath", Some(0.9), Some(0.8), false));
        core.step(T0 + 6_000, &[sig("bath", Some(0.9), Some(0.9), true)]);
        assert_eq!(open(&core, AlertKind::PossibleFall), 0, "a candidate alone is not an alert");
        run(&mut core, 7, 36, &sig("bath", Some(0.9), Some(0.02), false));
        assert_eq!(open(&core, AlertKind::PossibleFall), 0);
        core.step(T0 + 37_000, &[sig("bath", Some(0.9), Some(0.02), false)]);
        assert_eq!(open(&core, AlertKind::PossibleFall), 1);
        let a = core.alerts().next().unwrap();
        assert!(!a.reasons.iter().any(|r| r == "motion_unknown"));
    }

    #[test]
    fn getting_up_again_does_not_confirm_a_fall() {
        let mut core = AlertCore::default();
        core.step(T0, &[sig("bath", Some(0.9), Some(0.9), true)]);
        // Still for 20 s, then moving: the stillness timer restarts each time.
        for round in 0..5 {
            let base = 1 + round * 25;
            run(&mut core, base, base + 19, &sig("bath", Some(0.9), Some(0.02), false));
            run(&mut core, base + 20, base + 24, &sig("bath", Some(0.9), Some(0.6), false));
        }
        assert_eq!(open(&core, AlertKind::PossibleFall), 0);
        // Past the window the candidate is dropped, so later stillness is not a fall.
        run(&mut core, 126, 200, &sig("bath", Some(0.9), Some(0.02), false));
        assert_eq!(open(&core, AlertKind::PossibleFall), 0);
        assert!(core
            .events_after(0)
            .any(|e| e.kind == EventKind::FallCandidateCancelled && e.detail == "not_confirmed_in_window"));
    }

    #[test]
    fn leaving_the_zone_cancels_a_fall_candidate() {
        let mut core = AlertCore::default();
        core.step(T0, &[sig("bath", Some(0.9), Some(0.9), true)]);
        run(&mut core, 1, 10, &sig("bath", Some(0.9), Some(0.02), false));
        core.step(T0 + 11_000, &[sig("bath", Some(0.05), Some(0.0), false)]);
        run(&mut core, 12, 100, &sig("bath", Some(0.9), Some(0.02), false));
        assert_eq!(open(&core, AlertKind::PossibleFall), 0);
    }

    #[test]
    fn fall_without_any_motion_reading_fires_and_says_so() {
        let mut core = AlertCore::default();
        core.step(T0, &[sig("bath", Some(0.9), None, true)]);
        run(&mut core, 1, 31, &sig("bath", Some(0.9), None, false));
        let a = core.alerts().find(|a| a.kind == AlertKind::PossibleFall).expect("raised");
        assert!(a.reasons.iter().any(|r| r == "motion_unknown"));
    }

    #[test]
    fn a_held_fall_flag_opens_one_candidate() {
        let mut core = AlertCore::default();
        run(&mut core, 0, 40, &sig("bath", Some(0.9), Some(0.02), true));
        assert_eq!(core.alerts().count(), 1);
        core.acknowledge(1, T0 + 41_000).unwrap();
        run(&mut core, 41, 400, &sig("bath", Some(0.9), Some(0.02), true));
        assert_eq!(core.alerts().count(), 1, "the same held flag is not a second fall");
    }

    #[test]
    fn fall_alert_takes_the_weakest_evidence_seen() {
        let mut core = AlertCore::default();
        let mut first = sig("bath", Some(0.9), Some(0.9), true);
        first.evidence = "MEASURED";
        core.step(T0, &[first]);
        let mut later = sig("bath", Some(0.9), Some(0.0), false);
        later.evidence = "CLAIMED";
        run(&mut core, 1, 31, &later);
        assert_eq!(core.alerts().next().unwrap().evidence, "CLAIMED");
    }

    #[test]
    fn a_dark_sensor_is_reported_and_never_read_as_empty() {
        let mut core = AlertCore::default();
        core.set_mode(ArmMode::ArmedAway, T0);
        run(&mut core, 0, 100, &sig("hall", Some(0.05), None, false));
        // The source goes stale; the zone drops out of the signals entirely.
        for sec in 101..=160 {
            core.step(T0 + sec * 1000, &[]);
        }
        assert_eq!(open(&core, AlertKind::SensorOffline), 0);
        core.step(T0 + 161_000, &[]);
        assert_eq!(open(&core, AlertKind::SensorOffline), 1);
        assert_eq!(open(&core, AlertKind::Intrusion), 0);
        // It comes back: the alert resolves without anyone acknowledging it.
        core.step(T0 + 170_000, &[sig("hall", Some(0.05), None, false)]);
        assert_eq!(open(&core, AlertKind::SensorOffline), 0);
        assert!(core.alerts().any(|a| a.status == AlertStatus::Resolved));
    }

    #[test]
    fn a_zone_that_never_had_a_source_is_not_offline() {
        let mut core = AlertCore::default();
        run(&mut core, 0, 300, &sig("attic", None, None, false));
        assert_eq!(core.alerts().count(), 0);
    }

    #[test]
    fn history_is_ordered_bounded_and_pageable() {
        let mut core = AlertCore::default();
        for i in 0..(MAX_EVENTS as i64 + 50) {
            let mode = if i % 2 == 0 { ArmMode::ArmedHome } else { ArmMode::Disarmed };
            core.set_mode(mode, T0 + i);
        }
        let all: Vec<&Event> = core.events_after(0).collect();
        assert_eq!(all.len(), MAX_EVENTS);
        assert!(all.windows(2).all(|w| w[0].id < w[1].id));
        let last = all.last().unwrap().id;
        assert_eq!(core.events_after(last - 3).count(), 3);
    }

    #[test]
    fn alert_store_is_bounded() {
        let cfg = AlertConfig { cooldown_ms: 0, ..AlertConfig::default() };
        let mut core = AlertCore::new(cfg);
        core.set_mode(ArmMode::ArmedAway, T0);
        let s = sig("hall", Some(0.9), None, false);
        run(&mut core, 0, 70, &s);
        for i in 0..(MAX_ALERTS as i64 + 20) {
            let now = T0 + (71 + i) * 1000;
            let id = core.alerts().last().unwrap().id;
            core.acknowledge(id, now).unwrap();
            core.step(now, std::slice::from_ref(&s));
        }
        assert_eq!(core.alerts().count(), MAX_ALERTS);
        assert_eq!(open(&core, AlertKind::Intrusion), 1, "the newest alert is kept");
    }

    #[test]
    fn status_reports_mode_and_remaining_exit_delay() {
        let mut core = AlertCore::default();
        core.set_mode(ArmMode::ArmedAway, T0);
        let v = core.status_json(T0 + 20_000);
        assert_eq!(v["mode"], "armed_away");
        assert_eq!(v["exit_delay_remaining_ms"], 40_000);
        assert_eq!(v["thresholds_measured"], false);
    }
}
