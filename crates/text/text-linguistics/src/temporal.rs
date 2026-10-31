//! Deterministic normalization of historical temporal expressions (English and German).
//!
//! [`normalize_temporal_expressions`] finds full dates, month-year, years with a cue word or
//! era (BC/BCE/AD/CE, v./n. Chr.), ranges (`from … to`, `1914-18`), decades and ordinal
//! centuries (with `early`/`mid`/`late`), and reads each into a typed [`HistoricalDate`] with
//! its exact byte span. It records approximation, hedging and negation cues, and rejects
//! quantities, timestamps and references (`1500 people`, `the 1148 timestamp`, `page 1066`).
//!
//! Relative dates ("last year"), calendar conversion and event detection are out of scope.

use std::sync::LazyLock;

use regex::{Captures, Match, Regex};
use serde::{Deserialize, Serialize};
use text_core::TextSpan;

/// Producer identity of [`normalize_temporal_expressions`] for consumers that record provenance.
pub const TEMPORAL_NORMALIZER_PRODUCER: &str = "nlp-stack/text-linguistics/temporal";
/// Expressions below this confidence (e.g. a bare four-digit number) are dropped.
pub const TEMPORAL_MIN_CONFIDENCE: f32 = 0.6;

/// A typed historical date or date range. Years are historical (not astronomical) years:
/// negative is BC and there is no year 0.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalDate {
    /// First year of the reading (negative = BC).
    pub start_year: i32,
    /// Month 1-12 of the start, when stated.
    pub start_month: Option<u8>,
    /// Day 1-31 of the start, when stated (needs a month).
    pub start_day: Option<u8>,
    /// Last year of a range (inclusive, negative = BC).
    pub end_year: Option<i32>,
    /// Month 1-12 of the range end, when stated.
    pub end_month: Option<u8>,
    /// Day 1-31 of the range end, when stated.
    pub end_day: Option<u8>,
    /// The text marks the date as approximate (`around`, `c.`, `early 15th century`).
    pub approximate: bool,
}

impl HistoricalDate {
    /// A single year, optionally with month and day.
    pub fn point(year: i32, month: Option<u8>, day: Option<u8>) -> Self {
        Self {
            start_year: year,
            start_month: month,
            start_day: day,
            ..Self::default()
        }
    }

    /// An inclusive year range.
    pub fn years(start: i32, end: i32) -> Self {
        Self {
            start_year: start,
            end_year: Some(end),
            ..Self::default()
        }
    }

    /// Whether the reading is well-formed: no year 0, months 1-12, days 1-31 with their month,
    /// and a range end that does not precede its start.
    pub fn is_valid(&self) -> bool {
        let month_ok = |m: Option<u8>| m.is_none_or(|m| (1..=12).contains(&m));
        let day_ok = |d: Option<u8>| d.is_none_or(|d| (1..=31).contains(&d));
        if self.start_year == 0 || self.end_year == Some(0) {
            return false;
        }
        if !month_ok(self.start_month)
            || !month_ok(self.end_month)
            || !day_ok(self.start_day)
            || !day_ok(self.end_day)
        {
            return false;
        }
        if (self.start_day.is_some() && self.start_month.is_none())
            || (self.end_day.is_some() && self.end_month.is_none())
            || (self.end_year.is_none() && (self.end_month.is_some() || self.end_day.is_some()))
        {
            return false;
        }
        match self.end_year {
            Some(end_year) => {
                let key =
                    |y: i32, m: Option<u8>, d: Option<u8>| (y, m.unwrap_or(1), d.unwrap_or(1));
                key(self.start_year, self.start_month, self.start_day)
                    <= key(end_year, self.end_month, self.end_day)
            }
            None => true,
        }
    }
}

/// The surface form a temporal expression was read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalExpressionKind {
    /// Day, month and year (`14 October 1066`, `March 15, 44 BC`).
    FullDate,
    /// Month and year (`October 1066`).
    MonthYear,
    /// A year with an era or cue word (`586 BC`, `in 2010`, `im Jahr 1517`).
    Year,
    /// A year range (`from 1914 to 1918`, `1914-18`).
    Range,
    /// A decade (`1960s`, `the late 1960s`, `1960er Jahre`).
    Decade,
    /// An ordinal century (`5th century BC`, `4. Jahrhundert`).
    Century,
}

/// Whether the text states or hedges the date.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalCertainty {
    /// No hedging cue.
    Stated,
    /// The text hedges (`probably`, `vermutlich`).
    Hedged,
}

/// Whether the text affirms or denies the date.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalPolarity {
    /// No negation cue before the expression.
    Affirms,
    /// A negation precedes the expression (`not in 1054`).
    Denies,
}

/// One normalized temporal expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalExpression {
    /// UTF-8 byte span of the expression in the input text.
    pub span: TextSpan,
    /// The expression exactly as it appears in the text.
    pub text: String,
    /// The surface form it was read from.
    pub kind: TemporalExpressionKind,
    /// The typed reading.
    pub date: HistoricalDate,
    /// Rule confidence in `0.0..=1.0`.
    pub confidence: f32,
    /// Hedging of the containing text.
    pub certainty: TemporalCertainty,
    /// Negation before the expression.
    pub polarity: TemporalPolarity,
}

/// Expressions at or above [`TEMPORAL_MIN_CONFIDENCE`], and how many weaker ones were dropped.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TemporalExpressions {
    /// Kept expressions in text order.
    pub expressions: Vec<TemporalExpression>,
    /// Non-overlapping candidates dropped for low confidence.
    pub low_confidence_dropped: usize,
}

const MONTH: &str = r"((?:january|february|march|april|may|june|july|august|september|october|november|december|januar|februar|märz|maerz|mai|juni|juli|oktober|dezember|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|okt|nov|dec|dez)\b)\.?";
const ERA: &str = r"(b\.\s?c\.(?:e\.)?|(?:bce|bc|ad|ce)\b|a\.\s?d\.|c\.e\.|v\.\s?chr\.?|vor christus\b|n\.\s?chr\.?|nach christus\b)";

fn ci(pattern: &str) -> Regex {
    Regex::new(&format!(r"(?i)\b(?:{pattern})")).expect("valid date pattern")
}

static FULL_DAY_FIRST: LazyLock<Regex> = LazyLock::new(|| {
    ci(&format!(
        r"(\d{{1,2}})(?:st|nd|rd|th|\.)?\s+(?:of\s+)?{MONTH},?\s+(\d{{1,4}})(?:\s*{ERA})?"
    ))
});
static FULL_MONTH_FIRST: LazyLock<Regex> = LazyLock::new(|| {
    ci(&format!(
        r"{MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{1,4}})(?:\s*{ERA})?"
    ))
});
static MONTH_YEAR: LazyLock<Regex> =
    LazyLock::new(|| ci(&format!(r"{MONTH}\s+(?:of\s+)?(\d{{1,4}})(?:\s*{ERA})?")));
static RANGE: LazyLock<Regex> = LazyLock::new(|| {
    ci(&format!(
        r"(?:(from|between|von|zwischen)\s+)?(?:the\s+years?\s+)?(\d{{1,4}})(?:\s*{ERA})?\s*(-|–|—|to|until|till|and|bis|und)\s*(\d{{1,4}})(?:\s*{ERA})?"
    ))
});
static CENTURY: LazyLock<Regex> = LazyLock::new(|| {
    ci(&format!(
        r"(?:(early|mid|late|frühen|fruehen|späten|spaeten)[\s-]+)?(\d{{1,2}}(?:st|nd|rd|th|\.)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty-first)\s*-?\s*(century|centuries|jahrhunderts?)(?:\s*{ERA})?"
    ))
});
static DECADE: LazyLock<Regex> =
    LazyLock::new(|| ci(r"(?:(early|mid|late)[\s-]+)?(\d{3})0(?:'?s|er(?:\s+jahren?)?)"));
static ERA_YEAR: LazyLock<Regex> = LazyLock::new(|| {
    ci(&format!(
        r"(\d{{1,4}})\s*{ERA}|(?:a\.\s?d\.|ad\b|anno domini)\s+(\d{{1,4}})"
    ))
});
static CUE_YEAR: LazyLock<Regex> = LazyLock::new(|| {
    ci(&format!(
        r"(in the year|the year|im jahre?|in|since|until|till|before|after|during|from|around|about|circa|ca\.|c\.|approximately|roughly|seit|bis|um|etwa|anno)\s+(\d{{1,4}})(?:\s*{ERA})?"
    ))
});
static BARE_YEAR: LazyLock<Regex> = LazyLock::new(|| ci(r"(1\d{3}|20\d{2})"));

/// A number followed by one of these is a quantity, time or reference, never a date.
const UNIT_WORDS: &[&str] = &[
    "people",
    "persons",
    "person",
    "years",
    "year",
    "yrs",
    "percent",
    "%",
    "dollars",
    "euros",
    "euro",
    "pounds",
    "mark",
    "marks",
    "timestamp",
    "times",
    "subscribers",
    "subs",
    "followers",
    "views",
    "likes",
    "comments",
    "meters",
    "metres",
    "feet",
    "foot",
    "miles",
    "km",
    "kilometers",
    "kilometres",
    "pages",
    "page",
    "words",
    "verses",
    "manuscripts",
    "copies",
    "men",
    "women",
    "soldiers",
    "troops",
    "members",
    "students",
    "o'clock",
    "minutes",
    "minute",
    "seconds",
    "second",
    "hours",
    "hour",
    "days",
    "day",
    "am",
    "pm",
    "a.m.",
    "p.m.",
    "point",
    "plus",
    "k",
    "million",
    "billion",
    "thousand",
    "hundred",
    "calories",
    "degrees",
    "menschen",
    "leute",
    "jahre",
    "jahren",
    "prozent",
    "seiten",
    "stunden",
    "minuten",
    "uhr",
];
/// A number preceded by one of these is a reference or amount, never a date.
const REFERENCE_WORDS: &[&str] = &[
    "number", "no.", "nr.", "page", "verse", "chapter", "room", "#", "$", "€", "£", "at", "over",
    "under", "past", "top", "psalm", "seite", "kapitel",
];
const HEDGES: &[&str] = &[
    "probably",
    "possibly",
    "perhaps",
    "maybe",
    "might",
    "may have",
    "supposedly",
    "allegedly",
    "reportedly",
    "apparently",
    "i think",
    "i believe",
    "some say",
    "traditionally",
    "wahrscheinlich",
    "vermutlich",
    "vielleicht",
    "angeblich",
    "möglicherweise",
];
const NEGATIONS: &[&str] = &["not", "never", "nicht", "nie", "kein", "keine"];
const APPROXIMATE: &[&str] = &[
    "around",
    "about",
    "circa",
    "ca.",
    "c.",
    "approximately",
    "roughly",
    "um",
    "etwa",
    "ungefähr",
    "gegen",
];
const ORDINAL_WORDS: &[&str] = &[
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
    "twenty-first",
];

fn month_number(name: &str) -> Option<u8> {
    let name = name.trim_end_matches('.').to_lowercase();
    Some(match name.as_str() {
        "january" | "januar" | "jan" => 1,
        "february" | "februar" | "feb" => 2,
        "march" | "märz" | "maerz" | "mar" => 3,
        "april" | "apr" => 4,
        "may" | "mai" => 5,
        "june" | "juni" | "jun" => 6,
        "july" | "juli" | "jul" => 7,
        "august" | "aug" => 8,
        "september" | "sept" | "sep" => 9,
        "october" | "oktober" | "oct" | "okt" => 10,
        "november" | "nov" => 11,
        "december" | "dezember" | "dec" | "dez" => 12,
        _ => return None,
    })
}

fn is_bc(era: &str) -> bool {
    let era = era.to_lowercase();
    era.starts_with('b') || era.starts_with('v')
}

fn ordinal(word: &str) -> Option<i32> {
    let digits: String = word.chars().take_while(char::is_ascii_digit).collect();
    if !digits.is_empty() {
        return digits.parse().ok();
    }
    let word = word.to_lowercase();
    ORDINAL_WORDS
        .iter()
        .position(|w| *w == word)
        .map(|i| i as i32 + 1)
}

/// Words before `start` (lower-case, nearest last).
fn words_before(text: &str, start: usize, n: usize) -> Vec<String> {
    let mut words: Vec<String> = text[..start]
        .split_whitespace()
        .rev()
        .take(n)
        .map(|w| {
            w.trim_matches(|c: char| c == ',' || c == '(' || c == '"')
                .to_lowercase()
        })
        .collect();
    words.reverse();
    words
}

fn next_word(text: &str, end: usize) -> String {
    let rest = text[end..].trim_start();
    if rest.starts_with('%') {
        return "%".into();
    }
    rest.split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches([',', '.', ';', ':', '!', '?', ')'])
        .to_lowercase()
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The match does not start or end inside a word or number.
fn on_boundaries(text: &str, start: usize, end: usize) -> bool {
    let before = text[..start].chars().next_back();
    let after = text[end..].chars().next();
    !before.is_some_and(|c| is_word_char(c) || c == '.' && start >= 2)
        && !after.is_some_and(|c| is_word_char(c) || c == ':')
        && !before.is_some_and(|c| c == '$' || c == '#' || c == ':')
}

fn contains_phrase(lower: &str, phrase: &str) -> bool {
    lower.match_indices(phrase).any(|(i, _)| {
        let before = lower[..i].chars().next_back();
        let after = lower[i + phrase.len()..].chars().next();
        !before.is_some_and(is_word_char) && !after.is_some_and(is_word_char)
    })
}

struct Candidate {
    start: usize,
    end: usize,
    kind: TemporalExpressionKind,
    date: HistoricalDate,
    confidence: f32,
}

fn era_year(year: i32, era: Option<&str>) -> i32 {
    match era {
        Some(era) if is_bc(era) => -year,
        _ => year,
    }
}

fn plausible(year: i32, era: bool) -> bool {
    year != 0 && (era || (100..=2100).contains(&year))
}

fn full_date(
    caps: &Captures,
    day: usize,
    month: usize,
    year: usize,
    era: usize,
) -> Option<HistoricalDate> {
    let day: u8 = caps[day].parse().ok()?;
    let month = month_number(&caps[month])?;
    let era = caps.get(era).map(|m| m.as_str());
    let year: i32 = caps[year].parse().ok()?;
    if !(1..=31).contains(&day) || !plausible(year, era.is_some()) {
        return None;
    }
    Some(HistoricalDate::point(
        era_year(year, era),
        Some(month),
        Some(day),
    ))
}

fn century_range(n: i32, bc: bool, part: Option<&str>) -> Option<HistoricalDate> {
    if !(1..=21).contains(&n) {
        return None;
    }
    let (from, to, approximate) = match part.map(str::to_lowercase).as_deref() {
        Some("early" | "frühen" | "fruehen") => (1, 33, true),
        Some("mid") => (34, 66, true),
        Some("late" | "späten" | "spaeten") => (67, 100, true),
        _ => (1, 100, false),
    };
    let base = (n - 1) * 100;
    let (start, end) = if bc {
        // The 5th century BC runs from 500 BC down to 401 BC.
        (-(base + 101 - from), -(base + 101 - to))
    } else {
        (base + from, base + to)
    };
    Some(HistoricalDate {
        approximate,
        ..HistoricalDate::years(start, end)
    })
}

fn candidates(text: &str) -> Vec<Candidate> {
    use TemporalExpressionKind as Kind;
    let mut found = Vec::new();
    let mut push = |m: Match, kind, date: Option<HistoricalDate>, confidence: f32| {
        if let Some(date) = date {
            found.push(Candidate {
                start: m.start(),
                end: m.end(),
                kind,
                date,
                confidence,
            });
        }
    };
    for caps in FULL_DAY_FIRST.captures_iter(text) {
        push(
            caps.get(0).unwrap(),
            Kind::FullDate,
            full_date(&caps, 1, 2, 3, 4),
            0.95,
        );
    }
    for caps in FULL_MONTH_FIRST.captures_iter(text) {
        push(
            caps.get(0).unwrap(),
            Kind::FullDate,
            full_date(&caps, 2, 1, 3, 4),
            0.95,
        );
    }
    for caps in MONTH_YEAR.captures_iter(text) {
        let era = caps.get(3).map(|m| m.as_str());
        let date = caps[2].parse::<i32>().ok().and_then(|year| {
            let month = month_number(&caps[1])?;
            // "May 12" is a day, not a year; months only take three- or four-digit years.
            (plausible(year, era.is_some()) && (era.is_some() || caps[2].len() >= 3))
                .then(|| HistoricalDate::point(era_year(year, era), Some(month), None))
        });
        push(caps.get(0).unwrap(), Kind::MonthYear, date, 0.9);
    }
    for caps in RANGE.captures_iter(text) {
        let keyword = caps.get(1).map(|m| m.as_str().to_lowercase());
        let separator = caps[4].to_lowercase();
        let (start_text, end_text) = (&caps[2], &caps[5]);
        let (start_era, end_era) = (
            caps.get(3).map(|m| m.as_str()),
            caps.get(6).map(|m| m.as_str()),
        );
        let with_era = start_era.is_some() || end_era.is_some();
        let joins = matches!(separator.as_str(), "and" | "und");
        if joins && !matches!(keyword.as_deref(), Some("between" | "zwischen")) {
            continue;
        }
        if keyword.is_none() && !with_era && (start_text.len() < 4 || separator == "until") {
            continue;
        }
        if !with_era && start_text.len() < 3 {
            continue;
        }
        let Ok(start) = start_text.parse::<i32>() else {
            continue;
        };
        let Ok(mut end) = end_text.parse::<i32>() else {
            continue;
        };
        let dash = matches!(separator.as_str(), "-" | "–" | "—");
        if !with_era && end_text.len() < start_text.len() && (!dash || end_text.len() > 2) {
            continue;
        }
        // "1914-18": an abbreviated end keeps the start's leading digits.
        if !with_era && end_text.len() < start_text.len() {
            let scale = 10_i32.pow(end_text.len() as u32);
            end += start - start % scale;
        }
        let start_era = start_era.or(end_era);
        let end_era = end_era.or(start_era);
        let (start, end) = (era_year(start, start_era), era_year(end, end_era));
        if !plausible(start.abs(), with_era) || !plausible(end.abs(), with_era) || end <= start {
            continue;
        }
        let confidence = if keyword.is_some() || with_era {
            0.9
        } else {
            0.8
        };
        push(
            caps.get(0).unwrap(),
            Kind::Range,
            Some(HistoricalDate::years(start, end)),
            confidence,
        );
    }
    for caps in CENTURY.captures_iter(text) {
        let bc = caps.get(4).is_some_and(|m| is_bc(m.as_str()));
        let date =
            ordinal(&caps[2]).and_then(|n| century_range(n, bc, caps.get(1).map(|m| m.as_str())));
        push(caps.get(0).unwrap(), Kind::Century, date, 0.85);
    }
    for caps in DECADE.captures_iter(text) {
        let decade: i32 = caps[2].parse().unwrap_or(0) * 10;
        let (from, to, approximate) =
            match caps.get(1).map(|m| m.as_str().to_lowercase()).as_deref() {
                Some("early") => (0, 3, true),
                Some("mid") => (3, 6, true),
                Some("late") => (6, 9, true),
                _ => (0, 9, false),
            };
        // "1000s of people" is a quantity.
        let quantity = next_word(text, caps.get(0).unwrap().end()) == "of";
        let date = ((1000..=2090).contains(&decade) && !quantity).then(|| HistoricalDate {
            approximate,
            ..HistoricalDate::years(decade + from, decade + to)
        });
        push(caps.get(0).unwrap(), Kind::Decade, date, 0.8);
    }
    for caps in ERA_YEAR.captures_iter(text) {
        let date = match (caps.get(1), caps.get(3)) {
            (Some(year), _) => year
                .as_str()
                .parse::<i32>()
                .ok()
                .filter(|y| *y > 0)
                .map(|y| HistoricalDate::point(era_year(y, Some(&caps[2])), None, None)),
            (None, Some(year)) => year
                .as_str()
                .parse::<i32>()
                .ok()
                .filter(|y| *y > 0)
                .map(|y| HistoricalDate::point(y, None, None)),
            _ => None,
        };
        push(caps.get(0).unwrap(), Kind::Year, date, 0.95);
    }
    for caps in CUE_YEAR.captures_iter(text) {
        let cue = caps[1].to_lowercase();
        let year_cue = cue.contains("year") || cue.starts_with("im jahr") || cue == "anno";
        let year: i32 = caps[2].parse().unwrap_or(0);
        let era = caps.get(3).map(|m| m.as_str());
        let ok = if era.is_some() {
            year > 0
        } else if year_cue {
            (100..=2100).contains(&year)
        } else {
            caps[2].len() == 4 && (1000..=2100).contains(&year)
        };
        // A plain cue ("in") only lends confidence; "the year" is part of the expression.
        let whole = caps.get(0).unwrap();
        if ok {
            found.push(Candidate {
                start: if year_cue {
                    whole.start()
                } else {
                    caps.get(2).unwrap().start()
                },
                end: whole.end(),
                kind: Kind::Year,
                date: HistoricalDate::point(era_year(year, era), None, None),
                confidence: if year_cue || era.is_some() { 0.9 } else { 0.8 },
            });
        }
    }
    for m in BARE_YEAR.find_iter(text) {
        let year: i32 = m.as_str().parse().unwrap_or(0);
        found.push(Candidate {
            start: m.start(),
            end: m.end(),
            kind: Kind::Year,
            date: HistoricalDate::point(year, None, None),
            confidence: 0.45,
        });
    }
    found
}

/// Finds and normalizes the temporal expressions in one text.
///
/// Overlapping readings resolve to the longest, then most confident one. Expressions below
/// [`TEMPORAL_MIN_CONFIDENCE`] are counted in
/// [`TemporalExpressions::low_confidence_dropped`] instead of returned.
pub fn normalize_temporal_expressions(text: &str) -> TemporalExpressions {
    let lower = text.to_lowercase();
    // "the 1148 timestamp", "in 1148": four-digit numbers next to video timestamps are times.
    let timestamps = contains_phrase(&lower, "timestamp") || contains_phrase(&lower, "time stamp");
    let mut all: Vec<Candidate> = candidates(text)
        .into_iter()
        .filter(|c| on_boundaries(text, c.start, c.end) && c.date.is_valid())
        .filter(|c| !(timestamps && c.date.end_year.is_none() && c.date.start_month.is_none()))
        .filter(|c| {
            let next = next_word(text, c.end);
            let before = words_before(text, c.start, 1);
            !UNIT_WORDS.contains(&next.as_str())
                && !before
                    .last()
                    .is_some_and(|w| REFERENCE_WORDS.contains(&w.as_str()))
        })
        .collect();
    all.sort_by(|a, b| {
        (b.end - b.start)
            .cmp(&(a.end - a.start))
            .then(b.confidence.total_cmp(&a.confidence))
            .then(a.start.cmp(&b.start))
    });
    let mut chosen: Vec<Candidate> = Vec::new();
    for candidate in all {
        if chosen
            .iter()
            .all(|c| candidate.end <= c.start || c.end <= candidate.start)
        {
            chosen.push(candidate);
        }
    }
    chosen.sort_by_key(|c| c.start);

    let certainty = if HEDGES.iter().any(|hedge| contains_phrase(&lower, hedge)) {
        TemporalCertainty::Hedged
    } else {
        TemporalCertainty::Stated
    };
    let mut result = TemporalExpressions::default();
    for mut candidate in chosen {
        if candidate.confidence < TEMPORAL_MIN_CONFIDENCE {
            result.low_confidence_dropped += 1;
            continue;
        }
        let before = words_before(text, candidate.start, 3);
        if before
            .last()
            .is_some_and(|w| APPROXIMATE.contains(&w.as_str()))
        {
            candidate.date.approximate = true;
        }
        let denies = before
            .iter()
            .any(|w| NEGATIONS.contains(&w.as_str()) || w.ends_with("n't") || w.ends_with("n’t"));
        result.expressions.push(TemporalExpression {
            // Regex matches always fall on character boundaries.
            span: TextSpan::from_byte_range(text, candidate.start, candidate.end)
                .expect("regex match is a valid byte range"),
            text: text[candidate.start..candidate.end].to_string(),
            kind: candidate.kind,
            date: candidate.date,
            confidence: candidate.confidence,
            certainty,
            polarity: if denies {
                TemporalPolarity::Denies
            } else {
                TemporalPolarity::Affirms
            },
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dates(text: &str) -> Vec<(String, HistoricalDate)> {
        normalize_temporal_expressions(text)
            .expressions
            .into_iter()
            .map(|m| (m.text, m.date))
            .collect()
    }

    fn year(y: i32) -> HistoricalDate {
        HistoricalDate::point(y, None, None)
    }

    fn range(a: i32, b: i32) -> HistoricalDate {
        HistoricalDate::years(a, b)
    }

    fn approx(date: HistoricalDate) -> HistoricalDate {
        HistoricalDate {
            approximate: true,
            ..date
        }
    }

    // Lines below are excerpts of real YouTube auto-captions (youtube-corpus evaluation corpus).
    #[test]
    fn real_caption_lines_yield_dates_and_reject_numbers() {
        assert_eq!(
            dates("In 2010, he did a missions trip to"),
            vec![("2010".into(), year(2010))]
        );
        assert_eq!(
            dates("miracles. Hume was an 18th century"),
            vec![("18th century".into(), range(1701, 1800))]
        );
        assert_eq!(
            dates("everybody's favorite 6th century"),
            vec![("6th century".into(), range(501, 600))]
        );
        assert_eq!(
            dates("as in someone writing i a first century"),
            vec![("first century".into(), range(1, 100))]
        );
        assert_eq!(
            dates("population in 2010 to 29% in 2020."),
            vec![("2010".into(), year(2010)), ("2020".into(), year(2020))]
        );
        assert_eq!(
            dates("on the timeline there's the year 2022 ad"),
            vec![("the year 2022 ad".into(), year(2022))]
        );
        // False dates: timestamps, counts and amounts.
        for line in [
            "to the 1148 timestamp if you actually",
            "the 1130 reaper did not kill fred",
            "went to the 1148 Mark of palegia's",
            "heads and you're one of 1000 people or",
            "celebration video for having past 1500",
            "and 11 o'clock at 1107 in 30 seconds",
            "been going on for 2000 years is",
            "At this rate, in a century, -21% of the",
            "divided by 1001 roughly 99.9 percent uh",
            "timestamp in 1148 if there's any shred",
            "see page 1066 for details",
        ] {
            assert_eq!(dates(line), vec![], "{line}");
        }
    }

    #[test]
    fn bare_years_are_low_confidence_and_dropped() {
        let found = normalize_temporal_expressions("of skeptical atheism Paul Draper 2015");
        assert!(found.expressions.is_empty());
        assert_eq!(found.low_confidence_dropped, 1);
    }

    #[test]
    fn days_months_eras_and_ranges() {
        let text = "The Battle of Hastings was fought on 14 October 1066.";
        let found = normalize_temporal_expressions(text);
        let mention = &found.expressions[0];
        assert_eq!(mention.kind, TemporalExpressionKind::FullDate);
        assert_eq!(
            &text[mention.span.byte_start..mention.span.byte_end],
            "14 October 1066"
        );
        assert_eq!(
            dates(text),
            vec![(
                "14 October 1066".into(),
                HistoricalDate::point(1066, Some(10), Some(14))
            )]
        );
        assert_eq!(
            dates("on March 15, 44 BC Caesar died"),
            vec![(
                "March 15, 44 BC".into(),
                HistoricalDate::point(-44, Some(3), Some(15))
            )]
        );
        assert_eq!(
            dates("by October 1066 the north had fallen"),
            vec![(
                "October 1066".into(),
                HistoricalDate::point(1066, Some(10), None)
            )]
        );
        assert_eq!(dates("in 586 BC"), vec![("586 BC".into(), year(-586))]);
        assert_eq!(
            dates("around AD 70"),
            vec![("AD 70".into(), approx(year(70)))]
        );
        assert_eq!(
            dates("the war lasted from 1914 to 1918"),
            vec![("from 1914 to 1918".into(), range(1914, 1918))]
        );
        assert_eq!(
            dates("in 1914-18"),
            vec![("1914-18".into(), range(1914, 1918))]
        );
        assert_eq!(
            dates("from 500 to 300 BC"),
            vec![("from 500 to 300 BC".into(), range(-500, -300))]
        );
        assert_eq!(
            dates("in the 1960s"),
            vec![("1960s".into(), range(1960, 1969))]
        );
        assert_eq!(
            dates("the 5th century BC"),
            vec![("5th century BC".into(), range(-500, -401))]
        );
        assert_eq!(
            dates("late 15th century"),
            vec![("late 15th century".into(), approx(range(1467, 1500)))]
        );
        assert_eq!(dates("between 2 and 3 people"), vec![]);
    }

    #[test]
    fn german_dates() {
        assert_eq!(
            dates("Die Schlacht fand am 14. Oktober 1066 statt"),
            vec![(
                "14. Oktober 1066".into(),
                HistoricalDate::point(1066, Some(10), Some(14))
            )]
        );
        assert_eq!(
            dates("im 4. Jahrhundert"),
            vec![("4. Jahrhundert".into(), range(301, 400))]
        );
        assert_eq!(
            dates("um 44 v. Chr."),
            vec![("44 v. Chr.".into(), approx(year(-44)))]
        );
        assert_eq!(
            dates("im Jahr 1517"),
            vec![("im Jahr 1517".into(), year(1517))]
        );
    }

    #[test]
    fn approximation_hedging_and_negation() {
        let found = normalize_temporal_expressions("it was probably written around 1450");
        let mention = &found.expressions[0];
        assert!(mention.date.approximate);
        assert_eq!(mention.certainty, TemporalCertainty::Hedged);
        let found = normalize_temporal_expressions("the council did not meet in 1054");
        assert_eq!(found.expressions[0].polarity, TemporalPolarity::Denies);
        assert_eq!(found.expressions[0].certainty, TemporalCertainty::Stated);
    }

    #[test]
    fn readings_are_valid_historical_dates() {
        assert!(!HistoricalDate::point(0, None, None).is_valid());
        assert!(!HistoricalDate::point(1066, None, Some(14)).is_valid());
        assert!(!HistoricalDate::years(1918, 1914).is_valid());
        assert!(HistoricalDate::years(-500, -401).is_valid());
    }
}
