from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEZONE = ZoneInfo("Asia/Shanghai")
REGIONS = ("us", "eu", "jp")
NEWS_LIMIT = 8
DRIVER_LIMIT = 3
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/news/search"

REGION_QUERIES = {
    "us": '("US consumer sentiment" OR "United States consumer confidence" OR "Michigan consumer sentiment") inflation jobs households',
    "eu": '("euro area consumer confidence" OR "EU consumer confidence" OR "European consumer confidence") inflation households employment',
    "jp": '("Japan consumer confidence" OR "Japanese consumer confidence") wages employment households inflation',
}

REGION_LABELS = {
    "us": "US",
    "eu": "EU27",
    "jp": "Japan",
}

POSITIVE_WORDS = {
    "beat", "boost", "cooling", "eased", "eases", "gain", "gains", "grew", "growth", "improved",
    "improves", "increase", "increased", "optimism", "rebound", "recover", "recovered", "rise", "rose",
    "strong", "stronger", "support", "upbeat", "wage growth",
}

NEGATIVE_WORDS = {
    "anxiety", "decline", "declined", "drop", "fell", "inflation", "layoff", "layoffs", "pessimism",
    "pressure", "recession", "risk", "risks", "soft", "softer", "stress", "uncertain", "uncertainty",
    "weak", "weaker", "worry",
}


@dataclass(frozen=True)
class NewsEvent:
    region: str
    title: str
    source: str
    url: str
    published_at: date
    sentiment: float
    relevance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch live information inputs for ConsumerSim site updates.")
    parser.add_argument("--as-of", help="Information cutoff in YYYY-MM-DD. Defaults to today in Asia/Shanghai.")
    parser.add_argument("--lookback-days", type=int, default=35)
    parser.add_argument("--news-output-root", type=Path, default=ROOT / "examples")
    parser.add_argument("--driver-output", type=Path, default=ROOT / "data" / "forecast_driver_events.csv")
    parser.add_argument("--bing-endpoint", default=os.getenv("BING_NEWS_ENDPOINT", BING_ENDPOINT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    as_of = date.fromisoformat(args.as_of) if args.as_of else datetime.now(DEFAULT_TIMEZONE).date()
    bing_key = (
        os.getenv("CONSUMERSIM_BING_NEWS_API_KEY")
        or os.getenv("BING_NEWS_API_KEY")
        or os.getenv("BING_API_KEY")
        or ""
    ).strip()
    all_events: dict[str, list[NewsEvent]] = {}
    for region in REGIONS:
        events = fetch_region_events(region, as_of, args.lookback_days, bing_key, args.bing_endpoint)
        if not events:
            events = read_existing_events(args.news_output_root / region / "news.jsonl", region)
        all_events[region] = events[:NEWS_LIMIT]
        write_news_jsonl(args.news_output_root / region / "news.jsonl", all_events[region])
        write_indicators_csv(args.news_output_root / region / "indicators.csv", all_events[region], as_of)

    write_driver_events(args.driver_output, all_events, as_of)
    summary = ", ".join(f"{region}={len(events)}" for region, events in all_events.items())
    print(f"Fetched information inputs as of {as_of.isoformat()}: {summary}")


def fetch_region_events(region: str, as_of: date, lookback_days: int, bing_key: str, bing_endpoint: str) -> list[NewsEvent]:
    if bing_key:
        try:
            events = fetch_bing_events(region, as_of, lookback_days, bing_key, bing_endpoint)
            if events:
                return events
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"warning: Bing fetch failed for {region}: {exc}", file=sys.stderr)
    try:
        return fetch_rss_events(region, as_of, lookback_days)
    except (urllib.error.URLError, ET.ParseError, TimeoutError, ValueError) as exc:
        print(f"warning: RSS fetch failed for {region}: {exc}", file=sys.stderr)
        return []


def fetch_bing_events(region: str, as_of: date, lookback_days: int, api_key: str, endpoint: str) -> list[NewsEvent]:
    query = REGION_QUERIES[region]
    since = as_of - timedelta(days=lookback_days)
    params = {
        "q": query,
        "count": str(NEWS_LIMIT),
        "sortBy": "Date",
        "freshness": "Month",
        "textFormat": "Raw",
        "mkt": "en-US",
    }
    request = urllib.request.Request(
        f"{endpoint}?{urllib.parse.urlencode(params)}",
        headers={"Ocp-Apim-Subscription-Key": api_key, "User-Agent": "ConsumerSimBot/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    events: list[NewsEvent] = []
    for item in payload.get("value", []):
        published = parse_date(str(item.get("datePublished") or "")) or as_of
        if published < since or published > as_of:
            continue
        source = ((item.get("provider") or [{}])[0].get("name") or "Bing News").strip()
        title = clean_title(str(item.get("name") or ""))
        if title and region_matches(title, source, region):
            events.append(make_event(region, title, source, str(item.get("url") or ""), published))
    return unique_events(events)


def fetch_rss_events(region: str, as_of: date, lookback_days: int) -> list[NewsEvent]:
    query = REGION_QUERIES[region]
    since = as_of - timedelta(days=lookback_days)
    params = urllib.parse.urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    request = urllib.request.Request(
        f"https://news.google.com/rss/search?{params}",
        headers={"User-Agent": "ConsumerSimBot/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        root = ET.fromstring(response.read())
    events: list[NewsEvent] = []
    for item in root.findall("./channel/item"):
        title = clean_title(item.findtext("title") or "")
        link = item.findtext("link") or ""
        source = item.findtext("source") or source_from_title(title) or "Google News"
        published = parse_date(item.findtext("pubDate") or "") or as_of
        if published < since or published > as_of:
            continue
        if title and region_matches(title, source, region):
            events.append(make_event(region, title, source, link, published))
    return unique_events(events)[:NEWS_LIMIT]


def make_event(region: str, title: str, source: str, url: str, published: date) -> NewsEvent:
    score = sentiment_score(title)
    relevance = relevance_score(title, region)
    return NewsEvent(
        region=region,
        title=title,
        source=source or "News",
        url=url,
        published_at=published,
        sentiment=score,
        relevance=relevance,
    )


def read_existing_events(path: Path, region: str) -> list[NewsEvent]:
    if not path.exists():
        return []
    events: list[NewsEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            published = date.fromisoformat(str(row.get("published_at", ""))[:10])
            events.append(
                NewsEvent(
                    region=region,
                    title=str(row.get("title", "")),
                    source=str(row.get("source", "News")),
                    url=str(row.get("url", "")),
                    published_at=published,
                    sentiment=float(row.get("sentiment", 0.0)),
                    relevance=float(row.get("relevance", 1.0)),
                )
            )
    return sorted(
        [event for event in events if region_matches(event.title, event.source, region)],
        key=lambda event: event.published_at,
        reverse=True,
    )


def write_news_jsonl(path: Path, events: list[NewsEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in sorted(events, key=lambda item: item.published_at):
            handle.write(json.dumps({
                "published_at": event.published_at.isoformat(),
                "title": event.title,
                "source": event.source,
                "url": event.url,
                "sentiment": round(event.sentiment, 4),
                "relevance": round(event.relevance, 4),
            }, ensure_ascii=False) + "\n")


def write_indicators_csv(path: Path, events: list[NewsEvent], as_of: date) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recent = [event for event in events if event.published_at >= as_of - timedelta(days=14)]
    news_balance = weighted_mean([(event.sentiment, event.relevance) for event in recent])
    pressure = weighted_mean([(min(0.0, event.sentiment), event.relevance) for event in recent])
    coverage = min(3.0, max(-3.0, (len(recent) - 4) / 2.0))
    rows = [
        {"observed_at": as_of.isoformat(), "name": "news_sentiment_balance", "z_score": f"{news_balance:.4f}", "weight": "1.00"},
        {"observed_at": as_of.isoformat(), "name": "news_coverage_intensity", "z_score": f"{coverage:.4f}", "weight": "0.65"},
        {"observed_at": as_of.isoformat(), "name": "negative_news_pressure", "z_score": f"{pressure:.4f}", "weight": "0.80"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["observed_at", "name", "z_score", "weight"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_driver_events(path: Path, events_by_region: dict[str, list[NewsEvent]], as_of: date) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["region", "cadence", "sort_order", "headline", "source", "event_period", "tag", "summary", "url"]
    rows: list[dict[str, str]] = []
    for region in REGIONS:
        events = events_by_region.get(region, [])
        weekly = sorted(events, key=lambda event: event.published_at, reverse=True)[:DRIVER_LIMIT]
        monthly = sorted(events, key=lambda event: (abs(event.sentiment) * event.relevance, event.published_at), reverse=True)[:DRIVER_LIMIT]
        for cadence, selected in (("weekly", weekly), ("monthly", monthly)):
            for index, event in enumerate(selected, 1):
                rows.append({
                    "region": region,
                    "cadence": cadence,
                    "sort_order": str(index),
                    "headline": event.title,
                    "source": event.source,
                    "event_period": week_label(event.published_at) if cadence == "weekly" else month_label(event.published_at),
                    "tag": event_tag(event),
                    "summary": driver_summary(event, REGION_LABELS[region]),
                    "url": event.url,
                })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sentiment_score(text: str) -> float:
    normalized = text.lower()
    positive = sum(1 for word in POSITIVE_WORDS if word in normalized)
    negative = sum(1 for word in NEGATIVE_WORDS if word in normalized)
    score = (positive - negative) / max(2, positive + negative + 1)
    return max(-1.0, min(1.0, score))


def relevance_score(text: str, region: str) -> float:
    normalized = text.lower()
    terms = ["consumer", "confidence", "sentiment", "inflation", "wage", "jobs", "employment", "household"]
    region_terms = {
        "us": ["us", "u.s.", "united states", "american"],
        "eu": ["eu", "euro", "europe", "eurozone"],
        "jp": ["japan", "japanese"],
    }[region]
    matches = sum(1 for term in terms + region_terms if term in normalized)
    return min(1.5, 0.6 + matches * 0.12)


def region_matches(title: str, source: str, region: str) -> bool:
    normalized = f"{title} {source}".lower()
    region_terms = {
        "us": ["u.s.", "us", "united states", "american", "michigan", "conference board"],
        "eu": [
            "eu", "e.u.", "euro", "europe", "eurozone", "euro area", "germany", "france", "italy",
            "spain", "netherlands", "belgium", "austria", "portugal",
        ],
        "jp": ["japan", "japanese", "yen", "tokyo"],
    }[region]
    excluded_terms = {
        "us": ["india", "euro", "europe", "japan", "japanese"],
        "eu": ["united states", "u.s.", "us", "india", "japan", "japanese"],
        "jp": ["united states", "u.s.", "us", "euro", "europe", "india"],
    }[region]
    return any(contains_term(normalized, term) for term in region_terms) and not any(
        contains_term(normalized, term) for term in excluded_terms
    )


def contains_term(text: str, term: str) -> bool:
    if any(character in term for character in ". "):
        return term in text
    return bool(re.search(rf"\b{re.escape(term)}\b", text))


def event_tag(event: NewsEvent) -> str:
    title = event.title.lower()
    if "inflation" in title or "price" in title or "cost" in title:
        return "Inflation"
    if "job" in title or "employment" in title or "wage" in title:
        return "Labor"
    if "confidence" in title or "sentiment" in title:
        return "Confidence"
    if event.sentiment < -0.15:
        return "Risk"
    if event.sentiment > 0.15:
        return "Support"
    return "News"


def driver_summary(event: NewsEvent, label: str) -> str:
    direction = "positive" if event.sentiment > 0.05 else "negative" if event.sentiment < -0.05 else "neutral"
    return (
        f"Fetched {event.published_at.isoformat()} for {label}; "
        f"headline signal is {direction} ({event.sentiment:+.2f}) with relevance {event.relevance:.2f}."
    )


def unique_events(events: list[NewsEvent]) -> list[NewsEvent]:
    seen: set[str] = set()
    output: list[NewsEvent] = []
    for event in sorted(events, key=lambda item: item.published_at, reverse=True):
        key = re.sub(r"\W+", " ", event.title.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        output.append(event)
    return output


def clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r" - [^-]+$", "", title).strip()
    return title


def source_from_title(title: str) -> str:
    parts = title.rsplit(" - ", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError, IndexError):
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def weighted_mean(values: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in values if weight > 0)
    if total_weight == 0:
        return 0.0
    return sum(value * weight for value, weight in values if weight > 0) / total_weight


def week_label(day: date) -> str:
    week_number = ((day.day - 1) // 7) + 1
    return f"{day.strftime('%b')} W{week_number}"


def month_label(day: date) -> str:
    return f"{day.strftime('%b')}-{str(day.year)[2:]}"


if __name__ == "__main__":
    main()
