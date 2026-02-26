"""Leiden community detection for thematic song clustering.

Ported from GitNexus's community-processor.ts pattern.
Groups songs into thematic clusters based on shared themes, phrases,
and metaphor domains.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

from src.utils import PROCESSED_DIR


@dataclass
class ThematicClusterData:
    id: str
    label: str
    heuristic_label: str
    description: str
    enriched_by: str  # "heuristic" | "llm"
    cohesion: float
    song_count: int
    artist_id: str
    song_ids: list[str]
    keywords: list[str]


def run_thematic_clustering(
    artist_slug: str,
    graph_data: dict | None = None,
    advanced_data: dict | None = None,
) -> list[ThematicClusterData]:
    """Run Leiden community detection on songs to find thematic clusters.

    Uses igraph + leidenalg if available, falls back to simple
    theme-co-occurrence clustering if not installed.

    Args:
        artist_slug: Artist identifier.
        graph_data: Structural analysis data.
        advanced_data: Advanced analysis data (themes, metaphors).

    Returns:
        List of ThematicClusterData.
    """
    if graph_data is None:
        data_path = PROCESSED_DIR / f"{artist_slug}_graph_data.json"
        with open(data_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

    if advanced_data is None:
        adv_path = PROCESSED_DIR / f"{artist_slug}_advanced_analysis.json"
        with open(adv_path, "r", encoding="utf-8") as f:
            advanced_data = json.load(f)

    songs = graph_data.get("songs", [])
    themes = advanced_data.get("themes", [])

    if len(songs) < 3:
        return []

    # Build song-theme mapping
    song_themes = _build_song_theme_map(songs, themes)

    # Try Leiden algorithm
    try:
        clusters = _leiden_clustering(songs, song_themes, artist_slug)
    except ImportError:
        print("  leidenalg/igraph not available, using fallback clustering")
        clusters = _fallback_clustering(songs, song_themes, artist_slug)

    # Generate heuristic labels
    for cluster in clusters:
        cluster.heuristic_label = _generate_heuristic_label(cluster, song_themes, songs)
        cluster.label = cluster.heuristic_label
        cluster.enriched_by = "heuristic"

    print(f"  Found {len(clusters)} thematic clusters")
    for c in clusters:
        print(f"    {c.label}: {c.song_count} songs (cohesion: {c.cohesion:.2f})")

    # Save
    output_path = PROCESSED_DIR / f"{artist_slug}_clusters.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in clusters], f, ensure_ascii=False, indent=2)

    return clusters


def _build_song_theme_map(songs: list[dict], themes: list[dict]) -> dict[str, list[str]]:
    """Map each song ID to its theme names."""
    from src.analysis.fingerprint import THEME_KEYWORDS

    song_themes: dict[str, list[str]] = {}
    for song in songs:
        text_lower = song.get("full_lyrics_clean", "").lower()
        matched = []
        for theme_name, keywords in THEME_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score >= 2:
                matched.append(theme_name)
        song_themes[song["id"]] = matched

    return song_themes


def _leiden_clustering(
    songs: list[dict],
    song_themes: dict[str, list[str]],
    artist_slug: str,
) -> list[ThematicClusterData]:
    """Leiden algorithm clustering using igraph + leidenalg."""
    import igraph as ig
    import leidenalg

    # Build graph: nodes = songs, edges = shared themes
    song_ids = [s["id"] for s in songs]
    id_to_idx = {sid: i for i, sid in enumerate(song_ids)}

    g = ig.Graph()
    g.add_vertices(len(song_ids))

    edges = []
    weights = []
    for i in range(len(song_ids)):
        themes_i = set(song_themes.get(song_ids[i], []))
        for j in range(i + 1, len(song_ids)):
            themes_j = set(song_themes.get(song_ids[j], []))
            shared = themes_i & themes_j
            if shared:
                edges.append((i, j))
                weights.append(len(shared))

    if not edges:
        return _fallback_clustering(songs, song_themes, artist_slug)

    g.add_edges(edges)
    g.es["weight"] = weights

    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        weights=weights,
    )

    # Build clusters
    clusters = []
    for cluster_idx, members in enumerate(partition):
        if len(members) < 2:
            continue  # Skip singleton clusters

        member_ids = [song_ids[m] for m in members]

        # Compute cohesion: internal edges / possible internal edges
        internal_edges = sum(
            1 for e in g.es
            if e.source in members and e.target in members
        )
        possible = len(members) * (len(members) - 1) / 2
        cohesion = internal_edges / max(possible, 1)

        clusters.append(ThematicClusterData(
            id=f"{artist_slug}:cluster:{cluster_idx}",
            label="",
            heuristic_label="",
            description="",
            enriched_by="heuristic",
            cohesion=cohesion,
            song_count=len(members),
            artist_id=artist_slug,
            song_ids=member_ids,
            keywords=[],
        ))

    return clusters


def _fallback_clustering(
    songs: list[dict],
    song_themes: dict[str, list[str]],
    artist_slug: str,
) -> list[ThematicClusterData]:
    """Simple fallback clustering when Leiden is not available.

    Groups songs by their dominant theme.
    """
    theme_groups: defaultdict[str, list[str]] = defaultdict(list)

    for song in songs:
        themes = song_themes.get(song["id"], [])
        if themes:
            # Assign to most specific (least common) theme
            dominant = themes[0]
            theme_groups[dominant].append(song["id"])
        else:
            theme_groups["_uncategorized"].append(song["id"])

    clusters = []
    for idx, (theme, member_ids) in enumerate(theme_groups.items()):
        if len(member_ids) < 2:
            continue
        clusters.append(ThematicClusterData(
            id=f"{artist_slug}:cluster:{idx}",
            label="",
            heuristic_label="",
            description="",
            enriched_by="heuristic",
            cohesion=0.5,
            song_count=len(member_ids),
            artist_id=artist_slug,
            song_ids=member_ids,
            keywords=[],
        ))

    return clusters


def _generate_heuristic_label(
    cluster: ThematicClusterData,
    song_themes: dict[str, list[str]],
    songs: list[dict],
) -> str:
    """Generate a human-readable label for a cluster from its themes."""
    theme_counter: Counter = Counter()
    for sid in cluster.song_ids:
        for theme in song_themes.get(sid, []):
            theme_counter[theme] += 1

    if not theme_counter:
        return "Miscellaneous"

    # Top 2 themes become the label
    top = theme_counter.most_common(2)
    label_parts = [t[0].replace("_", " ").title() for t in top]
    label = " & ".join(label_parts)

    # Also set keywords from the top themes
    from src.analysis.fingerprint import THEME_KEYWORDS
    keywords = []
    for theme_name, _ in top:
        kws = THEME_KEYWORDS.get(theme_name, [])[:5]
        keywords.extend(kws)
    cluster.keywords = keywords[:10]
    cluster.description = f"Songs centered on {label.lower()}"

    return label
