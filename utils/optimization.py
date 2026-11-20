from pathlib import Path
import pandas as pd


def _find_col(df, aliases, required_name):
    for name in aliases:
        if name in df.columns:
            return name
    raise ValueError(
        f"Could not find a column for {required_name}. "
        f"Available columns: {list(df.columns)}"
    )


def prepare_overlap_recommendations(overlap_df, route_features):
    """Apply the overlap thresholds and route-characteristic joins from the notebook."""
    df = overlap_df.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
    )
    aliases = {
        "route_id_1": ["route_id_1", "route1", "route_a"],
        "route_id_2": ["route_id_2", "route2", "route_b"],
        "shared_stops": ["shared_stops", "common_stops"],
        "overlap_percentage": ["overlap_percentage", "overlap_percent", "overlap_pct", "jaccard_similarity"],
    }
    rename = {}
    for standard, options in aliases.items():
        actual = _find_col(df, options, standard)
        if actual != standard:
            rename[actual] = standard
    df = df.rename(columns=rename)

    for col in ["shared_stops", "overlap_percentage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["route_id_1", "route_id_2", "shared_stops", "overlap_percentage"]).copy()
    if not df.empty and df["overlap_percentage"].max() <= 1:
        df["overlap_percentage"] *= 100

    def category(value):
        if value >= 70:
            return "High"
        if value >= 40:
            return "Moderate"
        return "Low"

    def recommendation(value):
        if value >= 70:
            return "Investigate Possible Route Merging"
        if value >= 40:
            return "Review Shared Route Corridor"
        return "No Merging Recommendation"

    def reason(row):
        value = row["overlap_percentage"]
        if value >= 70:
            return f"High overlap of {value:.2f}% with {int(row['shared_stops'])} shared stops."
        if value >= 40:
            return f"Moderate overlap of {value:.2f}%. Review the shared corridor."
        return "Insufficient overlap for a merging recommendation."

    df["overlap_category"] = df["overlap_percentage"].apply(category)
    df["recommendation"] = df["overlap_percentage"].apply(recommendation)
    df["reason"] = df.apply(reason, axis=1)

    info = route_features[
        ["route_id", "avg_route_length_km", "total_trips", "total_unique_stops"]
    ].drop_duplicates("route_id").copy()
    info1 = info.rename(columns={
        "route_id":"route_id_1", "avg_route_length_km":"route_1_length_km",
        "total_trips":"route_1_total_trips", "total_unique_stops":"route_1_unique_stops"
    })
    info2 = info.rename(columns={
        "route_id":"route_id_2", "avg_route_length_km":"route_2_length_km",
        "total_trips":"route_2_total_trips", "total_unique_stops":"route_2_unique_stops"
    })
    df["route_id_1"] = df["route_id_1"].astype(str).str.strip()
    df["route_id_2"] = df["route_id_2"].astype(str).str.strip()
    info1["route_id_1"] = info1["route_id_1"].astype(str).str.strip()
    info2["route_id_2"] = info2["route_id_2"].astype(str).str.strip()
    return df.merge(info1,on="route_id_1",how="left").merge(info2,on="route_id_2",how="left")


def prepare_split_candidates(route_features):
    """Use the notebook's 90th-percentile length and median-stop screening rules."""
    required = ["route_id", "avg_route_length_km", "total_unique_stops", "total_trips"]
    missing = [c for c in required if c not in route_features.columns]
    if missing:
        raise ValueError(f"Missing route feature columns: {missing}")
    df = route_features[required].copy()
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).drop_duplicates("route_id").copy()
    if df.empty:
        return df, 0.0, 0.0
    length_threshold = float(df["avg_route_length_km"].quantile(.90))
    stop_threshold = float(df["total_unique_stops"].median())
    long_routes = df[df["avg_route_length_km"] >= length_threshold].copy()
    long_routes["high_stop_count"] = long_routes["total_unique_stops"] >= stop_threshold
    candidates = long_routes[long_routes["high_stop_count"]].copy()
    candidates = candidates.sort_values(
        ["avg_route_length_km", "total_unique_stops"], ascending=[False, False]
    )
    return candidates, length_threshold, stop_threshold


def build_split_points(split_candidates, trips_path, stop_times_path, chunksize=500000):
    """
    Reproduce the notebook's representative-trip and midpoint logic.
    Requires original GTFS trips.txt and stop_times.txt.
    """
    if split_candidates.empty:
        return pd.DataFrame()
    trips = pd.read_csv(trips_path, usecols=["trip_id", "route_id"], dtype=str)
    candidate_ids = set(split_candidates["route_id"].astype(str).str.strip())
    trips["route_id"] = trips["route_id"].astype(str).str.strip()
    candidate_trips = trips[trips["route_id"].isin(candidate_ids)].copy()
    if candidate_trips.empty:
        return pd.DataFrame()

    candidate_trip_ids = set(candidate_trips["trip_id"].astype(str))
    parts = []
    for chunk in pd.read_csv(
        stop_times_path, usecols=["trip_id", "stop_id", "stop_sequence"],
        chunksize=chunksize, dtype={"trip_id":str,"stop_id":str}
    ):
        chunk = chunk[chunk["trip_id"].isin(candidate_trip_ids)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame()

    stop_times = pd.concat(parts, ignore_index=True)
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")
    stop_times = stop_times.dropna(subset=["stop_sequence"])
    stop_times = stop_times.merge(candidate_trips, on="trip_id", how="left")
    stop_times = stop_times.sort_values(["route_id","trip_id","stop_sequence"])

    counts = stop_times.groupby(["route_id","trip_id"]).size().reset_index(name="stop_count")
    representative = counts.sort_values(
        ["route_id","stop_count","trip_id"], ascending=[True,False,True]
    ).drop_duplicates("route_id")

    records = []
    for _, row in representative.iterrows():
        trip = stop_times[stop_times["trip_id"] == row["trip_id"]].sort_values("stop_sequence")
        trip = trip.drop_duplicates(subset="stop_sequence").reset_index(drop=True)
        total = len(trip)
        if total < 4:
            continue
        midpoint = total // 2
        split_stop = trip.iloc[midpoint]
        records.append({
            "route_id": row["route_id"],
            "representative_trip_id": row["trip_id"],
            "total_stops": total,
            "first_segment_stops": midpoint,
            "second_segment_stops": total-midpoint,
            "candidate_split_stop_id": split_stop["stop_id"],
            "candidate_split_sequence": split_stop["stop_sequence"],
            "recommendation": "Investigate Possible Route Splitting",
            "reason": (
                f"Route has {total} stops in its representative trip. "
                f"Candidate midpoint is stop {split_stop['stop_id']}."
            )
        })
    return pd.DataFrame(records)
