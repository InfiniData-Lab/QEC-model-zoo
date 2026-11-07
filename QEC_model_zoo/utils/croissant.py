"""
Build Croissant (MLCommons metadata format, v1.1) JSON-LD descriptors for the
datasets listed in datasets/dataset_catalog.yaml.

Design notes:
- The catalog is the single source of truth. This module never hand-encodes
  per-dataset facts; it reads whatever the catalog entry provides and degrades
  gracefully when fields are missing (e.g. dataset 4, which is generated
  on-the-fly by Stim rather than backed by static files).
- When the referenced train/test parquet files are present on disk, the real
  column schema is introspected via pyarrow and used for the RecordSet
  fields, plus a sha256 checksum on the FileObject. When the files are not
  present (the common case here, since datasets/ is gitignored and populated
  manually), a best-effort RecordSet is built from the catalog's target_col /
  shot_column / n_rounds_column / feature_sets keys instead. Re-running the
  generator after data has been added automatically upgrades the output to
  the fully introspected version.
- Optional catalog keys beyond the original schema: `license` / `license_note`
  (see _LICENSE_PENDING_NOTE), `citation` (-> cr:citeAs), `url` (-> sc:url),
  `notes` (list of extra description sentences), and `live_note` (overrides
  the default explanation for entries with no folder_path, e.g. a live/
  procedurally-generated dataset that this repo does not itself generate).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CROISSANT_VERSION = "http://mlcommons.org/croissant/1.1"

CROISSANT_CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "sc": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "dct": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "annotation": "cr:annotation",
    "arrayShape": "cr:arrayShape",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "excludes": "cr:excludes",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isArray": "cr:isArray",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "readLines": "cr:readLines",
    "sdVersion": "cr:sdVersion",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "unArchive": "cr:unArchive",
    "value": "cr:value",
}

# Human-readable attribution for known catalog `source` values. Deliberately
# does not guess URLs or license terms that aren't in the catalog.
_SOURCE_NAMES = {
    "dicarlo": "DiCarlo Lab",
    "google": "Google Quantum AI",
    "vandersypen": "Vandersypen Lab",
    "stim": "Stim circuit simulation",
    "nvidia_ising_decoding": "NVIDIA (ising-decoding project)",
}

_PARQUET_ENCODING_FORMAT = "application/vnd.apache.parquet"

_LICENSE_PENDING_NOTE = (
    "License/usage terms for this dataset are not yet finalized (as of the catalog "
    "entry this file was generated from). Do not redistribute externally until confirmed "
    "with the data owner; set `license:` (and optionally `license_note:`) in "
    "dataset_catalog.yaml once terms are known."
)

def _slugify(name: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "_" for c in name)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrow_dtype_to_schema(arrow_type: Any) -> str:
    import pyarrow as pa

    if pa.types.is_boolean(arrow_type):
        return "sc:Boolean"
    if pa.types.is_integer(arrow_type):
        return "sc:Integer"
    if pa.types.is_floating(arrow_type):
        return "sc:Float"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "sc:Text"
    return "sc:Text"


def _introspect_parquet_fields(path: Path, record_set_id: str, file_object_id: str) -> list[dict] | None:
    """Return one cr:Field per real parquet column, or None if introspection isn't possible."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None

    try:
        schema = pq.ParquetFile(path).schema_arrow
    except Exception:
        return None

    fields = []
    for arrow_field in schema:
        fields.append({
            "@type": "cr:Field",
            "@id": f"{record_set_id}/{arrow_field.name}",
            "dataType": _arrow_dtype_to_schema(arrow_field.type),
            "source": {
                "fileObject": {"@id": file_object_id},
                "extract": {"column": arrow_field.name},
            },
        })
    return fields


def _fallback_fields(entry: dict, record_set_id: str, file_object_id: str) -> list[dict]:
    """Best-effort field list from catalog keys, used when the parquet file isn't on disk yet."""
    fields = []

    def add(col_key: str, data_type: str, description: str) -> None:
        col_name = entry.get(col_key)
        if not col_name:
            return
        fields.append({
            "@type": "cr:Field",
            "@id": f"{record_set_id}/{col_name}",
            "description": description,
            "dataType": data_type,
            "source": {
                "fileObject": {"@id": file_object_id},
                "extract": {"column": col_name},
            },
        })

    add("target_col", "sc:Boolean", "Binary logical-error indicator for this shot (1 = logical failure).")
    add("shot_column", "sc:Integer", "Per-shot identifier.")
    add("n_rounds_column", "sc:Integer", "Number of QEC rounds in this shot.")

    for feature_set in entry.get("feature_sets", []):
        fields.append({
            "@type": "cr:Field",
            "@id": f"{record_set_id}/features_{feature_set}",
            "description": (
                f"Placeholder for the '{feature_set}' feature group. "
                "Re-run generate_croissant.py once the parquet file is present on disk "
                "to auto-populate the exact columns via schema introspection."
            ),
            "dataType": "sc:Float",
        })

    return fields


def _file_object_and_fields(
    dataset_dir: Path,
    repo_root: Path,
    entry: dict,
    split: str,
    dataset_slug: str,
) -> tuple[dict, dict]:
    """Build the (FileObject, RecordSet) pair for one split (train/test)."""
    file_object_id = f"{dataset_slug}/{split}.parquet"
    record_set_id = f"{dataset_slug}/records_{split}"
    path = dataset_dir / f"{split}.parquet"
    rel_content_url = str((Path(entry["folder_path"]) / f"{split}.parquet"))

    file_object: dict[str, Any] = {
        "@type": "cr:FileObject",
        "@id": file_object_id,
        "contentUrl": rel_content_url,
        "encodingFormat": _PARQUET_ENCODING_FORMAT,
    }

    fields: list[dict] | None = None
    if path.is_file():
        file_object["sha256"] = _sha256(path)
        fields = _introspect_parquet_fields(path, record_set_id, file_object_id)
    else:
        file_object["description"] = (
            "Not present locally at generation time (data/ is populated manually and "
            "gitignored). contentUrl is a repo-relative path placeholder; replace with a "
            "public download location before publishing this dataset externally."
        )

    if fields is None:
        fields = _fallback_fields(entry, record_set_id, file_object_id)

    record_set = {
        "@type": "cr:RecordSet",
        "@id": record_set_id,
        "field": fields,
    }
    return file_object, record_set

def _add_provenance_rai_fields(metadata: dict, entry: dict) -> dict:
    source_dataset = entry.get("source_dataset")
    provenance = {
      "@id": source_dataset.get("url"),
      "prov:label": source_dataset.get("name"),
      "sc:license": source_dataset.get("license"),
      "prov:wasAttributedTo": {
        "prov:label": source_dataset.get("publisher"),
      },
    }
    metadata = {
        **metadata,
        "prov:wasDerivedFrom": [provenance],
    }

    return metadata

def _add_activities_and_agents(metadata: dict, entry: dict) -> dict:
    activities = entry.get("activities")
    activity_types_to_urls = {
        'quality review': 'https://www.wikidata.org/wiki/Q3306762',
        'data preprocessing': 'https://www.wikidata.org/wiki/Q5227332',
    }
    agent_type_mapping = {
        'agent': 'prov:Agent',
        'tool': 'prov:SoftwareAgent'
    }
    generated_by = []

    # Go over all activities, set fields and agents.
    for activity in activities:
        activity_type = activity.get("type").lower()

        if activity_type not in activity_types_to_urls:
            raise ValueError(f"Unknown activity type: {activity_type}, supported types are: {activity_types_to_urls.keys()}")
        activity_type_url = activity_types_to_urls[activity_type]
        agents_and_tools = activity.get("agents_and_tools")
        prov_agents = []
        for agent in agents_and_tools:
            agent_name = agent["name"]
            agent_id = agent_name.lower().replace(" ", "_")
            agent_type = agent["type"].lower()
            if agent_type not in agent_type_mapping:
                raise ValueError(f"Unknown agent type: {agent_type}, supported types are: {agent_type_mapping.keys()}")
            agent_type = agent_type_mapping[agent_type]
            description = agent["description"]
            prov_agents.append({
                "@type": agent_type,
                "@id": agent_id,
                "prov:label": agent_name,
                "sc:description": description,
            })
        generated_by.append({
            "@type": "prov:Activity",
            "prov:type": {
                "@id": activity_type_url,
            },
            "prov:label": activity.get("name"),
            "sc:description": activity.get("description"),
            "prov:wasAttributedTo": prov_agents,
        })

    metadata = {
        **metadata,
        "prov:wasGeneratedBy": generated_by,
    }

    return metadata

def _add_rai_fields(metadata: dict, entry: dict) -> dict:
    RAI_FIELDS = {
        'rai:dataLimitations': "The QEC model zoo dataset is intended solely for offline evaluation and is not designed for deployment in a real-time decoding environment. The fidelity per processed dataset is inherently restricted to the adopted shot-level feature construction scheme, which may omit or simplify information relevant to the full structure of the observed syndromes and underlying error patterns of the raw datasets. Although the Temporal and Spatial feature groups capture several features at each dimension across different datasets, the dataset is limited in capturing dynamic temporal-spatial information in a real-time, continuous decoding scenario. Consequently, the evaluation results on datasets should be interpreted within the scope of these feature-construction assumptions and should not be taken as a direct indication of real-time decoding performance.",
        'rai:dataBiases': "The real datasets are entirely sourced from specific quantum computing platforms or quantum processing units, mainly introducing platform- or hardware-specific bias, limiting the generalizability to other platforms. Additionally, moving to shot-level features adds new biases on top of the platform biases. Each shot is treated as an individual decoding sample characterized by a dataset-specific selection of engineered feature groups. Choosing these feature groups is subjective. While this representation is suited to shot-level AI-based decoding and enables evaluation of a broader range of AI models, it may not capture the full spectrum of syndrome measurement, potentially limiting the ability of AI-based decoders. For simulated datasets, biases are mainly from Stim's simulation models, noise models, circuit choices, and quantum data generation protocols with different code families for training, testing, or inference.",
        'rai:personalSensitiveInformation': "Although some dataset names contain personal names and organization names to facilitate differentiation between them, the datasets do not contain information related to gender, age, geographic location, language, culture, socio-economic status, health and medical data, and political or religious beliefs.",
        'rai:dataUseCases': "The shot-level sampling datasets are used to evaluate the performance of traditional machine learning and tabular deep learning models, including MLP, TabM, TabR, XGBoost, LightGBM, CatBoost, TabICLv2, and TabPFNv2 for QEC decoding tasks. These datasets can be used to evaluate more models that can consume them. In addition, these datasets support ablation studies to investigate the impact of different feature selections on decoding performance. By splitting the datasets into sliding windows, they could be further used for continuous decoding tasks, which would be better suited to real-time experiments.",
        'rai:dataSocialImpact': "The QEC model-zoo dataset enables rigorous empirical study of AI-based QEC decoders and supports feature-group ablation, allowing users to systematically investigate which input feature groups contribute most to the performance of different AI-based QEC decoders across code families. For QEC researchers, it provides a practical way to organize heterogeneous benchmark experiments, compare decoders under shared protocols, and reuse model configurations across datasets and hardware settings. For AI researchers, it lowers the barrier to entry by turning quantum decoding into a set of accessible benchmark tasks with shared inputs, metrics, and baselines. Potential negative impacts include the energy and resource costs of model training and benchmarking. Reusing released models and configurations can help reduce redundant training.",
        'rai:hasSyntheticData': False,
    }
    metadata = {**metadata, **RAI_FIELDS}

    metadata = _add_provenance_rai_fields(metadata, entry)
    metadata = _add_activities_and_agents(metadata, entry)

    return metadata

def build_dataset_croissant(dataset_id: int, entry: dict, repo_root: Path) -> dict:
    """Build a full Croissant JSON-LD dict for one datasets/dataset_catalog.yaml entry."""
    name = entry["dataset_name"]
    slug = _slugify(name)
    folder_path = entry.get("folder_path")

    description_parts = [name if name.endswith((".", "!", "?")) else name + "."]
    source = entry.get("source")
    if source:
        description_parts.append(f"Source: {_SOURCE_NAMES.get(source, source)}.")
    if "per_round" in entry:
        description_parts.append(f"Per-round granularity: {entry['per_round']}.")
    if "ancillas" in entry:
        description_parts.append(f"Ancilla qubit count: {entry['ancillas']}.")

    metadata: dict[str, Any] = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "name": name,
        "conformsTo": CROISSANT_VERSION,
        "url": entry.get("url"),
    }

    metadata = _add_rai_fields(metadata, entry)

    license_url = entry.get("license")
    if license_url:
        metadata["license"] = license_url
        if entry.get("license_note"):
            description_parts.append(str(entry["license_note"]))
    elif folder_path:
        # Real third-party data (not code-generated) with no confirmed license yet.
        # Deliberately not a fabricated `license` value: schema.org/license expects a
        # URL or CreativeWork, so an unconfirmed placeholder would make this field
        # look authoritative when it isn't. Surface the gap in prose instead.
        description_parts.append(_LICENSE_PENDING_NOTE)

    if entry.get("feature_sets"):
        metadata["variableMeasured"] = list(entry["feature_sets"])
    if entry.get("feature_groups"):
        metadata["featureGroupsMapping"] = {group.lower(): value for group, value in entry["feature_groups"].items()}

    if entry.get("citation"):
        metadata["citeAs"] = str(entry["citation"])
    if entry.get("url"):
        metadata["url"] = str(entry["url"])
    if entry.get("shot_column"):
        metadata["shotColumn"] = entry["shot_column"]
    if entry.get("n_rounds_column"):
        metadata["nRoundsColumn"] = entry["n_rounds_column"]
    if entry.get("target_col"):
        metadata["targetColumn"] = entry["target_col"]
    if entry.get("code_family"):
        metadata["codeFamily"] = entry["code_family"]
    if entry.get("platform"):
        metadata["platform"] = entry["platform"]
    if entry.get("distance"):
        metadata["distance"] = entry["distance"]
    if entry.get("stim_file_mapping_path"):
        metadata["stimFileMappingPath"] = entry["stim_file_mapping_path"]
    if entry.get("raw_features"):
        metadata["rawFeatures"] = entry["raw_features"]
    if entry.get("raw_round_column"):
        metadata["rawRoundColumn"] = entry["raw_round_column"]
    if entry.get("shot_features_sequence"):
        metadata["shotFeaturesSequence"] = entry["shot_features_sequence"]

    for note in entry.get("notes", []):
        description_parts.append(str(note))
    metadata["description"] = " ".join(description_parts)

    if not folder_path:
        # e.g. dataset 4: generated on-the-fly via Stim, not backed by static files.
        # `live_note` lets an entry override the default explanation, e.g. when the
        # generation isn't performed by this repo's own code (a reference/provenance
        # entry for an externally trained checkpoint).
        metadata["isLiveDataset"] = True
        default_live_note = (
            "Generated on-the-fly per experiment (see QEC_model_zoo/qec/dem_sampling.py "
            "and QEC_model_zoo/data/datapipe_stim.py) rather than distributed as static files."
        )
        metadata["description"] += " " + str(entry.get("live_note", default_live_note))
        metadata["distribution"] = []
        metadata["recordSet"] = []
        return metadata

    dataset_dir = repo_root / folder_path
    distribution = []
    record_sets = []
    for split in ("train", "test", "raw"):
        file_object, record_set = _file_object_and_fields(dataset_dir, repo_root, entry, split, slug)
        distribution.append(file_object)
        record_sets.append(record_set)

    metadata["distribution"] = distribution
    metadata["recordSet"] = record_sets

    return metadata


def load_catalog(catalog_path: Path) -> dict[int, dict]:
    with catalog_path.open() as f:
        raw = yaml.safe_load(f)
    return {int(k): v for k, v in raw.items()}


def generate_all(
    catalog_path: Path,
    output_dir: Path,
    dataset_ids: list[int] | None = None,
) -> list[Path]:
    """Generate one Croissant JSON file per requested dataset id. Returns paths written."""
    repo_root = catalog_path.parent.parent
    catalog = load_catalog(catalog_path)
    ids = dataset_ids if dataset_ids is not None else sorted(catalog)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset_id in ids:
        if dataset_id not in catalog:
            raise ValueError(f"dataset id {dataset_id} not found in {catalog_path}")
        entry = catalog[dataset_id]
        metadata = build_dataset_croissant(dataset_id, entry, repo_root)
        slug = _slugify(entry["dataset_name"])
        out_path = output_dir / f"{dataset_id:02d}_{slug}.json"
        out_path.write_text(json.dumps(metadata, indent=2) + "\n")
        written.append(out_path)
    return written
