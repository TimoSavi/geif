#!/usr/bin/env python3
"""
GEIF API Reference Documentation Generator
Generates a comprehensive, single-file Markdown API reference (docs/api_reference.md)
directly from Doxygen comments across all C17 headers and source files in GEIF.
"""

import os
import re
import sys
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
TARGET_FILE = os.path.join(DOCS_DIR, "api_reference.md")
USER_DOCS_TARGET = os.path.expanduser("~/docs/geif-api-reference.md")

def sanitize_gfm_math(text):
    """Ensures math blocks comply with GitHub Flavored Markdown rules."""
    # Replace \text{var_name} with var-name or subscripts
    text = re.sub(r'\\text\{([a-zA-Z0-9]+)_([a-zA-Z0-9]+)\}', r'\1_{\\text{\2}}', text)
    # Remove \left\{ and \right\}
    text = text.replace(r'\left\{', '{').replace(r'\right\}', '}')
    text = text.replace(r'\{', '{').replace(r'\}', '}')
    return text

def parse_doxygen_block(doc_lines):
    """Extracts brief, details, params, returns, and notes from a Doxygen comment block."""
    brief_lines = []
    details_lines = []
    params = []
    returns = []
    notes = []
    
    current_mode = "brief"
    current_param = None
    
    for raw in doc_lines:
        line = raw.strip()
        if line.startswith("/**"):
            line = line[3:].strip()
        elif line.startswith("/*"):
            line = line[2:].strip()
        if line.endswith("*/"):
            line = line[:-2].strip()
        if line.startswith("*"):
            line = line[1:].strip()
            
        if not line:
            if current_mode == "brief" and brief_lines:
                current_mode = "details"
            elif current_mode == "details":
                details_lines.append("")
            continue
            
        if line.startswith("@file"):
            continue
            
        if line.startswith("@brief"):
            current_mode = "brief"
            text = line[6:].strip()
            if text:
                brief_lines.append(text)
            continue
            
        param_match = re.match(r'@param(?:\[(.*?)\])?\s+([a-zA-Z0-9_]+)\s+(.*)', line)
        if param_match:
            current_mode = "param"
            pdir = param_match.group(1) or "in"
            pname = param_match.group(2)
            pdesc = param_match.group(3)
            current_param = {"dir": pdir, "name": pname, "desc": pdesc}
            params.append(current_param)
            continue
            
        return_match = re.match(r'@return\s+(.*)', line)
        if return_match:
            current_mode = "return"
            returns.append(return_match.group(1))
            continue
            
        note_match = re.match(r'@note\s+(.*)', line)
        if note_match:
            current_mode = "note"
            notes.append(note_match.group(1))
            continue

        # Continuation lines
        if current_mode == "brief":
            brief_lines.append(line)
        elif current_mode == "details":
            details_lines.append(line)
        elif current_mode == "param" and current_param:
            current_param["desc"] += " " + line
        elif current_mode == "return" and returns:
            returns[-1] += " " + line
        elif current_mode == "note" and notes:
            notes[-1] += " " + line

    brief = " ".join(brief_lines).strip()
    details = "\n".join(details_lines).strip()
    return brief, details, params, returns, notes

def parse_c_file(filepath):
    """Parses a C header or source file and extracts documented entities."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    file_brief = ""
    entities = []
    
    i = 0
    n = len(lines)
    while i < n:
        raw_line = lines[i].strip()
        if raw_line.startswith("/**"):
            doc_start_line = i + 1
            doc_lines = []
            while i < n:
                doc_lines.append(lines[i])
                if "*/" in lines[i]:
                    break
                i += 1
                
            doc_str = "\n".join(doc_lines)
            if "@file" in doc_str:
                for dl in doc_lines:
                    if "@brief" in dl:
                        file_brief = dl.split("@brief", 1)[1].strip()
                        break
                i += 1
                continue

            # Look ahead for declaration or definition
            sig_lines = []
            decl_start = i + 1
            i += 1
            brace_count = 0
            is_struct = False
            is_enum = False
            is_func = False
            is_define = False

            while i < n:
                l = lines[i].strip()
                if not l or l.startswith("//") or l.startswith("extern \"C\""):
                    i += 1
                    continue
                if l.startswith("#define"):
                    sig_lines.append(l)
                    is_define = True
                    break
                if l.startswith("typedef struct") or l.startswith("typedef enum"):
                    if "struct" in l: is_struct = True
                    if "enum" in l: is_enum = True
                    sig_lines.append(l)
                    # collect until closing semicolon
                    while i < n and ";" not in lines[i]:
                        i += 1
                        sig_lines.append(lines[i].strip())
                    break
                if l.startswith("/**"):
                    # New doc block without a preceding entity
                    i -= 1
                    break

                sig_lines.append(l)
                if ";" in l or "{" in l:
                    is_func = True
                    break
                i += 1

            sig = " ".join(sig_lines).strip()
            # Clean up function signature (remove trailing { or ;)
            if sig.endswith("{"):
                sig = sig[:-1].strip()
            if sig.endswith(";"):
                sig = sig[:-1].strip()

            brief, details, params, returns, notes = parse_doxygen_block(doc_lines)

            # Extract identifier name
            name = ""
            if is_func:
                m = re.search(r'([a-zA-Z0-9_]+)\s*\([^\)]*?\)$', sig)
                if m:
                    name = m.group(1)
                else:
                    # Multi-line args or pointer return
                    m2 = re.search(r'([a-zA-Z0-9_]+)\s*\(', sig)
                    if m2:
                        name = m2.group(1)
            elif is_struct or is_enum:
                m = re.search(r'\}\s*([a-zA-Z0-9_]+)$', sig)
                if m:
                    name = m.group(1)
            elif is_define:
                m = re.search(r'#define\s+([a-zA-Z0-9_]+)', sig)
                if m:
                    name = m.group(1)

            if name and name not in ("if", "for", "while", "switch", "return"):
                entities.append({
                    "name": name,
                    "type": "function" if is_func else ("struct" if is_struct else ("enum" if is_enum else "define")),
                    "signature": sig,
                    "brief": brief,
                    "details": details,
                    "params": params,
                    "returns": returns,
                    "notes": notes,
                    "line": doc_start_line,
                    "file": os.path.relpath(filepath, REPO_ROOT)
                })
        else:
            i += 1

    return file_brief, entities

def main():
    print(f"Generating GEIF API Reference from {REPO_ROOT}...")

    sections = [
        {
            "category": "Public Library C17 API",
            "description": "Public interfaces, forest/ensemble lifecycle, training dispatch, scoring, JSON persistence, and diagnostics.",
            "files": ["include/geif/geif.h", "include/geif/types.h", "include/geif/error.h"]
        },
        {
            "category": "Algorithm Engines & Mathematical Core",
            "description": "Pluggable algorithm operations vtable, geometric partition trees (Bubble, Voronoi, CEIF), Exemplar kernel density, and spatial metric depth traversal.",
            "files": [
                "src/lib/algo.h",
                "src/lib/algo_registry.c",
                "src/lib/algo_bubble.c",
                "src/lib/algo_voronoi.c",
                "src/lib/algo_exemplar.c",
                "src/lib/algo_ceif.c",
                "src/lib/tree_common.c",
                "src/lib/tree_common.h",
                "src/lib/geometry.h"
            ]
        },
        {
            "category": "Forest, Ensembles & Evaluation Engine",
            "description": "Streaming reservoir ingestion, multi-category ensemble routing, calibration, inlier baselines, dimension attribution, and sparse JSON I/O.",
            "files": [
                "src/lib/forest.c",
                "src/lib/train.c",
                "src/lib/evaluate.c",
                "src/lib/ensemble.c",
                "src/lib/reservoir.c",
                "src/lib/json_io.c",
                "src/lib/error.c"
            ]
        },
        {
            "category": "CLI Frontend, Tools & Infrastructure",
            "description": "Command-line pipeline driver, streaming loops, OOM-safe memory allocation, column extraction, color companding, RC configuration parsing, and legacy model migration.",
            "files": [
                "src/cli/main.c",
                "src/cli/columns.c",
                "src/cli/columns.h",
                "src/cli/template.c",
                "src/cli/template.h",
                "src/cli/rcfile.c",
                "src/cli/rcfile.h",
                "src/cli/test_grid.c",
                "src/cli/test_grid.h",
                "src/cli/xmalloc.c",
                "src/cli/xmalloc.h",
                "src/cli/ceif2geif.c"
            ]
        }
    ]

    out = []
    out.append("# GEIF: C17 API & Architecture Reference Manual\n\n")
    out.append("**Author / Maintainer:** Timo Savinen (AI-assisted)  \n")
    out.append("**Version:** 1.1.0  \n")
    out.append("**Repository:** [github.com/TimoSavi/geif](https://github.com/TimoSavi/geif)  \n\n")
    out.append("> This manual is automatically generated from the in-code Doxygen documentation across all C17 headers and implementation files. It documents all public functions, algorithm implementations, mathematical helpers, data structures, and CLI utilities.\n\n")
    out.append("---\n\n")

    # Table of Contents
    out.append("## Table of Contents\n\n")
    out.append("- [**Data Types, Constants & Return Codes**](#data-types-constants--return-codes)\n")
    for sec in sections:
        sec_anchor = sec["category"].lower().replace(" ", "-").replace(",", "").replace("&", "")
        out.append(f"- [**{sec['category']}**](#{sec_anchor})\n")
        for fpath in sec["files"]:
            file_anchor = fpath.replace("/", "").replace(".", "").lower()
            out.append(f"  - [`{fpath}`](#{file_anchor})\n")
    out.append("\n---\n\n")

    # Chapter 1: Constants & Data Types
    out.append("## Data Types, Constants & Return Codes\n\n")
    out.append("### Tuning Constants & Safety Thresholds (`include/geif/types.h`, `src/lib/tree_common.h`)\n\n")
    out.append("| Macro Constant | Value | Description & Safety Semantics |\n")
    out.append("| :--- | :--- | :--- |\n")
    out.append("| `GEIF_DEFAULT_TREE_COUNT` | `100` | Default number of isolation trees per ensemble. |\n")
    out.append("| `GEIF_DEFAULT_SAMPLES_PER_TREE` | `256` | Sub-sample size $\\psi$ drawn without replacement per tree. |\n")
    out.append("| `GEIF_DEFAULT_KAPPA` | `1.25` | Headroom factor for Zero Kelvin baseline depth ($H_{\\max} = 1.25 \\times H_{\\text{train-max}}$). |\n")
    out.append("| `GEIF_MIN_REL_DIST` | `0.05` | Minimum relative Euclidean distance floor for leaf neighbor adjustment. |\n")
    out.append("| `GEIF_OUTER_DECAY_RATE` | `0.10` | Exponential approach rate towards asymptotic 1.0 ceiling in outer space. |\n")
    out.append("| `GEIF_STACK_BUFFER_DIMS` | `64U` | Maximum dimension count for zero-allocation stack scratch buffers. |\n")
    out.append("| `GEIF_MAX_LEAF_NEAREST_SAMPLES` | `32U` | Maximum nearest leaf neighbors tracked via stack max-heap for relative distance. |\n")
    out.append("| `GEIF_MAX_LEAF_NEAREST_DIM_CAP` | `5U` | Dimension threshold ($2^5 = 32$) switching from full orthant scan to binary max-heap. |\n")
    out.append("| `GEIF_INITIAL_TREE_NODES` | `64U` | Initial capacity for flat dynamic tree node allocations. |\n")
    out.append("| `GEIF_INITIAL_NORMALS_COUNT` | `64U` | Initial capacity multiplier for contiguous normal vector buffer. |\n")
    out.append("| `GEIF_INITIAL_LEAF_SAMPLES` | `128U` | Initial capacity for contiguous leaf sample index storage. |\n\n")

    out.append("### Typed Status & Return Codes (`geif_status_t`)\n\n")
    out.append("| Status Code | Value | Meaning / Condition |\n")
    out.append("| :--- | :--- | :--- |\n")
    out.append("| `GEIF_OK` | `0` | Success / normal execution. |\n")
    out.append("| `GEIF_ERR_INVALID_ARG` | `-1` | Null pointer or out-of-range argument provided. |\n")
    out.append("| `GEIF_ERR_OUT_OF_MEMORY` | `-2` | Heap memory allocation failed. |\n")
    out.append("| `GEIF_ERR_EMPTY_DATASET` | `-3` | Forest fed with zero samples or empty CSV. |\n")
    out.append("| `GEIF_ERR_FILE_IO` | `-4` | Unable to open, read, or write file descriptor. |\n")
    out.append("| `GEIF_ERR_JSON_PARSE` | `-5` | Malformed JSON format encountered during model loading. |\n")
    out.append("| `GEIF_ERR_NOT_TRAINED` | `-6` | Scoring attempted on un-trained forest. |\n")
    out.append("| `GEIF_ERR_DIMENSION_MISMATCH` | `-7` | Query coordinate vector length does not match model dimensionality. |\n\n")

    out.append("### Algorithm Engine Selector (`geif_algo_type_t`)\n\n")
    out.append("| Enum Constant | Value | CLI Name | Description |\n")
    out.append("| :--- | :--- | :--- | :--- |\n")
    out.append("| `GEIF_ALGO_CEIF` | `0` | `ceif` / `eif` | Data-anchored isotropic Gaussian cuts with Zero Kelvin scale floor and outer decay. |\n")
    out.append("| `GEIF_ALGO_BUBBLE` | `1` | `bubble` | Hyperspherical Bubble tree cavity carving with median quickselect (default). |\n")
    out.append("| `GEIF_ALGO_EXEMPLAR` | `2` | `exemplar` | Direct SIMD Cauchy kernel density evaluation on reservoir pool samples. |\n")
    out.append("| `GEIF_ALGO_VORONOI` | `3` | `voronoi` | Scale-invariant Voronoi perpendicular bisector hyperplane cuts. |\n\n")
    out.append("---\n\n")

    total_funcs = 0

    # Chapters by module category
    for sec in sections:
        sec_anchor = sec["category"].lower().replace(" ", "-").replace(",", "").replace("&", "")
        out.append(f"## {sec['category']}\n\n")
        out.append(f"{sec['description']}\n\n")

        for fpath in sec["files"]:
            abs_path = os.path.join(REPO_ROOT, fpath)
            if not os.path.exists(abs_path):
                continue

            file_anchor = fpath.replace("/", "").replace(".", "").lower()
            file_brief, entities = parse_c_file(abs_path)

            out.append(f"### `{fpath}`\n\n")
            if file_brief:
                out.append(f"**Module Purpose:** {file_brief}\n\n")

            if not entities:
                out.append("*Header / definition file containing structs, constants, and macros.*\n\n")
                out.append("---\n\n")
                continue

            for ent in entities:
                total_funcs += 1
                out.append(f"#### [`{ent['name']}`]({ent['file']}#L{ent['line']})\n\n")
                out.append("```c\n" + ent["signature"] + ";\n```\n\n")

                if ent["brief"]:
                    out.append(f"**Description:** {ent['brief']}\n\n")
                if ent["details"]:
                    out.append(f"{ent['details']}\n\n")

                if ent["params"]:
                    out.append("**Parameters:**\n\n")
                    out.append("| Parameter | Direction | Description |\n")
                    out.append("| :--- | :--- | :--- |\n")
                    for p in ent["params"]:
                        direction = f"`[{p['dir']}]`" if p['dir'] else "`[in]`"
                        out.append(f"| `{p['name']}` | {direction} | {p['desc']} |\n")
                    out.append("\n")

                if ent["returns"]:
                    out.append(f"**Returns:** {' '.join(ent['returns'])}\n\n")

                if ent["notes"]:
                    out.append("> [!NOTE]\n")
                    for n in ent["notes"]:
                        out.append(f"> {n}\n")
                    out.append("\n")

                out.append("---\n\n")

    final_content = sanitize_gfm_math("".join(out))
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(final_content)

    print(f"Generated {TARGET_FILE} ({total_funcs} documented functions/entities).")

    # Copy to ~/docs/ per user guidelines
    os.makedirs(os.path.dirname(USER_DOCS_TARGET), exist_ok=True)
    shutil.copyfile(TARGET_FILE, USER_DOCS_TARGET)
    print(f"Copied to {USER_DOCS_TARGET}.")

if __name__ == "__main__":
    main()
