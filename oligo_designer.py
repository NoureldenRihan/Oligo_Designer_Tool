#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PCA OLIGO DESIGNER                                  ║
║                                                                              ║
║  Designs overlapping alternating oligos for Polymerase Cycling Assembly.    ║
║                                                                              ║
║  Libraries used:                                                             ║
║    Tm calculation  → primer3-py  (SantaLucia 1998 + salt correction)        ║
║    Rev. complement → Biopython Seq                                           ║
║    GC content      → Biopython gc_fraction                                  ║
║    Output          → pandas DataFrame + CSV                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEPENDENCIES:
    pip install primer3-py biopython pandas

WHAT THIS SCRIPT DOES:
    1. Takes a DNA sequence (any length).
    2. Splits it into fragments of a specified length range — each fragment
       will be assembled independently by PCA in the lab.
    3. For each fragment, tiles alternating forward (F) and reverse (R) oligos
       that overlap with their neighbours. These overlaps are what the
       polymerase extends during PCA cycling to build the full fragment.
    4. Chooses overlaps that satisfy four quality criteria:
         (a) Tm in your target range              → reliable annealing
         (b) GC% in your target range             → avoids GC-extreme overlaps
         (c) No homopolymer runs (e.g. AAAA)      → prevents polymerase slippage
         (d) No dinucleotide repeats (e.g. ATAT)  → prevents slippage + mispriming
    5. Checks each full oligo for potential hairpin formation.
    6. If a perfect overlap cannot be found, uses the closest available option
       and flags it clearly in the warnings column.

USAGE EXAMPLE:
    from pca_oligo_designer import design_pca_oligos

    df = design_pca_oligos(
        sequence="ATGC...",
        output_csv="my_oligos.csv"   # set to None to skip saving
    )

    # Or with full custom parameters:
    df = design_pca_oligos(
        sequence="ATGC...",
        fragment_length_range=(400, 500),
        oligo_length_range=(40, 60),
        overlap_length_range=(15, 25),
        overlap_tm_range=(55, 65),
        overlap_gc_range=(40.0, 70.0),
        max_homopolymer_run=4,
        max_dinucleotide_repeat=3,
        repeat_vicinity_buffer=5,
        hairpin_dg_threshold=-2.0,
        misprime_length=18,
        misprime_tip_length=6,
        misprime_max_mismatches=8,
        output_csv="my_oligos.csv"
    )

ALL PARAMETERS AND THEIR DEFAULTS:
    fragment_length_range  : (400, 500)   bp   — None = whole seq as one fragment
    oligo_length_range     : (40, 60)     bp
    overlap_length_range   : (15, 25)     bp
    overlap_tm_range       : (55, 65)     C
    overlap_gc_range       : (40.0, 70.0) %    — GC% bounds on overlap region
    max_homopolymer_run    : 4            bp   — flag runs of same base >= this
    max_dinucleotide_repeat: 3            reps — flag dinucleotide repeated >= this
    repeat_vicinity_buffer : 5            bp   — extra bp scanned each side of overlap for repeat proximity
    hairpin_dg_threshold   : -4.0         kcal/mol — flag hairpins with dG <= this (primer3.calc_hairpin)
    misprime_length        : 18           bp   — 3' probe length for misprime scan
    misprime_tip_length    : 6            bp   — exact-match tip length at 3' end
    misprime_max_mismatches: 4            mismatches — max allowed in non-tip region
    junction_overlap_length: 25           bp — overlap shared between adjacent fragments for stitching PCR
    junction_tm_range      : (58.0, 68.0) C  — Tm range for junction overlaps (wider than PCA overlap range)
    mv_conc                : 50           mM   — monovalent salt (Na+/K+)
    dv_conc                : 1.5          mM   — divalent salt (Mg2+)
    dntp_conc              : 0.2          mM   — dNTPs
    dna_conc               : 250          nM   — oligo concentration
    output_csv             : "pca_oligos.csv"
"""

import re
import pandas as pd
import primer3
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from typing import Tuple, Optional, List

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# All defaults are based on established PCA literature practices.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_FRAGMENT_LENGTH_RANGE   = (400, 500)
DEFAULT_OLIGO_LENGTH_RANGE      = (40, 60)
DEFAULT_OVERLAP_LENGTH_RANGE    = (15, 25)
DEFAULT_OVERLAP_TM_RANGE        = (55.0, 65.0)
DEFAULT_OVERLAP_GC_RANGE        = (40.0, 70.0)   # see Problem 1 notes below
DEFAULT_MAX_HOMOPOLYMER_RUN     = 4               # see Problem 2 notes below
DEFAULT_MAX_DINUCLEOTIDE_REPEAT = 3               # see Problem 3 notes below
DEFAULT_REPEAT_VICINITY_BUFFER  = 5               # extra bp scanned each side of overlap for repeat proximity
DEFAULT_HAIRPIN_DG_THRESHOLD    = -4.0            # kcal/mol — primer3.calc_hairpin() cutoff (more negative = more stable)
                                                  # Changed from -2.0: at -2.0 nearly every 40-60bp oligo gets flagged
                                                  # because short self-complementary stretches are statistically common.
                                                  # -4.0 targets hairpins stable enough to genuinely compete with PCA annealing.
DEFAULT_MV_CONC                 = 50.0    # mM
DEFAULT_DV_CONC                 = 1.5     # mM
DEFAULT_DNTP_CONC               = 0.2     # mM
DEFAULT_DNA_CONC                = 250.0   # nM
DEFAULT_OUTPUT_CSV              = "pca_oligos.csv"

# Misprime checker defaults — modelled after DNAWorks (Hoover & Lubkowski 2002)
# See check_misprime_all() and the SECTION 2 comment block for full explanation.
DEFAULT_MISPRIME_LENGTH          = 18   # bp from 3' end of each oligo to compare
DEFAULT_MISPRIME_TIP_LENGTH      = 6    # bp at the very 3' tip that must match exactly
DEFAULT_MISPRIME_MAX_MISMATCHES  = 4    # max mismatches allowed in the non-tip region
                                        # Changed from 6: 6/12 = 50% mismatch tolerance is too permissive —
                                        # a duplex that is 50% mismatched will not support efficient polymerase
                                        # extension under real PCA conditions, so those flags were mostly false
                                        # positives. 4/12 = 33% tolerance still catches real mispriming risks
                                        # (especially 0-2 mismatch hits) while reducing noise on clean sequences.

DEFAULT_JUNCTION_OVERLAP_LENGTH  = 25   # bp shared between adjacent fragments for stitching PCR
                                        # This is the overlap the assembled PCA products need
                                        # to anneal to each other in the downstream stitching PCR.
                                        # 25 bp gives a Tm of ~55-65C for typical sequences.
                                        # Set to 0 to disable junction design (single-fragment or manual).

DEFAULT_JUNCTION_TM_RANGE        = (58.0, 68.0)
                                        # Tm range specifically for inter-fragment junction overlaps.
                                        # Intentionally wider and slightly higher than DEFAULT_OVERLAP_TM_RANGE
                                        # because junctions are used in stitching PCR (typically run at
                                        # 60-65C annealing), not in PCA cycling (typically 55-60C).
                                        # Using the same Tm range for both caused borderline junction hits
                                        # (e.g. 65.1C) to be flagged even though they are perfectly fine
                                        # for stitching PCR conditions.


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: BASIC SEQUENCE UTILITIES
# Backed by primer3 and Biopython so we don't reinvent the wheel.
# ──────────────────────────────────────────────────────────────────────────────

def calc_tm(seq: str, mv_conc: float, dv_conc: float,
            dntp_conc: float, dna_conc: float) -> float:
    """
    Calculate the melting temperature (Tm) of a DNA sequence.

    Uses the SantaLucia 1998 nearest-neighbor thermodynamic model via
    primer3, which is the gold standard for short oligo Tm calculation.
    Salt correction is also applied using the SantaLucia method, accounting
    for the stabilizing effect of Na+ and Mg2+ ions on the DNA duplex.

    Parameters:
        seq       : DNA sequence (5'→3', top strand)
        mv_conc   : monovalent salt concentration in mM (e.g. NaCl, KCl)
        dv_conc   : divalent salt concentration in mM (e.g. MgCl2)
        dntp_conc : dNTP concentration in mM (dNTPs chelate Mg2+, reducing
                    free Mg2+ and affecting the effective salt correction)
        dna_conc  : total oligo strand concentration in nM
    """
    return primer3.calc_tm(
        seq.upper(),
        mv_conc=mv_conc,
        dv_conc=dv_conc,
        dntp_conc=dntp_conc,
        dna_conc=dna_conc,
        tm_method='santalucia',
        salt_corrections_method='santalucia',
    )


def reverse_complement(seq: str) -> str:
    """
    Return the reverse complement of a DNA sequence.

    This is what you order for a reverse-strand oligo — you give the lab
    the sequence as it reads 5'→3' on the bottom strand, which is the
    reverse complement of the top strand sequence.
    """
    return str(Seq(seq).reverse_complement())


def gc_percent(seq: str) -> float:
    """Return GC content of a sequence as a percentage (0–100)."""
    return round(gc_fraction(seq) * 100, 1)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: SEQUENCE QUALITY CHECKS
#
# These four functions address known failure modes for PCA oligo design.
# Each function returns a list of warning strings — empty list means no issues.
# They are designed to be called independently so they can be easily tested,
# updated, or disabled individually without touching the tiling logic.
# ──────────────────────────────────────────────────────────────────────────────


# ── Problem 1: GC-rich or GC-poor overlaps ────────────────────────────────────
#
# WHY THIS MATTERS:
#   The GC content of an overlap affects both its thermodynamic behaviour and
#   its structural risk:
#
#   TOO HIGH GC (>70%): GC base pairs (3 hydrogen bonds) are much more stable
#     than AT pairs (2 bonds). A very GC-rich overlap can:
#       - Form stable secondary structures / hairpins that compete with the
#         intended inter-oligo hybridization.
#       - Look similar to other GC-rich overlaps in the pool, causing an oligo
#         to anneal to the wrong partner (mispriming), leading to mis-assembled
#         products.
#       - Cause the polymerase to stall during extension.
#
#   TOO LOW GC (<40%): AT-rich overlaps may appear to have an acceptable Tm
#     under certain salt conditions but are inherently less stable and more
#     prone to breathing (partial melting), leading to unreliable annealing
#     during the PCA annealing step.
#
# WHAT WE DO:
#   Flag any overlap whose GC% falls outside the specified range.
#   The algorithm will try other (oligo_len, ov_len) combinations first
#   before accepting an out-of-range GC% with a warning.
#
# VALIDATION NOTE:
#   The GC bounds are computationally verifiable — you can cross-check flagged
#   overlaps in mfold or RNAfold to confirm they have worse secondary structure
#   predictions. Full validation of improved PCA success rate requires wet lab.

def check_overlap_gc(overlap_seq: str,
                     gc_range: Tuple[float, float]) -> List[str]:
    """
    Check whether the overlap GC content is within the acceptable range.

    Args:
        overlap_seq : the overlap DNA sequence
        gc_range    : (min_gc%, max_gc%) acceptable range

    Returns:
        List of warning strings (empty = no issues).
    """
    gc = gc_percent(overlap_seq)
    min_gc, max_gc = gc_range
    warnings = []

    if gc > max_gc:
        warnings.append(
            f"Overlap GC% is {gc}% which is ABOVE the maximum {max_gc}% — "
            f"risk of secondary structure formation and mispriming with other "
            f"GC-rich oligos in the pool"
        )
    elif gc < min_gc:
        warnings.append(
            f"Overlap GC% is {gc}% which is BELOW the minimum {min_gc}% — "
            f"AT-rich overlaps may anneal unreliably during PCA cycling"
        )
    return warnings


# ── Problem 2: Homopolymer runs ───────────────────────────────────────────────
#
# WHY THIS MATTERS:
#   A homopolymer run is a stretch of the same nucleotide repeated consecutively,
#   e.g. AAAAAAA, TTTTTT, GGGG, CCCC.
#
#   These are problematic for two reasons:
#
#   1. POLYMERASE SLIPPAGE: During PCA, the polymerase extends an annealed
#      overlap. When it hits a long run of the same base, it can "slip" — losing
#      its position in the template and re-annealing at the wrong position within
#      the run. This produces insertions or deletions (frameshifts) in the
#      assembled product. This is well-documented in the sequencing literature
#      and is why homopolymer regions are a known challenge in Nanopore and
#      Illumina sequencing as well.
#
#   2. Tm CALCULATION UNRELIABILITY: The SantaLucia nearest-neighbor model
#      calculates Tm based on all dinucleotide step parameters. A homopolymer
#      run collapses to a single repeated dinucleotide (e.g. AA/TT repeated N
#      times), which does not reflect the cooperative melting behaviour of
#      a real mixed sequence. The Tm may be over- or under-estimated.
#
# WHAT WE DO:
#   Use a regex to find any run of the same base with length >= threshold.
#   Default threshold is 4 — runs of 4+ are consistently reported as
#   problematic in synthesis literature.
#
# VALIDATION NOTE:
#   The detection is fully dry-lab verifiable (inspect output vs sequence).
#   The claim that avoiding these runs improves PCA fidelity is literature-
#   supported (cite polymerase slippage studies) but wet-lab-dependent for
#   original experimental evidence.

def check_homopolymer(seq: str, max_run: int) -> List[str]:
    """
    Detect homopolymer runs in a sequence.

    A homopolymer run is a consecutive repeat of the same nucleotide,
    e.g. AAAA, TTTTT, GGGG. Runs >= max_run are flagged.

    Args:
        seq     : DNA sequence to scan
        max_run : minimum run length to flag (default 4)

    Returns:
        List of warning strings (empty = no issues).
    """
    warnings = []
    # Regex: match any single character repeated max_run or more times.
    # r'(.)\1{n,}' means: capture any char (.), then match it again n+ more times.
    # So for max_run=4, we look for the char + 3 more = total 4+.
    pattern = re.compile(r'(.)\1{' + str(max_run - 1) + r',}')
    matches = pattern.finditer(seq.upper())

    for match in matches:
        run = match.group()
        warnings.append(
            f"Homopolymer run detected: '{run}' ({len(run)} bp of '{run[0]}') "
            f"at position {match.start()}–{match.end()-1} — "
            f"risk of polymerase slippage producing insertions/deletions in "
            f"the assembled product; Tm calculation may also be unreliable"
        )
    return warnings


# ── Problem 3: Dinucleotide repeats ──────────────────────────────────────────
#
# WHY THIS MATTERS:
#   A dinucleotide repeat is a two-base motif repeated consecutively,
#   e.g. ATATATATAT, CGCGCGCG, TATATAT.
#
#   These are problematic for the same polymerase slippage reason as
#   homopolymers, but they also introduce a second distinct failure mode:
#
#   MISPRIMING / WRONG-PARTNER ANNEALING: In a PCA reaction, you have many
#   different oligos in the same tube. If multiple overlaps contain similar
#   dinucleotide repeat motifs, they can anneal to the wrong partners — an
#   oligo designed to pair with oligo #4 might partially anneal to oligo #8
#   if both have ATATAT-like regions. This produces chimeric, incorrectly
#   assembled fragments that can be very hard to detect without sequencing.
#
# WHAT WE DO:
#   Use a regex to find any dinucleotide (two chars) repeated >= threshold
#   times consecutively. Default is 3 repetitions (6+ bp of repeat), which
#   is a conservative but practical cutoff.
#
# VALIDATION NOTE:
#   Same as homopolymer — detection is dry-lab verifiable; the biological
#   consequence claim is literature-supported but wet-lab-dependent.

def check_dinucleotide_repeat(seq: str, max_repeats: int) -> List[str]:
    """
    Detect consecutive dinucleotide repeats in a sequence.

    E.g. ATATATATAT is the dinucleotide AT repeated 5 times.
    Repeats >= max_repeats are flagged.

    Args:
        seq         : DNA sequence to scan
        max_repeats : minimum number of consecutive repeats to flag (default 3)

    Returns:
        List of warning strings (empty = no issues).
    """
    warnings = []
    # Regex: capture any two characters (..), then match that same pair
    # repeated (max_repeats-1) or more additional times.
    # r'(..)\1{n,}' means: capture 2 chars, then see the same 2 chars n+ more times.
    pattern = re.compile(r'(..)\1{' + str(max_repeats - 1) + r',}')
    matches = pattern.finditer(seq.upper())

    for match in matches:
        repeat_unit = match.group(1)   # the dinucleotide being repeated
        full_match  = match.group()    # the full repeated stretch
        n_repeats   = len(full_match) // 2
        warnings.append(
            f"Dinucleotide repeat detected: '{repeat_unit}' repeated {n_repeats}x "
            f"('{full_match}') at position {match.start()}–{match.end()-1} — "
            f"risk of polymerase slippage and mispriming with other oligos "
            f"containing similar repeat motifs"
        )
    return warnings


# ── Edge clipping fix: vicinity buffer for repeats ────────────────────────────
#
# THE PROBLEM WE OBSERVED IN TESTING:
#   During Test C, the algorithm placed overlaps that technically did not
#   CONTAIN a homopolymer or dinucleotide repeat — but the overlap STARTED
#   right at the end of a bad region. For example, an overlap starting at
#   position 48 when a homopolymer AAAAAAAA ran from 42-49. The overlap
#   began with "AA" — below our 4-base threshold — so no flag was raised.
#   But those two A's are still part of the problematic region and can
#   contribute to polymerase slippage and Tm calculation errors.
#
# THE FIX — VICINITY BUFFER:
#   Before accepting any overlap, we scan a wider zone:
#     [ overlap_start - buffer, overlap_end + buffer ]
#   capped to the fragment boundaries.
#
#   If a homopolymer run or dinucleotide repeat is detected anywhere in this
#   extended zone, the overlap is flagged even if the overlap itself looks
#   clean. The buffer size is user-configurable (repeat_vicinity_buffer,
#   default 5 bp). This catches edge cases where bad regions are clipped
#   but not fully avoided.
#
# WHY 5 bp DEFAULT:
#   5 bp is larger than our homopolymer threshold (4 bp) and large enough
#   to catch the edge clip cases we observed, while not being so large that
#   it makes avoidance impossible in sequences with scattered short repeats.
#
# NOTE: This function is called IN ADDITION TO check_homopolymer and
#   check_dinucleotide_repeat on the overlap itself. The two checks serve
#   different purposes: the direct checks catch repeats inside the overlap;
#   this function catches repeats that are adjacent to the overlap boundary.

def check_repeat_vicinity(
    fragment: str,
    overlap_start: int,
    overlap_end: int,
    buffer: int,
    max_homopolymer: int,
    max_dinucleotide: int,
) -> List[str]:
    """
    Scan a buffered zone around the overlap for homopolymer / dinucleotide
    repeats that are adjacent to but not fully inside the overlap.

    This catches the edge-clipping problem: an overlap that starts right
    at the tail of a homopolymer run won't be caught by checking the overlap
    alone, but will be caught by scanning the vicinity.

    Args:
        fragment       : the full fragment sequence being tiled
        overlap_start  : start position of the overlap (0-based, on fragment)
        overlap_end    : end position of the overlap (exclusive)
        buffer         : number of extra bases to scan on each side (default 5)
        max_homopolymer: homopolymer run threshold (same as check_homopolymer)
        max_dinucleotide: repeat count threshold (same as check_dinucleotide_repeat)

    Returns:
        List of warning strings for any repeat found in the vicinity but
        OUTSIDE the overlap itself (so warnings are clearly labelled as
        'proximity' warnings, not overlap warnings).
    """
    frag_len = len(fragment)

    # Extended zone: buffer bp before and after the overlap, clamped to [0, frag_len]
    zone_start = max(0, overlap_start - buffer)
    zone_end   = min(frag_len, overlap_end + buffer)
    zone_seq   = fragment[zone_start: zone_end]

    # Run the same checks on the extended zone
    zone_hp = check_homopolymer(zone_seq, max_homopolymer)
    zone_dn = check_dinucleotide_repeat(zone_seq, max_dinucleotide)

    warnings = []

    for w in zone_hp:
        # Only report if the problem is in the buffer zone (not the overlap itself)
        # We check this by running the same detector on just the overlap — if it
        # wasn't caught there, it must be in the buffer zone.
        overlap_hp = check_homopolymer(fragment[overlap_start:overlap_end], max_homopolymer)
        if not overlap_hp:
            warnings.append(
                f"Proximity warning — homopolymer run near overlap boundary "
                f"(within {buffer} bp buffer): {w} "
                f"The overlap may partially anneal into a problematic region."
            )

    for w in zone_dn:
        overlap_dn = check_dinucleotide_repeat(fragment[overlap_start:overlap_end], max_dinucleotide)
        if not overlap_dn:
            warnings.append(
                f"Proximity warning — dinucleotide repeat near overlap boundary "
                f"(within {buffer} bp buffer): {w} "
                f"The overlap may partially anneal into a repetitive region."
            )

    return warnings


# ── Problem 4: Oligo hairpin / self-complementarity ──────────────────────────
#
# WHY THIS MATTERS:
#   A hairpin forms when a single-stranded oligo folds back on itself because
#   part of its own sequence is complementary to another part of itself.
#
#   Example: 5'-GCATCG---CGATGC-3' → the 5' end (GCATCG) is complementary to
#   the 3' end (CGATGC), so the oligo folds into a stem-loop structure.
#
#   Hairpins are problematic for PCA because:
#   - An oligo that is folded into a hairpin has its ends tied up in
#     intramolecular base-pairing, reducing its effective concentration
#     available for the intended inter-oligo hybridization.
#   - Hairpins compete with productive annealing at the PCA annealing step,
#     reducing yield and potentially causing incorrect assemblies.
#   - The primer3 Tm calculation assumes the oligo is single-stranded and
#     freely available; hairpins violate this assumption.
#
# WHAT WE DO (v2 — thermodynamic, not heuristic):
#   We call primer3.calc_hairpin() which runs a real thermodynamic folding
#   calculation and returns a ΔG value (kcal/mol) for the most stable
#   intramolecular hairpin the oligo can form.
#
#   ΔG interpretation:
#     ΔG = 0       → no stable hairpin at all
#     ΔG = -1.0    → weak hairpin, unlikely to compete with productive annealing
#     ΔG = -2.0    → moderate — our default warning threshold
#     ΔG = -4.0+   → strong hairpin, will significantly compete with PCA
#
#   The threshold is user-configurable (hairpin_dg_threshold, default -2.0)
#   so you can be more or less strict depending on your reaction conditions.
#   More negative threshold = only flag severe hairpins.
#   Less negative threshold = flag even weak hairpins (more conservative).
#
# WHY THIS IS BETTER THAN THE OLD APPROACH:
#   The previous version used a sliding window to find self-complementary
#   regions of >= N bp. This produced massive false positive rates because
#   any 40-60 bp oligo will statistically contain short self-complementary
#   stretches by chance — they don't all form stable hairpins. The ΔG
#   approach tells you whether those stretches are actually thermodynamically
#   stable enough to compete with productive hybridization. The result:
#   far fewer false positives, much more actionable warnings.
#
# CUSTOMIZABILITY:
#   hairpin_dg_threshold: float (default -2.0 kcal/mol)
#     Set more negative (e.g. -4.0) to only flag strong hairpins.
#     Set less negative (e.g. -1.0) to flag even weak ones.
#
# VALIDATION NOTE:
#   This is the most dry-lab-validatable of all four checks. The ΔG value
#   from primer3.calc_hairpin() can be directly cross-checked with mfold
#   (http://www.unafold.org/). You can demonstrate that flagged oligos have
#   more negative ΔG than unflagged ones — a publishable dry-lab result.
#   Full wet-lab validation would compare PCA yield with and without
#   hairpin-filtered oligo sets.

def check_hairpin(oligo_seq: str, dg_threshold: float) -> List[str]:
    """
    Check a full oligo for thermodynamically stable hairpin formation.

    Uses primer3.calc_hairpin() to compute the folding free energy (ΔG)
    of the most stable intramolecular hairpin the oligo can form under
    the standard reaction conditions assumed by primer3.

    A more negative ΔG = more stable hairpin = more problematic.
    The default threshold of -2.0 kcal/mol corresponds to hairpins that
    are stable enough to compete meaningfully with productive PCA annealing.

    Args:
        oligo_seq     : full oligo sequence (as it would be ordered, 5'→3')
        dg_threshold  : ΔG cutoff in kcal/mol — flag if dG <= this value
                        Default: -2.0  (more negative = only flag stronger hairpins)

    Returns:
        List of warning strings (empty = no issues).
    """
    result = primer3.calc_hairpin(oligo_seq.upper())
    dg = result.dg / 1000.0   # primer3 returns dG in cal/mol — convert to kcal/mol

    if dg <= dg_threshold:
        return [
            f"Stable hairpin detected: primer3 calculated dG = {dg:.2f} kcal/mol "
            f"(threshold is {dg_threshold} kcal/mol) — the oligo may fold on itself "
            f"under reaction conditions, competing with productive PCA annealing. "
            f"Consider relaxing the hairpin_dg_threshold if this oligo cannot be "
            f"avoided, or adjust oligo boundaries to escape this fold."
        ]
    return []


# ── Problem 5: Mispriming ─────────────────────────────────────────────────────
#
# WHY THIS MATTERS:
#   In a PCA reaction, all oligos are in the same tube simultaneously. The
#   polymerase doesn't know which oligo is "supposed" to prime at which position —
#   it will extend from any 3' end that finds a complementary sequence stable
#   enough to anneal. A misprime occurs when the 3' end of one oligo anneals
#   to an UNINTENDED position elsewhere in the sequence pool and gets extended.
#
#   The consequences:
#   - The polymerase extends from the wrong position, stitching together two
#     sequence segments that shouldn't be joined. The resulting product is a
#     CHIMERA — it has the right length on a gel but wrong sequence internally.
#   - Multiple misprimes in one reaction produce a smear on the gel (many wrong
#     sized products), or a correctly-sized band with a wrong sequence that only
#     sequencing reveals. Both are expensive, time-consuming failures in the lab.
#
#   Mispriming is MORE likely when:
#   - The sequence has repetitive regions (similar sequences appear multiple times)
#   - The GC content is high (GC-rich 3' ends are very stable even at mismatched sites)
#   - The annealing temperature is too low (more non-specific annealing overall)
#   - Oligos are long (longer 3' tails have more chances to find partial matches)
#
# WHAT WE DO (modelled after DNAWorks, Hoover & Lubkowski 2002):
#   For each oligo, we take the last misprime_length bp from its 3' end (the
#   "probe"). We then slide this probe across the ENTIRE input sequence on BOTH
#   strands, looking for unintended match sites. A site is flagged as a potential
#   misprime if:
#     (a) The last tip_length bp of the probe match EXACTLY at that position
#         (the "tip" — the very 3' end is the most critical for priming)
#     (b) The remaining non-tip bases have <= max_mismatches total mismatches
#
#   We exclude the oligo's own legitimate binding site (plus a small buffer)
#   from the search so we don't flag the intended annealing as a misprime.
#
# ARCHITECTURE — WHY THIS IS A POST-PROCESSING STEP:
#   Unlike GC, homopolymer, and dinucleotide checks (which run inside the tiling
#   loop on individual overlaps), the misprime check must run AFTER all oligos
#   are placed. This is because:
#   - We need the final oligo sequence, not just the overlap
#   - We need the full input sequence to compare against ALL positions
#   - We're comparing each oligo against every other part of the sequence,
#     which requires knowing where all oligos are first
#
#   It runs after _tile_fragment, on the completed DataFrame.
#
# DEFAULT PARAMETERS (matching DNAWorks for comparability):
#   misprime_length         : 18  bp — probe length from 3' end
#   misprime_tip_length     : 6   bp — exact-match tip at 3' end
#   misprime_max_mismatches : 8   mismatches allowed in the non-tip region
#
#   Interpretation: we're flagging any unintended site where the last 6 bp
#   match perfectly AND the preceding 12 bp have <= 8 mismatches. This is
#   deliberately permissive (8/12 = 67% mismatch tolerance) to catch even
#   weak potential misprimes — better to over-warn than miss a real one.
#   Users can tighten this with smaller max_mismatches.
#
# VALIDATION NOTE:
#   Detection is computationally verifiable (you can manually BLAST each
#   oligo against the target sequence to confirm hits). Whether detected
#   misprimes actually cause failure in practice depends on reaction conditions
#   (temperature, Mg2+, polymerase fidelity) — wet-lab validation required
#   for definitive claims about impact on assembly success rate.

def _build_tip_index(seq: str, probe_len: int, tip_length: int) -> dict:
    """
    Build a lookup index mapping each tip sequence to all window-start positions
    in `seq` where that tip appears at the END of a `probe_len`-length window.

    This is the key performance optimization for the misprime check.

    OLD APPROACH (O(n) per oligo):
        For each oligo, slide a probe across every position in the full sequence.
        On a 50KB gene with 1000 oligos this means ~100M character comparisons —
        potentially minutes of runtime in pure Python.

    NEW APPROACH (O(n) once to build index, O(hits) per oligo):
        Pre-compute, for every possible tip sequence, the list of positions in
        the full sequence where that exact tip appears at the end of a probe-length
        window. Then per oligo, look up only those positions — skipping the
        entire rest of the sequence. For a non-repetitive sequence, each 6-mer
        tip occurs ~n/4^6 times on average, so lookup is ~12 positions on a
        50KB sequence instead of 50,000. Speedup is roughly 4000x for clean seqs.

    Args:
        seq        : the full sequence (or its reverse complement) to index
        probe_len  : length of the sliding probe window
        tip_length : number of bp at the end of each window that form the tip key

    Returns:
        dict mapping tip_sequence (str) → list of window start positions (int)
    """
    from collections import defaultdict
    index = defaultdict(list)
    n = len(seq)
    for i in range(n - probe_len + 1):
        # The tip is the last tip_length bases of the window seq[i : i+probe_len]
        tip = seq[i + probe_len - tip_length : i + probe_len]
        index[tip].append(i)
    return dict(index)


def check_misprime_all(
    df: 'pd.DataFrame',
    fragments: List[str],
    full_sequence: str,
    misprime_length: int,
    tip_length: int,
    max_mismatches: int,
) -> 'pd.DataFrame':
    """
    Post-processing misprime check: scan each oligo's 3' end for accidental
    matches at unintended positions in the full sequence (both strands).

    This must run after tiling is complete because it compares each oligo
    against the entire input sequence, requiring all oligo positions to be
    known first.

    PERFORMANCE: Uses _build_tip_index() to pre-index all tip k-mers in the
    sequence. Per-oligo cost is O(hits_for_that_tip) instead of O(sequence_length),
    giving ~1000-4000x speedup on large sequences like 50KB genes.

    Args:
        df              : completed oligo DataFrame from _tile_fragment
        fragments       : list of fragment sequences (used to map local→global positions)
        full_sequence   : the complete input DNA sequence
        misprime_length : bp from each oligo's 3' end to use as the search probe
        tip_length      : bp at the very 3' tip that must match exactly at any hit site
        max_mismatches  : max mismatches allowed in the non-tip region of the probe

    Returns:
        Updated DataFrame with 'flag_misprime' column added and misprime
        warnings appended to the 'warnings' column.
    """
    full_seq = full_sequence.upper()
    full_rc  = reverse_complement(full_seq)   # bottom strand, read 5'→3'
    n        = len(full_seq)

    # ── Build tip indexes ONCE for both strands ───────────────────────────────
    # This is the core of the performance fix. Instead of sliding the probe
    # across every position for every oligo, we pre-build a dict that maps
    # each possible tip k-mer to the list of positions where it appears.
    # Then per oligo we only visit those positions — skipping everything else.
    top_index = _build_tip_index(full_seq, misprime_length, tip_length)
    rc_index  = _build_tip_index(full_rc,  misprime_length, tip_length)

    # ── Build fragment global-start lookup ────────────────────────────────────
    # Maps fragment index → start position in the full sequence.
    # Needed to convert (fragment_idx, local_position) → global position.
    frag_global_starts = []
    cumulative = 0
    for frag in fragments:
        frag_global_starts.append(cumulative)
        cumulative += len(frag)

    misprime_flags = []
    misprime_warns = []

    for _, row in df.iterrows():
        oligo_seq  = str(row['oligo_sequence']).upper()
        frag_idx   = int(row['fragment']) - 1   # convert 1-based to 0-based
        frag_start = int(row['frag_start'])
        frag_end   = int(row['frag_end'])

        # Global coordinates of this oligo on the full input sequence
        global_start = frag_global_starts[frag_idx] + frag_start
        global_end   = frag_global_starts[frag_idx] + frag_end

        # ── Build the probe: last misprime_length bp of the oligo (the 3' end) ──
        # If the oligo is shorter than misprime_length, use the whole oligo.
        probe     = oligo_seq[-misprime_length:]
        probe_len = len(probe)

        # The "tip" — the very last tip_length bp of the probe.
        tip = probe[-tip_length:]

        # ── Non-tip region of the probe (everything except the tip) ──────────
        non_tip_probe = probe[:-tip_length]

        # ── Define the safe zone (legitimate binding site) ─────────────────────
        safe_buffer = probe_len + 5
        safe_start  = max(0, global_start - safe_buffer)
        safe_end    = min(n, global_end   + safe_buffer)

        hits = []

        # ── Check only positions where the tip matches — using the index ──────
        # For each strand, look up only the positions where this oligo's tip
        # appears. For non-repetitive sequences this is ~10-20 positions total
        # instead of the full sequence length.
        strand_configs = [
            ('top strand',    top_index, full_seq, False),
            ('bottom strand', rc_index,  full_rc,  True),
        ]

        for strand_label, index, search_seq, is_rc in strand_configs:
            candidate_positions = index.get(tip, [])

            for i in candidate_positions:
                # i is the window start in search_seq for a window of length probe_len
                # (tip already matches at this position — that's why it's in the index)

                # Map to top-strand coordinates for safe-zone check
                if is_rc:
                    top_start = n - i - probe_len
                    top_end   = n - i
                else:
                    top_start = i
                    top_end   = i + probe_len

                # Skip the safe zone — this is the expected binding site
                if not (top_end <= safe_start or top_start >= safe_end):
                    continue

                # ── Count mismatches in the non-tip region ────────────────────
                # The tip already matches exactly (guaranteed by index lookup).
                # We only need to check the non-tip part.
                non_tip_window = search_seq[i : i + probe_len - tip_length]
                mismatches = sum(a != b for a, b in zip(non_tip_probe, non_tip_window))

                if mismatches <= max_mismatches:
                    hits.append((strand_label, top_start, top_end, mismatches))

        # ── Record result for this oligo ───────────────────────────────────────
        if hits:
            hit_descriptions = [
                f"{strand} pos {hs}–{he} ({mm} mismatch{'es' if mm != 1 else ''})"
                for strand, hs, he, mm in hits[:3]
            ]
            extra = f" (+{len(hits)-3} more)" if len(hits) > 3 else ""
            misprime_flags.append(True)
            misprime_warns.append(
                f"Potential misprime: 3' end ({probe_len} bp probe) matches "
                f"{len(hits)} unintended site(s) — "
                f"{', '.join(hit_descriptions)}{extra}. "
                f"This oligo may prime from an incorrect position during PCA, "
                f"producing chimeric products. Options: (1) adjust oligo boundaries "
                f"to shift the 3' end away from the matching region, "
                f"(2) increase misprime_max_mismatches to relax the check if these "
                f"hits are not biologically concerning, "
                f"(3) redesign the sequence to reduce repetitiveness."
            )
        else:
            misprime_flags.append(False)
            misprime_warns.append(None)

    # ── Update the DataFrame ──────────────────────────────────────────────────
    df = df.copy()
    df['flag_misprime'] = misprime_flags

    def _merge(existing, misprime_warn):
        if misprime_warn is None:
            return existing
        return misprime_warn if existing == 'OK' else existing + '; ' + misprime_warn

    df['warnings'] = [_merge(w, mw) for w, mw in zip(df['warnings'], misprime_warns)]

    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: FRAGMENTATION
# ──────────────────────────────────────────────────────────────────────────────

def _fragment_sequence(sequence: str,
                       frag_range: Tuple[int, int],
                       junction_overlap: int = 0) -> List[str]:
    """
    Divide the full sequence into fragments whose length falls within frag_range.

    Strategy: greedy chunking at the midpoint of the range.
    The last chunk will often be shorter than the minimum (unavoidable) — it
    is kept as-is and a warning is attached to its first oligo downstream.

    Junction overlap:
        When junction_overlap > 0, adjacent fragments share that many bp at
        their boundary. Fragment N ends junction_overlap bp past its cut point;
        Fragment N+1 starts at the same cut point — so the last junction_overlap
        bp of fragment N equal the first junction_overlap bp of fragment N+1.

        This means the independently assembled PCA products for adjacent fragments
        will share a complementary overlap region, which is exactly what you need
        for the downstream stitching PCR that joins all fragments into the full gene.

        Example with junction_overlap=25:
          cut point = 450
          Fragment 1 = sequence[0   : 475]   (last 25 bp are the junction)
          Fragment 2 = sequence[450 : 925]   (first 25 bp are the same junction)
          → assembled Fragment 1 and Fragment 2 share 25 bp → can be PCR-stitched.

    Args:
        sequence        : full input DNA sequence
        frag_range      : (min_bp, max_bp)
        junction_overlap: bp to share between adjacent fragments (default 0 = original behaviour)
    """
    min_f, max_f = frag_range
    target = (min_f + max_f) // 2   # aim for the midpoint of the range
    n = len(sequence)

    if n <= max_f:
        # Entire sequence fits in one fragment — no splitting needed
        return [sequence]

    # ── Compute clean cut points first (same as before) ──────────────────────
    cut_points = []
    pos = 0
    while pos < n:
        remaining = n - pos
        if remaining <= max_f:
            cut_points.append(pos)
            break
        cut_points.append(pos)
        pos += target

    # ── Build fragments with optional junction overlap ────────────────────────
    fragments = []
    for idx, start in enumerate(cut_points):
        if idx < len(cut_points) - 1:
            # Not the last fragment: extend end by junction_overlap bp
            raw_end = cut_points[idx + 1]
            end     = min(n, raw_end + junction_overlap)
        else:
            # Last fragment: take everything that remains
            end = n
        fragments.append(sequence[start:end])

    return fragments


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: RECORD BUILDERS
# These build the dict that becomes a row in the final DataFrame.
# ──────────────────────────────────────────────────────────────────────────────

def _make_record(
    frag_idx: int, oligo_idx: int, direction: str,
    frag_start: int, frag_end: int,
    oligo_seq: str, overlap_seq: Optional[str],
    overlap_tm: Optional[float], warnings: List[str], is_last: bool,
    # Quality flag columns — one per check, for easy DataFrame filtering
    tm_flag: bool = False,          # True if overlap Tm is outside the specified range
    gc_flag: bool = False,
    homopolymer_flag: bool = False,
    dinucleotide_flag: bool = False,
    hairpin_flag: bool = False,
) -> dict:
    """
    Build a single oligo record dictionary.

    The direction determines what sequence goes into 'oligo_sequence':
    - Forward oligos: top-strand sequence, read 5'→3' left to right
    - Reverse oligos: reverse complement of the top strand, also read 5'→3'
      This is what you order from IDT/Twist/etc. for the reverse oligo.
    """
    dir_tag = 'F' if direction == 'forward' else 'R'
    return {
        # ── Identity ─────────────────────────────────────────────────────────
        'oligo_name':          f"Frag{frag_idx+1:02d}_Oligo{oligo_idx+1:02d}_{dir_tag}",
        'fragment':            frag_idx + 1,
        'oligo_in_frag':       oligo_idx + 1,
        'direction':           direction,

        # ── Position on fragment ──────────────────────────────────────────────
        'frag_start':          frag_start,   # 0-based, start of this oligo
        'frag_end':            frag_end,     # exclusive end (Python slice style)

        # ── Oligo properties ─────────────────────────────────────────────────
        'oligo_length':        frag_end - frag_start,
        'oligo_gc_%':          gc_percent(oligo_seq),
        'oligo_sequence':      oligo_seq,    # ready to order as-is

        # ── Overlap properties (None for the last oligo in each fragment) ─────
        'overlap_sequence':    overlap_seq if overlap_seq else 'N/A',
        'overlap_length':      len(overlap_seq) if overlap_seq else None,
        'overlap_tm_C':        overlap_tm if not is_last else None,
        'overlap_gc_%':        gc_percent(overlap_seq) if overlap_seq else None,

        # ── Quality flags — True means a problem was detected ─────────────────
        # These columns let you quickly filter the DataFrame for problematic oligos:
        #   df[df['flag_tm_out_of_range']]     → overlaps whose Tm is outside the target range
        #   df[df['flag_overlap_gc']]          → overlaps with extreme GC content
        #   df[df['flag_homopolymer']]         → overlaps with AAAA/TTTT/etc.
        #   df[df['flag_dinucleotide_repeat']] → overlaps with ATAT/CGCG/etc.
        #   df[df['flag_hairpin']]             → oligos with stable hairpin (dG <= threshold)
        #   df[df['flag_misprime']]            → oligos whose 3' end may prime at wrong sites
        #   NOTE: flag_misprime is added by check_misprime_all() after tiling,
        #         so it is not present in the raw _make_record dict.
        #         It is injected into each record in design_pca_oligos Step 2b.
        'flag_tm_out_of_range':     tm_flag,
        'flag_overlap_gc':          gc_flag,
        'flag_homopolymer':         homopolymer_flag,
        'flag_dinucleotide_repeat': dinucleotide_flag,
        'flag_hairpin':             hairpin_flag,

        # ── Status ────────────────────────────────────────────────────────────
        'is_last_in_frag':     is_last,
        # All warnings for this oligo concatenated — 'OK' if none
        'warnings':            '; '.join(warnings) if warnings else 'OK',
    }


def _build_candidate(
    frag_idx: int, oligo_idx: int, direction: str,
    pos: int, oligo_end: int, fragment: str,
    overlap_seq: str, tm: float, warnings: List[str],
    tm_flag: bool, gc_flag: bool, homopolymer_flag: bool,
    dinucleotide_flag: bool, hairpin_flag: bool,
) -> dict:
    """
    Build a best-effort record when no combination fully satisfies all constraints.
    The flags and warnings document exactly which constraints were violated.
    """
    oligo_seq_top = fragment[pos: oligo_end]
    oligo_seq = oligo_seq_top if direction == 'forward' else reverse_complement(oligo_seq_top)
    return _make_record(
        frag_idx, oligo_idx, direction, pos, oligo_end,
        oligo_seq, overlap_seq, round(tm, 2), warnings, is_last=False,
        tm_flag=tm_flag, gc_flag=gc_flag, homopolymer_flag=homopolymer_flag,
        dinucleotide_flag=dinucleotide_flag, hairpin_flag=hairpin_flag,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3b: INTER-FRAGMENT JUNCTION DESIGN
#
# WHY THIS IS NEEDED:
#   PCA reliably assembles fragments up to ~500 bp per reaction. For longer
#   genes you split into multiple fragments, assemble each separately, then
#   stitch them together in a second PCR. For that stitching PCR to work,
#   adjacent fragments MUST share an overlap region — typically 20–30 bp —
#   so the assembled PCR products can anneal to each other and be extended.
#
#   Without this: you can have 10 perfectly designed fragment oligo sets but no
#   way to join the assembled fragments into the full gene. This function designs
#   and evaluates those junctions.
#
# HOW IT WORKS:
#   When junction_overlap > 0, _fragment_sequence produces overlapping fragments
#   so that the last junction_overlap bp of Fragment N = the first junction_overlap
#   bp of Fragment N+1. _design_junction_primers then:
#     1. Extracts that shared junction sequence for each consecutive pair
#     2. Computes its Tm, GC%, and quality flags (same criteria as overlap checks)
#     3. Outputs a forward stitching primer (top strand, ending at the junction point)
#        and a reverse stitching primer (RC, starting at the junction point)
#     4. Flags junctions that have poor Tm, bad GC, homopolymer runs, etc.
#
# OUTPUT:
#   A DataFrame with one row per junction (N-1 junctions for N fragments).
#   Also prints a junction summary alongside the oligo summary.
#   If output_csv is set, saves a separate "pca_junctions.csv".
# ──────────────────────────────────────────────────────────────────────────────

def _design_junction_primers(
    fragments: List[str],
    full_sequence: str,
    junction_overlap: int,
    tm_range: Tuple[float, float],
    junction_tm_range: Tuple[float, float],
    gc_range: Tuple[float, float],
    max_homopolymer: int,
    max_dinucleotide: int,
    hairpin_dg_threshold: float,
    mv_conc: float,
    dv_conc: float,
    dntp_conc: float,
    dna_conc: float,
) -> pd.DataFrame:
    """
    Design and evaluate stitching primers for the junctions between fragments.

    For each consecutive pair of fragments (N, N+1), extracts the shared
    junction_overlap bp sequence that connects them. This shared region is
    the overlap you need for stitching PCR after independent PCA assembly.

    Returns a DataFrame with one row per junction containing:
        junction_name        — e.g. "Junction_Frag01_Frag02"
        frag_left            — left fragment number (1-based)
        frag_right           — right fragment number (1-based)
        junction_start       — start position in the full sequence (0-based)
        junction_end         — end position (exclusive)
        junction_length      — length in bp
        junction_sequence    — the shared overlap sequence (top strand, 5'→3')
        fwd_stitch_primer    — forward stitching primer (= junction_sequence, same as ordering a F outer primer ending at the boundary)
        rev_stitch_primer    — reverse complement of junction_sequence (= what you order as the R outer primer for fragment N+1)
        junction_tm_C        — Tm of the junction overlap
        junction_gc_%        — GC% of the junction overlap
        flag_tm_out_of_range — junction Tm outside the overlap_tm_range
        flag_overlap_gc      — GC% outside gc_range
        flag_homopolymer     — homopolymer run in junction
        flag_dinucleotide_repeat — dinucleotide repeat in junction
        flag_hairpin         — hairpin in either stitching primer
        warnings             — plain-English description of all issues
    """
    if len(fragments) < 2 or junction_overlap == 0:
        return pd.DataFrame()   # no junctions for single fragment or no overlap

    records = []
    # Compute the global start of each fragment in the full sequence
    frag_global_starts = []
    cumulative = 0
    for frag in fragments:
        frag_global_starts.append(cumulative)
        cumulative += len(frag)

    for i in range(len(fragments) - 1):
        frag_left  = fragments[i]
        frag_right = fragments[i + 1]

        # The junction sequence is the last junction_overlap bp of fragment i.
        # Because we built overlapping fragments, this equals the first
        # junction_overlap bp of fragment i+1 — same sequence in the full gene.
        junc_seq   = frag_left[-junction_overlap:]
        junc_start = frag_global_starts[i] + len(frag_left) - junction_overlap
        junc_end   = junc_start + len(junc_seq)

        # Verify (sanity check): junction must match the same region in frag_right
        expected_in_right = frag_right[:junction_overlap]
        sequence_consistent = (junc_seq.upper() == expected_in_right.upper())

        # ── Quality checks on junction overlap ───────────────────────────────
        tm         = calc_tm(junc_seq, mv_conc, dv_conc, dntp_conc, dna_conc)
        tm_ok      = junction_tm_range[0] <= tm <= junction_tm_range[1]
        gc_warn    = check_overlap_gc(junc_seq, gc_range)
        hp_warn    = check_homopolymer(junc_seq, max_homopolymer)
        dn_warn    = check_dinucleotide_repeat(junc_seq, max_dinucleotide)

        # Hairpin check on each stitching primer
        fwd_primer = junc_seq
        rev_primer = reverse_complement(junc_seq)
        hp_fwd     = check_hairpin(fwd_primer, hairpin_dg_threshold)
        hp_rev     = check_hairpin(rev_primer, hairpin_dg_threshold)
        hp_primer_warn = hp_fwd + hp_rev

        all_warnings = []
        if not tm_ok:
            direction_word = 'BELOW minimum' if tm < junction_tm_range[0] else 'ABOVE maximum'
            bound = junction_tm_range[0] if tm < junction_tm_range[0] else junction_tm_range[1]
            all_warnings.append(
                f"Junction Tm {tm:.1f}C is {direction_word} {bound}C — "
                f"stitching PCR may not work at your intended annealing temperature"
            )
        all_warnings += gc_warn + hp_warn + dn_warn + hp_primer_warn

        if not sequence_consistent:
            all_warnings.append(
                "WARNING: junction sequence does not match between left and right fragment — "
                "this may indicate a fragmentation bug. Please report."
            )

        records.append({
            'junction_name':          f"Junction_Frag{i+1:02d}_Frag{i+2:02d}",
            'frag_left':              i + 1,
            'frag_right':             i + 2,
            'junction_start':         junc_start,
            'junction_end':           junc_end,
            'junction_length':        len(junc_seq),
            'junction_sequence':      junc_seq,
            'fwd_stitch_primer':      fwd_primer,
            'rev_stitch_primer':      rev_primer,
            'junction_tm_C':          round(tm, 2),
            'junction_gc_%':          gc_percent(junc_seq),
            'flag_tm_out_of_range':   not tm_ok,
            'flag_overlap_gc':        bool(gc_warn),
            'flag_homopolymer':       bool(hp_warn),
            'flag_dinucleotide_repeat': bool(dn_warn),
            'flag_hairpin':           bool(hp_primer_warn),
            'warnings':               '; '.join(all_warnings) if all_warnings else 'OK',
        })

    return pd.DataFrame(records)



#
# This is the core algorithm. For each fragment it places oligos one at a time
# from left to right, choosing the shortest (oligo_len, overlap_len) pair that
# satisfies ALL four quality checks simultaneously.
#
# "Greedy" means: as soon as a valid combination is found, it is accepted and
# we move on to the next oligo — we do not look ahead or backtrack to find a
# globally optimal solution. This is simpler and fast, and usually produces
# good results. A global optimizer could be added in a future version.
#
# PRIORITY ORDER when searching for a valid overlap:
#   1. Tm in range AND GC in range AND no homopolymer AND no dinucleotide repeat
#      → perfect, accept immediately
#   2. If no perfect option exists across all (oligo_len, ov_len) combinations,
#      use the combination closest to the target Tm midpoint and flag all
#      violated constraints in the warnings column.
#   3. If no combination at all was evaluated (extreme constraints), force a
#      fallback oligo with max length and min overlap.
#
# NOTE: The hairpin check is run on the full oligo AFTER the overlap is chosen,
# because hairpin depends on the whole oligo sequence, not just the overlap.
# ──────────────────────────────────────────────────────────────────────────────

def _tile_fragment(
    fragment: str, frag_idx: int,
    oligo_range: Tuple[int, int],
    overlap_range: Tuple[int, int],
    tm_range: Tuple[float, float],
    gc_range: Tuple[float, float],
    max_homopolymer: int,
    max_dinucleotide: int,
    vicinity_buffer: int,       # NEW: extra bp scanned around overlap for repeat proximity
    hairpin_dg_threshold: float, # NEW: replaces min_hairpin_length — uses primer3 dG
    mv_conc: float, dv_conc: float, dntp_conc: float, dna_conc: float,
    frag_len_range: Optional[Tuple[int, int]],
) -> List[dict]:
    """
    Tile one PCA fragment into alternating forward / reverse oligos.

    PCA tiling structure (schematic):

      Fragment:  5'═══════════════════════════════════════3'
                                                          (top strand)

      Oligo 1 F: 5'══════════════╗
                                 ║ ← this overlapping region has
      Oligo 2 R:        5'═══════╩══════════╗     the overlap Tm we optimize
                                             ║
      Oligo 3 F:                5'═══════════╩══════════╗
                                                         ║
      Oligo 4 R:                        5'═══════════════╩════...

    In a PCA reaction tube, all oligos are mixed together. During the
    annealing step, oligo 1 and oligo 2 hybridize at their shared 3' ends.
    The polymerase then extends both, filling in the gap. After many cycles,
    the full fragment is assembled from overlapping oligo pairs.
    """
    min_ol, max_ol   = oligo_range
    min_ov, max_ov   = overlap_range
    min_tm, max_tm   = tm_range
    target_tm        = (min_tm + max_tm) / 2.0
    frag_len         = len(fragment)
    # vicinity_buffer and hairpin_dg_threshold are passed in as parameters —
    # they are used in the inner loop and in check_hairpin / check_repeat_vicinity calls

    # ── Fragment-level size warning ───────────────────────────────────────────
    # The last fragment is often shorter than the specified minimum — this is
    # expected and unavoidable. We attach this note to the first oligo of the
    # affected fragment so it's visible in the output.
    frag_warnings = []
    if frag_len_range:
        if frag_len < frag_len_range[0]:
            frag_warnings.append(
                f"Fragment length {frag_len} bp is below minimum "
                f"{frag_len_range[0]} bp (last fragment is commonly shorter)"
            )
        elif frag_len > frag_len_range[1]:
            frag_warnings.append(
                f"Fragment length {frag_len} bp is above maximum {frag_len_range[1]} bp"
            )

    oligos    = []
    pos       = 0     # current position (start of the oligo being placed)
    oligo_idx = 0     # 0-based oligo counter within this fragment
    safety    = 0     # counts loop iterations as a hard infinite-loop guard

    while pos < frag_len:
        safety += 1
        if safety > 10_000:
            # Should never be reached with valid constraints, but kept as
            # an absolute safety net against unforeseen edge cases.
            break

        # Alternate between forward and reverse oligos.
        # Even indices (0, 2, 4, ...) → forward (top strand)
        # Odd indices  (1, 3, 5, ...) → reverse (bottom strand, rev-comped)
        direction    = 'forward' if oligo_idx % 2 == 0 else 'reverse'
        remaining    = frag_len - pos

        # Carry fragment-level warnings to the first oligo only
        cur_warnings      = list(frag_warnings)
        frag_warnings     = []   # clear so subsequent oligos don't repeat them

        # ── Last oligo: consume everything that remains ───────────────────────
        # When what's left is short enough to fit in one oligo, we take it all.
        # The last oligo has no downstream neighbour, so it has no overlap to
        # optimize — no Tm or GC checks apply here.
        if remaining <= max_ol:
            oligo_seq_top = fragment[pos:]
            oligo_seq = (oligo_seq_top if direction == 'forward'
                         else reverse_complement(oligo_seq_top))
            if len(oligo_seq_top) < min_ol:
                cur_warnings.append(
                    f"Last oligo length {len(oligo_seq_top)} bp is below "
                    f"minimum {min_ol} bp — consider adjusting fragment or "
                    f"oligo length range"
                )
            # Run hairpin check on the last oligo too.
            # Uses primer3.calc_hairpin() via check_hairpin() — dG-based, not heuristic.
            hp_warn = check_hairpin(oligo_seq, hairpin_dg_threshold)
            oligos.append(_make_record(
                frag_idx, oligo_idx, direction,
                pos, pos + len(oligo_seq_top),
                oligo_seq, None, None,
                cur_warnings + hp_warn, is_last=True,
                tm_flag=False,   # last oligo has no overlap, so Tm is not applicable
                hairpin_flag=bool(hp_warn),
            ))
            break

        # ── Search for valid (oligo_len, overlap_len) combination ─────────────
        # We iterate over all possible oligo lengths (shortest first to keep
        # oligos as short as possible), and for each, all possible overlap
        # lengths. We accept the first combination that passes ALL checks.
        placed        = False
        best_record   = None        # best fallback if no perfect option found
        best_tm_delta = float('inf')  # how far from target Tm the best fallback is

        for oligo_len in range(min_ol, min(max_ol, remaining - 1) + 1):
            oligo_end   = pos + oligo_len
            # The overlap cannot be longer than the oligo itself — that's
            # geometrically impossible. Cap max_ov at oligo_len - 1.
            max_ov_here = min(max_ov, oligo_len - 1)

            for ov_len in range(min_ov, max_ov_here + 1):
                # The overlap is the 3' end of the current oligo.
                # It is the region shared with the next oligo (which starts
                # where this overlap begins).
                overlap_seq = fragment[oligo_end - ov_len: oligo_end]

                # ── Run all quality checks on this overlap ────────────────────
                tm      = calc_tm(overlap_seq, mv_conc, dv_conc, dntp_conc, dna_conc)
                tm_ok   = min_tm <= tm <= max_tm

                # Direct checks: is the bad pattern INSIDE the overlap?
                gc_warn = check_overlap_gc(overlap_seq, gc_range)
                hp_warn = check_homopolymer(overlap_seq, max_homopolymer)
                dn_warn = check_dinucleotide_repeat(overlap_seq, max_dinucleotide)

                gc_ok = len(gc_warn) == 0
                hp_ok = len(hp_warn) == 0
                dn_ok = len(dn_warn) == 0

                # Vicinity check: is a bad pattern ADJACENT to the overlap boundary?
                # This catches the edge-clipping problem (Test C finding): an overlap
                # that starts right at the tail of a homopolymer run won't be caught
                # by the direct check alone, but is caught by scanning the buffer zone.
                # overlap_start = oligo_end - ov_len (position on the fragment)
                overlap_start_pos = oligo_end - ov_len
                vicinity_warn = check_repeat_vicinity(
                    fragment, overlap_start_pos, oligo_end,
                    vicinity_buffer, max_homopolymer, max_dinucleotide,
                )
                # Treat proximity warnings as soft failures: they don't block the
                # overlap from being chosen but they are bundled into the warning list
                # and will cause the oligo to be flagged. We only hard-reject if
                # BOTH the direct check AND the vicinity check fail — i.e. the
                # overlap is inside a bad region AND adjacent to one.
                # If only the vicinity fires, we accept the overlap but flag it.
                vic_ok = len(vicinity_warn) == 0

                # ── Perfect combination: all direct checks pass ───────────────
                # Note: vicinity_warn (soft) does NOT block placement — the overlap
                # is accepted but proximity warnings are recorded for the user.
                if tm_ok and gc_ok and hp_ok and dn_ok:
                    oligo_seq_top = fragment[pos: oligo_end]
                    oligo_seq = (oligo_seq_top if direction == 'forward'
                                 else reverse_complement(oligo_seq_top))

                    # Run hairpin check on the complete oligo now that we know
                    # its boundaries. Uses primer3.calc_hairpin() via check_hairpin()
                    # for a real thermodynamic dG value — not the old heuristic.
                    full_hp_warn = check_hairpin(oligo_seq, hairpin_dg_threshold)

                    # Combine: direct-check warnings (none here since all passed),
                    # vicinity proximity warnings (soft), and hairpin warning (if any)
                    all_warn = cur_warnings + vicinity_warn + full_hp_warn

                    oligos.append(_make_record(
                        frag_idx, oligo_idx, direction,
                        pos, oligo_end,
                        oligo_seq, overlap_seq, round(tm, 2),
                        all_warn, is_last=False,
                        tm_flag=False,   # Tm passed — this is why we're in the perfect branch
                        hairpin_flag=bool(full_hp_warn),
                    ))
                    # Advance pos to where the next oligo starts.
                    # The next oligo begins at the start of the current overlap —
                    # that shared region is where the two oligos hybridize.
                    pos = oligo_end - ov_len
                    oligo_idx += 1
                    placed = True
                    break

                # ── Imperfect combination: track the best fallback ────────────
                # "Best" = closest to the target Tm midpoint, so that even
                # when we fall back, we're as close to the user's intent as possible.
                delta = abs(tm - target_tm)
                if delta < best_tm_delta:
                    best_tm_delta = delta
                    # Collect all warnings for this imperfect candidate
                    w = list(cur_warnings)
                    if not tm_ok:
                        direction_word = 'BELOW minimum' if tm < min_tm else 'ABOVE maximum'
                        bound = min_tm if tm < min_tm else max_tm
                        advice = ('overlap may not anneal reliably' if tm < min_tm
                                  else 'overlap may be too stable / cause mispriming')
                        w.append(
                            f"Overlap Tm {tm:.1f}C is {direction_word} {bound}C "
                            f"— {advice}"
                        )
                    w += gc_warn + hp_warn + dn_warn

                    # Also check hairpin on this candidate's full oligo.
                    # dG-based via primer3.calc_hairpin().
                    oligo_seq_top = fragment[pos: oligo_end]
                    oligo_seq_cand = (oligo_seq_top if direction == 'forward'
                                     else reverse_complement(oligo_seq_top))
                    full_hp_warn_cand = check_hairpin(oligo_seq_cand, hairpin_dg_threshold)
                    # Include vicinity warnings in the fallback record too
                    w += vicinity_warn + full_hp_warn_cand

                    best_record = _build_candidate(
                        frag_idx, oligo_idx, direction,
                        pos, oligo_end, fragment,
                        overlap_seq, tm, w,
                        tm_flag=not tm_ok,        # True if Tm was outside the target range
                        gc_flag=not gc_ok,
                        homopolymer_flag=not hp_ok,
                        dinucleotide_flag=not dn_ok,
                        hairpin_flag=bool(full_hp_warn_cand),
                    )

            if placed:
                break

        # ── No valid combination found: use best fallback ─────────────────────
        if not placed:
            if best_record is None:
                # Absolute last resort — no (oligo_len, ov_len) pair was even
                # evaluated (can happen if remaining is very small).
                # Force max oligo length and min overlap and document it.
                oligo_len   = min(max_ol, remaining - 1)
                oligo_end   = pos + oligo_len
                ov_len      = max(min_ov, 1)
                overlap_seq = fragment[oligo_end - ov_len: oligo_end]
                tm = calc_tm(overlap_seq, mv_conc, dv_conc, dntp_conc, dna_conc)
                forced_w = cur_warnings + [
                    f"No valid oligo/overlap combination found within constraints. "
                    f"Forced oligo={oligo_len} bp, overlap={ov_len} bp, Tm={tm:.1f}C. "
                    f"All four quality checks may have failed. "
                    f"Review your constraints — they may be too strict for this region."
                ]
                oligo_seq_top  = fragment[pos: oligo_end]
                oligo_seq_f    = (oligo_seq_top if direction == 'forward'
                                  else reverse_complement(oligo_seq_top))
                hp_w = check_hairpin(oligo_seq_f, hairpin_dg_threshold)
                best_record = _build_candidate(
                    frag_idx, oligo_idx, direction,
                    pos, oligo_end, fragment, overlap_seq, tm,
                    forced_w + hp_w,
                    tm_flag=True,           # forced fallback: assume Tm violated too
                    gc_flag=True, homopolymer_flag=True,
                    dinucleotide_flag=True, hairpin_flag=bool(hp_w),
                )

            oligos.append(best_record)

            # CRITICAL: pos must always move forward.
            # Without this guard, a fallback with a large overlap relative to
            # a short oligo can make pos stay the same or go backwards,
            # causing an infinite loop. max(candidate, pos+1) guarantees progress.
            pos_candidate = best_record['frag_end'] - (best_record['overlap_length'] or min_ov)
            pos = max(pos_candidate, pos + 1)
            oligo_idx += 1

    return oligos


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: MAIN PUBLIC FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def design_pca_oligos(
    sequence: str,
    # ── Structural parameters ─────────────────────────────────────────────────
    fragment_length_range:   Optional[Tuple[int, int]]   = DEFAULT_FRAGMENT_LENGTH_RANGE,
    oligo_length_range:      Tuple[int, int]             = DEFAULT_OLIGO_LENGTH_RANGE,
    overlap_length_range:    Tuple[int, int]             = DEFAULT_OVERLAP_LENGTH_RANGE,
    # ── Quality thresholds ────────────────────────────────────────────────────
    overlap_tm_range:        Tuple[float, float]         = DEFAULT_OVERLAP_TM_RANGE,
    overlap_gc_range:        Tuple[float, float]         = DEFAULT_OVERLAP_GC_RANGE,
    max_homopolymer_run:     int                         = DEFAULT_MAX_HOMOPOLYMER_RUN,
    max_dinucleotide_repeat: int                         = DEFAULT_MAX_DINUCLEOTIDE_REPEAT,
    repeat_vicinity_buffer:  int                         = DEFAULT_REPEAT_VICINITY_BUFFER,
    hairpin_dg_threshold:    float                       = DEFAULT_HAIRPIN_DG_THRESHOLD,
    # ── Misprime detection ────────────────────────────────────────────────────
    misprime_length:         int                         = DEFAULT_MISPRIME_LENGTH,
    misprime_tip_length:     int                         = DEFAULT_MISPRIME_TIP_LENGTH,
    misprime_max_mismatches: int                         = DEFAULT_MISPRIME_MAX_MISMATCHES,
    # ── Thermodynamic reaction conditions ────────────────────────────────────
    mv_conc:                 float                       = DEFAULT_MV_CONC,
    dv_conc:                 float                       = DEFAULT_DV_CONC,
    dntp_conc:               float                       = DEFAULT_DNTP_CONC,
    dna_conc:                float                       = DEFAULT_DNA_CONC,
    # ── Output ────────────────────────────────────────────────────────────────
    output_csv:              Optional[str]               = DEFAULT_OUTPUT_CSV,
    # ── Inter-fragment junction ───────────────────────────────────────────────
    junction_overlap_length: int                         = DEFAULT_JUNCTION_OVERLAP_LENGTH,
    junction_tm_range:       Tuple[float, float]         = DEFAULT_JUNCTION_TM_RANGE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Design PCA oligos for a given DNA sequence.

    Parameters
    ----------
    sequence : str
        Input DNA sequence, 5'→3', top strand. Upper or lowercase accepted.
        Spaces and newlines are stripped. Only A, T, G, C are supported.

    fragment_length_range : tuple (min_bp, max_bp) or None
        The input sequence is first split into fragments of this length.
        Each fragment is an independent PCA reaction in the lab.
        Set to None to treat the entire sequence as one fragment (no splitting).
        Default: (400, 500)

    oligo_length_range : tuple (min_bp, max_bp)
        Length range for each individual oligo. Shorter oligos are tried first
        to keep synthesis cost down. Default: (40, 60)

    overlap_length_range : tuple (min_bp, max_bp)
        Length of the overlap region shared between consecutive oligos.
        This is the region that hybridizes during PCA. Default: (15, 25)

    overlap_tm_range : tuple (min_C, max_C)
        Melting temperature range for the overlap region (SantaLucia 1998).
        All oligos in a PCA reaction should have similar Tm values so they
        anneal at the same cycling temperature. Default: (55, 65)

    overlap_gc_range : tuple (min_%, max_%)
        GC content bounds for the overlap region. Overlaps outside this range
        are flagged — extreme GC can cause secondary structures or mispriming.
        Default: (40.0, 70.0)

    max_homopolymer_run : int
        Runs of the same nucleotide >= this length in an overlap are flagged.
        These cause polymerase slippage and unreliable Tm calculation.
        Default: 4

    max_dinucleotide_repeat : int
        Dinucleotide motifs (e.g. AT, CG) repeated >= this many times
        consecutively in an overlap are flagged. Default: 3

    repeat_vicinity_buffer : int
        Number of bp to scan on each side of the overlap region when checking
        for homopolymer runs and dinucleotide repeats. This catches the
        edge-clipping problem: a bad region right at the boundary of an overlap
        that doesn't fully fall inside it. Default: 5

    hairpin_dg_threshold : float
        Free energy threshold (kcal/mol) for hairpin flagging, using
        primer3.calc_hairpin(). If an oligo's most stable hairpin has
        dG <= this value, it is flagged. More negative = only flag stronger
        hairpins. Default: -4.0 kcal/mol.
        Examples:
          -1.0 → flag even weak hairpins (very conservative)
          -2.0 → flag moderate+ hairpins (was previous default)
          -4.0 → only flag strong hairpins (current default — practical threshold)

    misprime_length : int
        Number of bp from each oligo's 3' end used as the search probe when
        scanning for unintended binding sites. Longer = more specific, fewer
        false positives. Default: 18 (matching DNAWorks)

    misprime_tip_length : int
        Number of bp at the very 3' tip of the probe that must match EXACTLY
        at any candidate misprime site. The 3' tip is critical — polymerases
        require a perfectly matched 3' end to initiate extension. Default: 6

    misprime_max_mismatches : int
        Maximum mismatches allowed in the non-tip region of the probe for a
        site to be considered a misprime candidate. Lower = stricter (fewer
        flags), higher = more sensitive (more flags). Default: 8

    mv_conc   : float  Monovalent salt [Na+/K+] in mM. Default: 50
    dv_conc   : float  Divalent salt [Mg2+] in mM.     Default: 1.5
    dntp_conc : float  dNTP concentration in mM.        Default: 0.2
    dna_conc  : float  Oligo strand conc. in nM.        Default: 250

    output_csv : str or None
        File path to save the results as a CSV. None = skip. Default: "pca_oligos.csv"

    Returns
    -------
    pd.DataFrame — one row per oligo with the following columns:

        IDENTITY:
          global_oligo_num       — sequential number across all fragments (1-based)
          oligo_name             — unique name: Frag##_Oligo##_F or _R
          fragment               — which fragment (1-based)
          oligo_in_frag          — oligo position within its fragment (1-based)
          direction              — 'forward' or 'reverse'

        POSITION:
          frag_start             — start position within the fragment (0-based)
          frag_end               — end position (exclusive, Python-style)

        OLIGO PROPERTIES:
          oligo_length           — total oligo length in bp
          oligo_gc_%             — GC% of the full oligo
          oligo_sequence         — the sequence to order (rev-comped for reverse oligos)

        OVERLAP PROPERTIES (None for the last oligo in each fragment):
          overlap_sequence       — overlap region on the top strand
          overlap_length         — overlap length in bp
          overlap_tm_C           — overlap Tm in °C
          overlap_gc_%           — GC% of the overlap

        QUALITY FLAGS (True = problem detected, False = OK):
          flag_tm_out_of_range   — overlap Tm outside the specified range
          flag_overlap_gc        — overlap GC% outside specified range
          flag_homopolymer       — homopolymer run detected in overlap
          flag_dinucleotide_repeat — dinucleotide repeat detected in overlap
          flag_hairpin           — potential hairpin in the full oligo
          flag_misprime          — 3' end matches unintended site(s) in the sequence

        STATUS:
          is_last_in_frag        — True for the terminal oligo of each fragment
          warnings               — 'OK' or all warning messages concatenated
    """

    # ── Input cleaning ────────────────────────────────────────────────────────
    sequence = sequence.upper().strip().replace('\n', '').replace(' ', '')

    # ── Basic input validation ────────────────────────────────────────────────
    invalid = set(sequence) - set('ACGT')
    if invalid:
        raise ValueError(
            f"Sequence contains invalid characters: {invalid}. "
            f"Only A, T, G, C are supported (no IUPAC ambiguity codes)."
        )
    if len(sequence) < oligo_length_range[0]:
        raise ValueError(
            f"Sequence length ({len(sequence)} bp) is shorter than the minimum "
            f"oligo length ({oligo_length_range[0]} bp). Nothing to tile."
        )

    # ── Cross-parameter logic checks ──────────────────────────────────────────
    # These catch contradictory constraint combinations before any computation,
    # preventing silent failures or the infinite-loop bug we fixed earlier.
    min_ol, max_ol = oligo_length_range
    min_ov, max_ov = overlap_length_range

    if min_ol < 1:
        raise ValueError(f"oligo_length_range min ({min_ol}) must be >= 1 bp.")
    if min_ol > max_ol:
        raise ValueError(f"oligo_length_range: min ({min_ol}) must be <= max ({max_ol}).")
    if min_ov < 1:
        raise ValueError(f"overlap_length_range min ({min_ov}) must be >= 1 bp.")
    if min_ov > max_ov:
        raise ValueError(f"overlap_length_range: min ({min_ov}) must be <= max ({max_ov}).")
    if min_ov >= min_ol:
        raise ValueError(
            f"Impossible: minimum overlap ({min_ov} bp) must be strictly less "
            f"than minimum oligo length ({min_ol} bp). "
            f"The overlap is a sub-region of the oligo and cannot be >= the oligo itself."
        )
    if overlap_tm_range[0] > overlap_tm_range[1]:
        raise ValueError(
            f"overlap_tm_range: min ({overlap_tm_range[0]}C) must be <= max ({overlap_tm_range[1]}C)."
        )
    if overlap_gc_range[0] > overlap_gc_range[1]:
        raise ValueError(
            f"overlap_gc_range: min ({overlap_gc_range[0]}%) must be <= max ({overlap_gc_range[1]}%)."
        )
    if fragment_length_range:
        if fragment_length_range[0] > fragment_length_range[1]:
            raise ValueError("fragment_length_range: min must be <= max.")
        if fragment_length_range[0] <= max_ov:
            raise ValueError(
                f"Impossible: minimum fragment length ({fragment_length_range[0]} bp) "
                f"must be greater than maximum overlap ({max_ov} bp). "
                f"A fragment cannot be shorter than its overlaps."
            )

    # ── Step 1: Fragment the sequence ─────────────────────────────────────────
    fragments = (
        [sequence] if fragment_length_range is None
        else _fragment_sequence(sequence, fragment_length_range,
                                junction_overlap=junction_overlap_length)
    )

    # ── Step 2: Tile each fragment independently ──────────────────────────────
    all_records, global_num = [], 1

    for frag_idx, fragment in enumerate(fragments):
        records = _tile_fragment(
            fragment=fragment, frag_idx=frag_idx,
            oligo_range=oligo_length_range,
            overlap_range=overlap_length_range,
            tm_range=overlap_tm_range,
            gc_range=overlap_gc_range,
            max_homopolymer=max_homopolymer_run,
            max_dinucleotide=max_dinucleotide_repeat,
            vicinity_buffer=repeat_vicinity_buffer,
            hairpin_dg_threshold=hairpin_dg_threshold,
            mv_conc=mv_conc, dv_conc=dv_conc,
            dntp_conc=dntp_conc, dna_conc=dna_conc,
            frag_len_range=fragment_length_range,
        )
        for r in records:
            r['global_oligo_num'] = global_num
            global_num += 1
        all_records.extend(records)

    # ── Step 2b: Misprime check (post-processing across all oligos) ──────────
    # This must run after tiling because it compares every oligo's 3' end
    # against the entire input sequence — it needs all oligo positions to
    # be known before it can define the "safe zone" for each one.
    # We pass the raw records list as a temporary DataFrame, run the check,
    # then assemble the final DataFrame from the updated version below.
    _pre_df = pd.DataFrame(all_records)
    _pre_df = check_misprime_all(
        df=_pre_df,
        fragments=fragments,
        full_sequence=sequence,
        misprime_length=misprime_length,
        tip_length=misprime_tip_length,
        max_mismatches=misprime_max_mismatches,
    )
    # Write updated warnings and new flag back to the records list so
    # col_order assembly below picks them up correctly.
    for i, rec in enumerate(all_records):
        rec['flag_misprime'] = _pre_df['flag_misprime'].iloc[i]
        rec['warnings']      = _pre_df['warnings'].iloc[i]

    # ── Step 3: Assemble the DataFrame ────────────────────────────────────────
    col_order = [
        'global_oligo_num', 'oligo_name', 'fragment', 'oligo_in_frag',
        'direction', 'frag_start', 'frag_end', 'oligo_length', 'oligo_gc_%',
        'oligo_sequence', 'overlap_sequence', 'overlap_length',
        'overlap_tm_C', 'overlap_gc_%',
        'flag_tm_out_of_range', 'flag_overlap_gc', 'flag_homopolymer',
        'flag_dinucleotide_repeat', 'flag_hairpin', 'flag_misprime',
        'is_last_in_frag', 'warnings',
    ]
    df = pd.DataFrame(all_records)[col_order]

    # ── Step 4b: Design inter-fragment junction primers ───────────────────────
    # Only runs when there are multiple fragments AND junction_overlap_length > 0.
    # Produces a second DataFrame describing the stitching overlap for each
    # consecutive fragment pair — the sequences you need for the downstream
    # stitching PCR that joins the independently assembled PCA products.
    junction_df = _design_junction_primers(
        fragments=fragments,
        full_sequence=sequence,
        junction_overlap=junction_overlap_length,
        tm_range=overlap_tm_range,
        junction_tm_range=junction_tm_range,
        gc_range=overlap_gc_range,
        max_homopolymer=max_homopolymer_run,
        max_dinucleotide=max_dinucleotide_repeat,
        hairpin_dg_threshold=hairpin_dg_threshold,
        mv_conc=mv_conc, dv_conc=dv_conc,
        dntp_conc=dntp_conc, dna_conc=dna_conc,
    )

    # ── Step 5: Print summary ─────────────────────────────────────────────────
    n_warnings      = (df['warnings'] != 'OK').sum()
    valid_tms       = df['overlap_tm_C'].dropna()
    n_gc_flags      = df['flag_overlap_gc'].sum()
    n_hp_flags      = df['flag_homopolymer'].sum()
    n_dn_flags      = df['flag_dinucleotide_repeat'].sum()
    n_hairpin_flags = df['flag_hairpin'].sum()
    n_tm_flags      = df['flag_tm_out_of_range'].sum()
    n_misprime_flags = df['flag_misprime'].sum()

    print("\n" + "=" * 60)
    print("  PCA OLIGO DESIGN SUMMARY")
    print("=" * 60)
    print(f"  Input sequence        : {len(sequence)} bp")
    print(f"  Fragments             : {len(fragments)}")
    print(f"  Fragment lengths      : {[len(f) for f in fragments]} bp")
    if len(fragments) > 1 and junction_overlap_length > 0:
        print(f"  Junction overlap      : {junction_overlap_length} bp (shared between adjacent fragments)")
    print(f"  Total oligos          : {len(df)}")
    if len(valid_tms):
        print(f"  Overlap Tm range      : {valid_tms.min():.1f}C -- {valid_tms.max():.1f}C")
        print(f"  Overlap Tm mean       : {valid_tms.mean():.1f}C")
    print(f"  --- Oligo quality flags ---")
    print(f"  Tm out of range       : {n_tm_flags} oligo(s)")
    print(f"  Overlap GC out of range : {n_gc_flags} oligo(s)")
    print(f"  Homopolymer runs      : {n_hp_flags} oligo(s)")
    print(f"  Dinucleotide repeats  : {n_dn_flags} oligo(s)")
    print(f"  Potential hairpins    : {n_hairpin_flags} oligo(s)")
    print(f"  Potential misprimes   : {n_misprime_flags} oligo(s)")
    print(f"  Total with warnings   : {n_warnings} oligo(s)")
    if n_warnings == 0:
        print(f"  Status                : OK -- all oligos passed all checks")
    else:
        print(f"  Status                : See 'warnings' column for details")

    # ── Junction summary ──────────────────────────────────────────────────────
    if not junction_df.empty:
        n_junc_warnings = (junction_df['warnings'] != 'OK').sum()
        print(f"\n  --- Inter-fragment junction summary ---")
        print(f"  Junctions designed    : {len(junction_df)}")
        for _, jrow in junction_df.iterrows():
            status = 'OK' if jrow['warnings'] == 'OK' else 'FLAGGED'
            print(f"  {jrow['junction_name']:30s}  Tm={jrow['junction_tm_C']:.1f}C  "
                  f"GC={jrow['junction_gc_%']:.1f}%  [{status}]")
        if n_junc_warnings > 0:
            print(f"  Junction warnings     : {n_junc_warnings} — see junction CSV for details")
        else:
            print(f"  Junction status       : OK -- all junctions passed all checks")

    print("=" * 60)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"  Oligos CSV saved to   : {output_csv}")
        if not junction_df.empty:
            junc_csv = output_csv.replace('.csv', '_junctions.csv')
            if not junc_csv.endswith('_junctions.csv'):
                junc_csv = output_csv + '_junctions.csv'
            junction_df.to_csv(junc_csv, index=False)
            print(f"  Junctions CSV saved to: {junc_csv}")
        print("=" * 60)

    print()
    return df, junction_df


if __name__ == '__main__':
    import random

    # ── Test 1: Clean baseline — should have zero misprime flags ─────────────
    print("=" * 60)
    print("TEST 1: 800 bp truly random sequence (seed=99)")
    print("        Expect: zero or very few misprime flags")
    print("=" * 60)
    random.seed(99)
    clean_seq = ''.join(random.choices('ACGT', k=800))
    df1, jdf1 = design_pca_oligos(sequence=clean_seq, output_csv="test1_clean.csv")
    print(df1[['oligo_name', 'oligo_length', 'overlap_tm_C',
               'flag_misprime', 'warnings']].to_string())

    # ── Test 2: Repetitive sequence — maximum misprime stress test ───────────
    print("\n" + "=" * 60)
    print("TEST 2: Highly repetitive sequence (ATCGATCG x50 = 400 bp)")
    print("        Expect: misprime flags on most/all oligos")
    print("=" * 60)
    repeat_seq = "ATCGATCG" * 50
    df2, jdf2 = design_pca_oligos(
        sequence=repeat_seq,
        fragment_length_range=None,
        output_csv="test2_repetitive.csv",
    )
    print(df2[['oligo_name', 'frag_start', 'frag_end',
               'flag_misprime', 'warnings']].to_string())
    print(f"\nMisprime flags: {df2['flag_misprime'].sum()} / {len(df2)}")

    # ── Test 3: One injected repeat in a clean background ────────────────────
    print("\n" + "=" * 60)
    print("TEST 3: Clean background + one duplicated 40 bp segment")
    print("        Expect: misprime flags ONLY near the duplicated region")
    print("=" * 60)
    random.seed(99)
    clean_a   = ''.join(random.choices('ACGT', k=150))
    dup_block = ''.join(random.choices('ACGT', k=40))
    clean_mid = ''.join(random.choices('ACGT', k=100))
    one_dup_seq = clean_a + dup_block + clean_mid + dup_block + ''.join(random.choices('ACGT', k=150))
    df3, jdf3 = design_pca_oligos(
        sequence=one_dup_seq,
        fragment_length_range=None,
        output_csv="test3_one_repeat.csv",
    )
    print(df3[['oligo_name', 'frag_start', 'frag_end',
               'flag_misprime', 'warnings']].to_string())
    dup_start  = len(clean_a)
    dup_end    = dup_start + len(dup_block)
    dup2_start = dup_end + len(clean_mid)
    dup2_end   = dup2_start + len(dup_block)
    print(f"\nDuplicated block 1: positions {dup_start}–{dup_end}")
    print(f"Duplicated block 2: positions {dup2_start}–{dup2_end}")
    print(f"Misprime flags: {df3['flag_misprime'].sum()} / {len(df3)}")

    # ── Test 4: Tighter misprime settings ────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4: Same sequence as Test 3, tighter misprime settings")
    print("        tip_length=8, max_mismatches=4")
    print("        Expect: fewer flags than Test 3 (stricter = fewer positives)")
    print("=" * 60)
    df4, jdf4 = design_pca_oligos(
        sequence=one_dup_seq,
        fragment_length_range=None,
        misprime_tip_length=8,
        misprime_max_mismatches=4,
        output_csv="test4_strict_misprime.csv",
    )
    print(df4[['oligo_name', 'frag_start', 'frag_end',
               'flag_misprime', 'warnings']].to_string())
    print(f"\nTest 3 misprime flags: {df3['flag_misprime'].sum()} / {len(df3)}")
    print(f"Test 4 misprime flags: {df4['flag_misprime'].sum()} / {len(df4)}")
    print(f"Delta: {df3['flag_misprime'].sum() - df4['flag_misprime'].sum()} flags removed by stricter settings")

    # ── Test 5: Real sequence — E. coli lacZ first 378 bp ────────────────────
    print("\n" + "=" * 60)
    print("TEST 5: First 378 bp of E. coli lacZ (real coding sequence)")
    print("        Expect: clean design with few or zero misprime flags")
    print("=" * 60)
    lacz = (
        "ATGACCATGATTACGCCAAGCTATTTTAGCGAAACGCTTAAAAAATGGCGATTTTCCGT"
        "GCGATGAATCATGAAAATGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCTGGCGT"
        "TACCCAACTTAATCGCCTTGCAGCACATCCCCCTTTCGCCAGCTGGCGTAATAGCGAAG"
        "AGGCCCGCACCGATCGCCCTTCCCAACAGTTGCGCAGCCTGAATGGCGAATGGCGCCTG"
        "ATGCGGTATTTTCTCCTTACGCATCTGTGCGGTATTTCACACCGCATATGGTGCACTCT"
        "CAGTACAATCTGCTCTGATGCCGCATAGTTAAGCCAGTATACACTCCGCTATCGCTACG"
        "TGACGGGGCTCGCTTGGGTGGCACCCTCGTCGACCACC"
    )
    df5, jdf5 = design_pca_oligos(
        sequence=lacz,
        fragment_length_range=None,
        output_csv="test5_lacz.csv",
    )
    print(df5[['oligo_name', 'direction', 'oligo_length', 'overlap_tm_C',
               'flag_tm_out_of_range', 'flag_overlap_gc', 'flag_homopolymer',
               'flag_dinucleotide_repeat', 'flag_hairpin',
               'flag_misprime', 'warnings']].to_string())

    # Coverage sanity check
    print("\n--- Coverage check (all positions covered?) ---")
    covered = [False] * len(lacz)
    for _, row in df5.iterrows():
        for i in range(int(row['frag_start']), int(row['frag_end'])):
            covered[i] = True
    gaps = [i for i, c in enumerate(covered) if not c]
    print(f"  Uncovered positions: {len(gaps)} (expected 0)")
    if gaps:
        print(f"  Gap positions: {gaps}")
    valid_tms = df5['overlap_tm_C'].dropna()
    print(f"  Tm std dev: {valid_tms.std():.2f}C (< 2.0C is good, < 1.0C is excellent)")
    print(f"  Misprime flags: {df5['flag_misprime'].sum()} / {len(df5)}")

    # ── Test 6: Multi-fragment gene — junction design validation ─────────────
    # A 1200 bp random gene that requires 3 fragments.
    # Expect: 2 junctions designed, each with a good Tm and GC%.
    # The last junction_overlap bp of fragment N should equal the
    # first junction_overlap bp of fragment N+1 exactly.
    print("\n" + "=" * 60)
    print("TEST 6: 1200 bp random sequence — multi-fragment with junction design")
    print("        Expect: 3 fragments, 2 junctions, junction sequences verified")
    print("=" * 60)
    random.seed(42)
    long_seq = ''.join(random.choices('ACGT', k=1200))
    df6, jdf6 = design_pca_oligos(
        sequence=long_seq,
        fragment_length_range=(350, 450),
        junction_overlap_length=25,
        output_csv="test6_multifrag.csv",
    )
    print(f"\nOligo table shape: {df6.shape}")
    print(f"Fragments in oligo table: {sorted(df6['fragment'].unique())}")
    if not jdf6.empty:
        print("\nJunction table:")
        print(jdf6[['junction_name', 'junction_length', 'junction_tm_C',
                     'junction_gc_%', 'warnings']].to_string())

        # Verify junction sequences match between fragments
        print("\n--- Junction sequence verification ---")
        fragments_6 = [long_seq[i:j] for i, j in
                       zip([0] + list(jdf6['junction_end'] - 25),
                           list(jdf6['junction_end']) + [len(long_seq)])]
        for _, jrow in jdf6.iterrows():
            js = jrow['junction_sequence']
            pos = jrow['junction_start']
            actual = long_seq[pos : pos + len(js)]
            match = "✓ MATCH" if js.upper() == actual.upper() else "✗ MISMATCH"
            print(f"  {jrow['junction_name']}: junction seq vs full sequence — {match}")
    else:
        print("  No junctions (single fragment or junction_overlap_length=0)")


