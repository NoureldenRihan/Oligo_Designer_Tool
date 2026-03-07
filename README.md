# PCA Oligo Designer

A robust, bioinformatics-driven Python tool for designing overlapping, alternating oligonucleotides for Polymerase Cycling Assembly (PCA). 

This script automates the complex process of DNA synthesis design. It takes a target DNA sequence of any length, fragments it into manageable pieces, and tiles forward and reverse oligos that overlap with their neighbors. By rigorously screening these overlaps for thermodynamic stability and known biological failure modes, it ensures reliable polymerase extension during PCA cycling.



## 🧬 Core Features & Quality Control

Designing oligos for PCA requires careful thermodynamic balancing to prevent the reaction from failing in the lab. This tool implements several critical quality checks:

* Thermodynamic Balancing: Uses the gold-standard SantaLucia 1998 nearest-neighbor model (via `primer3-py`) with salt correction to calculate overlap melting temperatures (Tm). 
* GC Content Optimization: Ensures overlap GC% falls within a specified range to avoid stable secondary structures (high GC) or unreliable annealing (low GC).
* Polymerase Slippage Prevention: Automatically detects and flags homopolymer runs (e.g., AAAA) and dinucleotide repeats (e.g., ATATAT) which are known to cause polymerases to "slip" and produce insertions/deletions.
* Vicinity Buffer Scanning: Prevents edge-clipping issues by scanning a configurable buffer zone (default 5 bp) around the overlap, ensuring oligos do not partially anneal into problematic repetitive regions just outside their boundaries.
* Thermodynamic Hairpin Detection: Calculates the actual folding free energy (ΔG) of each full oligo using primer3.calc_hairpin(). It flags oligos with stable intramolecular structures (default ΔG ≤ -4.0 kcal/mol) that would compete with productive PCA annealing.
* Global Misprime Scanning: Simulates the reaction tube environment by taking the 3' end of every oligo (the probe) and sliding it across the entire sequence on both strands. It flags potential misprimes where unintended annealing could create chimeric products. This utilizes a highly optimized pre-indexed k-mer search for maximum performance.
* Inter-Fragment Junction Design: For genes too long for a single reaction, the tool automatically designs overlapping junctions (default 25 bp) between adjacent fragments. This ensures the independently assembled fragments can be successfully joined in a downstream stitching PCR.

## ⚙️ Installation & Dependencies
The script relies on standard scientific Python libraries. Install them via pip:

```bash
pip install primer3-py biopython pandas
```

---

## 🚀 Usage
You can use the tool out-of-the-box with established default parameters derived from PCA literature, or fully customize the constraints for your specific reaction conditions.

### Basic Example

```python
from pca_oligo_designer import design_pca_oligos

# Generate oligos and save to CSV
df, junction_df = design_pca_oligos(
    sequence="ATGACCATGATTACGCCAAGCTATTTTAGCGAA...",
    output_csv="my_oligos.csv"
)
```
### Advanced Customization

For complete control over thermodynamic limits, lengths, and salt concentrations:

```python
df, junction_df = design_pca_oligos(
    sequence="ATGC...",
    fragment_length_range=(400, 500),
    oligo_length_range=(40, 60),
    overlap_length_range=(15, 25),
    overlap_tm_range=(55.0, 65.0),
    overlap_gc_range=(40.0, 70.0),
    max_homopolymer_run=4,
    max_dinucleotide_repeat=3,
    repeat_vicinity_buffer=5,
    hairpin_dg_threshold=-4.0,
    misprime_length=18,
    misprime_tip_length=6,
    misprime_max_mismatches=4,
    junction_overlap_length=25,
    junction_tm_range=(58.0, 68.0),
    mv_conc=50.0,
    dv_conc=1.5,
    dntp_conc=0.2,
    dna_conc=250.0,
    output_csv="custom_oligos.csv"
)
```

---

## 🎛️ Detailed Configuration Reference

### Structural Parameters
- **fragment_length_range**: (min, max) length of each independent PCA fragment. Default is **(400, 500)**. Set to `None` to treat the whole sequence as one fragment.
- **oligo_length_range**: (min, max) length of each oligo. Default is **(40, 60)**. The algorithm prioritizes shorter oligos first.
- **overlap_length_range**: (min, max) length of the hybridization overlap. Default is **(15, 25)**.

### Quality Thresholds
- **overlap_tm_range**: Target melting temperature range. Default is **(55.0, 65.0)**.
- **overlap_gc_range**: Acceptable GC percentage for overlaps. Default is **(40.0, 70.0)**.
- **hairpin_dg_threshold**: The ΔG cutoff (in kcal/mol) for intramolecular hairpins. Default is **-4.0** (flags strong hairpins).

### Misprime Detection Rules
- **misprime_length**: Probe length from the 3' end used for scanning. Default is **18**.
- **misprime_tip_length**: Exact-match required at the absolute 3' tip. Default is **6**.
- **misprime_max_mismatches**: Tolerance for mismatches in the non-tip region. Default is **4**.

---

## 📊 Output Format

The script returns **Pandas DataFrames** and optionally saves them as CSV files.

**Oligos File**
- Contains names, fragment allocation, positions, full sequences, and Tm/GC data.
- Includes Boolean flag columns (e.g., `flag_homopolymer`, `flag_misprime`) for easy filtering.

**Junctions File**
- Generated for multi-fragment projects.
- Details shared sequences and provides stitching primers.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the Project  
2. Create your Feature Branch  
   ```bash
   git checkout -b feature/Feature
   ```
3. Commit your Changes  
   ```bash
   git commit -m "Add some Feature"
   ```
4. Push to the Branch  
   ```bash
   git push origin feature/Feature
   ```
5. Open a Pull Request
