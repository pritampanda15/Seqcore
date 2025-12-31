#!/usr/bin/env python3
"""
Single-Cell RNA-seq Analysis with Seqcore
==========================================

This example demonstrates how to use Seqcore for single-cell RNA sequencing
data analysis using real data from the Human Cell Atlas.

Dataset: 10x Genomics PBMC 3k dataset
- ~2,700 peripheral blood mononuclear cells
- Commonly used benchmark dataset for single-cell analysis

Author: Dr. Pritam Kumar Panda @ Stanford University
"""

import os
import urllib.request
from pathlib import Path

import numpy as np

# Check if required packages are available
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Note: h5py not installed. Install with: pip install h5py")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Note: pandas not installed. Install with: pip install pandas")


def download_dataset(url: str, filepath: Path, description: str = "file") -> bool:
    """Download a file with progress indication."""
    if filepath.exists():
        print(f"Dataset already exists: {filepath}")
        return True

    print(f"Downloading {description}...")
    print(f"URL: {url}")

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        print(f"\nDownloaded: {filepath} ({filepath.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def example_1_basic_h5ad_reading():
    """
    Example 1: Reading and exploring H5AD files

    H5AD is the standard format for single-cell data (AnnData).
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Reading H5AD Single-Cell Data")
    print("=" * 60)

    if not HAS_H5PY:
        print("Skipping: h5py not installed")
        return

    import seqcore as sc

    # Download PBMC 3k dataset (filtered)
    data_dir = Path("data/single_cell")
    h5ad_url = "https://cf.10xgenomics.com/samples/cell-exp/7.0.1/SC3pv3_GEX_Human_PBMC/SC3pv3_GEX_Human_PBMC_filtered_feature_bc_matrix.h5"
    h5ad_file = data_dir / "pbmc3k_filtered.h5"

    if not download_dataset(h5ad_url, h5ad_file, "PBMC 3k dataset"):
        print("Using synthetic data instead...")
        # Create synthetic single-cell data for demonstration
        n_cells = 1000
        n_genes = 500

        # Simulate count matrix
        counts = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
        cell_barcodes = [f"Cell_{i:04d}" for i in range(n_cells)]

        print(f"\nSynthetic data created:")
        print(f"  Cells: {n_cells}")
        print(f"  Genes: {n_genes}")
        print(f"  Matrix shape: {counts.shape}")
        print(f"  Total counts: {counts.sum():,}")
        print(f"  Mean counts/cell: {counts.sum(axis=1).mean():.1f}")
        return

    # Read H5 file structure
    print(f"\nReading H5 file: {h5ad_file}")

    with h5py.File(h5ad_file, 'r') as f:
        print("\nH5 file structure:")
        def print_structure(name, obj):
            print(f"  {name}: {type(obj).__name__}")
        f.visititems(print_structure)

        # 10x Genomics format
        if 'matrix' in f:
            matrix_group = f['matrix']

            # Get dimensions
            shape = matrix_group['shape'][:]
            n_genes, n_cells = shape

            # Get data
            data = matrix_group['data'][:]
            indices = matrix_group['indices'][:]
            indptr = matrix_group['indptr'][:]

            # Get gene names
            features = matrix_group['features']
            gene_names = [x.decode() for x in features['name'][:]]
            gene_ids = [x.decode() for x in features['id'][:]]

            # Get cell barcodes
            barcodes = [x.decode() for x in matrix_group['barcodes'][:]]

            print(f"\nDataset Summary:")
            print(f"  Cells: {n_cells:,}")
            print(f"  Genes: {n_genes:,}")
            print(f"  Non-zero entries: {len(data):,}")
            print(f"  Sparsity: {1 - len(data)/(n_genes*n_cells):.2%}")

            # Convert sparse to dense for small subset
            print("\nFirst 10 genes:")
            for i, (name, gid) in enumerate(zip(gene_names[:10], gene_ids[:10])):
                print(f"  {i+1}. {name} ({gid})")

            print("\nFirst 5 cell barcodes:")
            for i, bc in enumerate(barcodes[:5]):
                print(f"  {i+1}. {bc}")


def example_2_quality_control():
    """
    Example 2: Quality Control Metrics

    Calculate common QC metrics for single-cell data.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Single-Cell Quality Control")
    print("=" * 60)

    if not HAS_H5PY or not HAS_PANDAS:
        print("Skipping: h5py and pandas required")
        return

    import seqcore as sc

    # Create realistic synthetic data
    np.random.seed(42)
    n_cells = 2700
    n_genes = 1000

    # Simulate different cell populations
    # Some cells are high quality, some are dying (high mito), some are doublets

    # Base expression
    counts = np.random.negative_binomial(2, 0.3, size=(n_cells, n_genes)).astype(np.float32)

    # Add mitochondrial genes (genes 0-12 are "MT-" genes)
    n_mito_genes = 13
    mito_mask = np.zeros(n_genes, dtype=bool)
    mito_mask[:n_mito_genes] = True

    # Some cells have high mitochondrial content (dying cells)
    dying_cells = np.random.choice(n_cells, size=int(n_cells * 0.1), replace=False)
    counts[dying_cells, :n_mito_genes] *= 5

    # Some cells are doublets (high total counts)
    doublets = np.random.choice(n_cells, size=int(n_cells * 0.05), replace=False)
    counts[doublets, :] *= 2

    gene_names = [f"MT-{i}" if i < n_mito_genes else f"Gene_{i}" for i in range(n_genes)]

    print(f"\nSimulated PBMC dataset:")
    print(f"  Cells: {n_cells:,}")
    print(f"  Genes: {n_genes:,}")
    print(f"  Mitochondrial genes: {n_mito_genes}")

    # Calculate QC metrics
    print("\nCalculating QC metrics...")

    # Total counts per cell
    total_counts = counts.sum(axis=1)

    # Genes detected per cell
    genes_detected = (counts > 0).sum(axis=1)

    # Mitochondrial percentage
    mito_counts = counts[:, mito_mask].sum(axis=1)
    mito_pct = (mito_counts / total_counts) * 100

    print("\nQC Summary Statistics:")
    print(f"\n  Total counts per cell:")
    print(f"    Min: {total_counts.min():.0f}")
    print(f"    Max: {total_counts.max():.0f}")
    print(f"    Mean: {total_counts.mean():.1f}")
    print(f"    Median: {np.median(total_counts):.1f}")

    print(f"\n  Genes detected per cell:")
    print(f"    Min: {genes_detected.min()}")
    print(f"    Max: {genes_detected.max()}")
    print(f"    Mean: {genes_detected.mean():.1f}")

    print(f"\n  Mitochondrial percentage:")
    print(f"    Min: {mito_pct.min():.2f}%")
    print(f"    Max: {mito_pct.max():.2f}%")
    print(f"    Mean: {mito_pct.mean():.2f}%")

    # Apply QC filters
    print("\n  Applying QC filters:")
    min_genes = 200
    max_genes = 2500
    max_mito_pct = 20

    pass_filter = (
        (genes_detected >= min_genes) &
        (genes_detected <= max_genes) &
        (mito_pct <= max_mito_pct)
    )

    print(f"    Min genes: {min_genes}")
    print(f"    Max genes: {max_genes}")
    print(f"    Max mito %: {max_mito_pct}%")
    print(f"\n    Cells passing QC: {pass_filter.sum():,} / {n_cells:,} ({pass_filter.mean()*100:.1f}%)")

    # Filter the data
    counts_filtered = counts[pass_filter]
    print(f"    Filtered matrix shape: {counts_filtered.shape}")


def example_3_gene_expression_analysis():
    """
    Example 3: Gene Expression Analysis with Seqcore

    Analyze gene expression patterns using seqcore's vectorized operations.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Gene Expression Analysis")
    print("=" * 60)

    import seqcore as sc

    # Create synthetic gene sequences for marker genes
    print("\nAnalyzing marker gene sequences...")

    # Common immune cell marker genes (simplified sequences)
    marker_genes = {
        "CD3D": "ATGGCCCCTGGAGCTAGCCTGGCCCAGCTCCTGGTGCTGGTG",  # T cells
        "CD4": "ATGAACCGGGGAGTCCCTTTTAGGCACTTGCTTCTGGTGCTG",   # Helper T cells
        "CD8A": "ATGGCCTTACCAGTGACCGCCTTGCTCCTGCCGCTGGCCTTG",  # Cytotoxic T cells
        "CD19": "ATGCCACCTCCTCGCCTCCTCTTCTTCCTCCTCTTCCTCACC",  # B cells
        "CD14": "ATGGAGCGCGCGTCCTGCTTGTTGCTGCTGCTGCTGCTGCTG",  # Monocytes
        "NCAM1": "ATGCTGCGAACTAAGGATCTCGTGCTGGCGCTGCTCCTGCTG", # NK cells
    }

    # Create DNA arrays for each marker
    sequences = list(marker_genes.values())
    names = list(marker_genes.keys())

    dna_array = sc.DNAArray(sequences)

    print(f"\nMarker genes analyzed: {len(names)}")

    # Calculate sequence properties
    gc_content = sc.gc_content(dna_array)
    lengths = sc.length(dna_array)

    print("\nMarker Gene Properties:")
    print("-" * 50)
    print(f"{'Gene':<10} {'Length':<10} {'GC%':<10}")
    print("-" * 50)

    for name, length, gc in zip(names, lengths, gc_content):
        print(f"{name:<10} {length:<10} {gc:.1f}%")

    # Translate to protein
    print("\nTranslating marker genes to protein...")
    proteins = sc.translate(dna_array)

    print("\nProtein sequences (first 15 aa):")
    for name, prot in zip(names, proteins.sequences):
        print(f"  {name}: {prot[:15]}...")

    # K-mer analysis
    print("\nK-mer analysis (k=3) for CD3D:")
    cd3d_seq = sc.DNAArray([marker_genes["CD3D"]])
    kmers = sc.count_kmers(cd3d_seq, k=3)

    # Get top k-mers (count_kmers returns dict for single sequence)
    top_kmers = sorted(kmers.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  Top 5 trinucleotides:")
    for kmer, count in top_kmers:
        print(f"    {kmer}: {count}")


def example_4_batch_processing():
    """
    Example 4: Batch Processing Large Datasets

    Demonstrate memory-efficient processing of large single-cell datasets.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Processing Large Datasets")
    print("=" * 60)

    import seqcore as sc

    # Simulate processing a large dataset in batches
    print("\nSimulating batch processing of 100,000 cells...")

    total_cells = 100_000
    batch_size = 10_000
    n_genes = 2000

    # Track metrics across batches
    all_total_counts = []
    all_genes_detected = []

    print(f"\nProcessing {total_cells:,} cells in batches of {batch_size:,}")
    print("-" * 50)

    for batch_idx in range(total_cells // batch_size):
        # Simulate loading a batch
        batch_counts = np.random.negative_binomial(
            2, 0.3, size=(batch_size, n_genes)
        ).astype(np.float32)

        # Calculate metrics for this batch
        batch_total = batch_counts.sum(axis=1)
        batch_detected = (batch_counts > 0).sum(axis=1)

        all_total_counts.extend(batch_total)
        all_genes_detected.extend(batch_detected)

        # Memory-efficient: process and discard
        cells_processed = (batch_idx + 1) * batch_size
        print(f"  Batch {batch_idx + 1}: Processed {cells_processed:,} cells")

    # Final statistics
    all_total_counts = np.array(all_total_counts)
    all_genes_detected = np.array(all_genes_detected)

    print("\nFinal Statistics (all batches):")
    print(f"  Total cells processed: {len(all_total_counts):,}")
    print(f"  Mean counts/cell: {all_total_counts.mean():.1f}")
    print(f"  Mean genes/cell: {all_genes_detected.mean():.1f}")

    # Use seqcore timer
    print("\nBenchmarking with seqcore timer:")
    with sc.timer() as t:
        # Simulate a compute-intensive operation
        large_array = sc.DNAArray(["ACGT" * 250] * 1000)
        gc = sc.gc_content(large_array)
    print(f"  GC content for 1000 sequences: {t.elapsed:.4f}s")


def example_5_cellxgene_integration():
    """
    Example 5: Working with CellxGene Data

    Demonstrate how to work with data from CellxGene portal.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: CellxGene Data Integration")
    print("=" * 60)

    print("""
CellxGene (https://cellxgene.cziscience.com/) hosts thousands of
single-cell datasets. Here's how to work with them using Seqcore:

1. Download a dataset from CellxGene:
   - Go to https://cellxgene.cziscience.com/
   - Find a dataset (e.g., "Tabula Sapiens")
   - Click "Download" → Select H5AD format

2. Load with Seqcore:

   import seqcore as sc

   # Read the H5AD file
   data = sc.read("tabula_sapiens.h5ad")

   # Access expression matrix
   X = data.X  # Sparse or dense matrix

   # Access gene names
   genes = data.var_names

   # Access cell metadata
   cells = data.obs

   # Filter by cell type
   t_cells = data[data.obs['cell_type'] == 'T cell']

3. Analyze with Seqcore:

   # Get gene sequences for differentially expressed genes
   de_genes = ['CD3D', 'CD4', 'CD8A', 'FOXP3']
   sequences = sc.fetch_sequences(de_genes, database='ensembl')

   # Analyze sequence properties
   gc = sc.gc_content(sequences)
   motifs = sc.find_pattern(sequences, 'TATA[AT]A')

Note: Full H5AD support requires: pip install anndata scanpy
""")


def main():
    """Run all single-cell analysis examples."""
    print("=" * 60)
    print("SEQCORE SINGLE-CELL ANALYSIS EXAMPLES")
    print("Author: Dr. Pritam Kumar Panda @ Stanford University")
    print("=" * 60)

    # Run examples
    example_1_basic_h5ad_reading()
    example_2_quality_control()
    example_3_gene_expression_analysis()
    example_4_batch_processing()
    example_5_cellxgene_integration()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)

    print("""
Next Steps:
-----------
1. Install additional dependencies:
   pip install h5py pandas anndata scanpy

2. Download real data from:
   - 10x Genomics: https://www.10xgenomics.com/resources/datasets
   - CellxGene: https://cellxgene.cziscience.com/
   - GEO: https://www.ncbi.nlm.nih.gov/geo/

3. Try GPU acceleration for large datasets:
   pip install seqcore[gpu]

   with sc.device("cuda"):
       gc = sc.gc_content(large_sequences)
""")


if __name__ == "__main__":
    main()
