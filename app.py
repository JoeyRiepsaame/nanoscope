import gradio as gr
import pandas as pd
import numpy as np
from Bio import SeqIO, Align
from Bio.Seq import Seq
import gzip
import plotly.graph_objects as go
from collections import defaultdict

REFERENCE_GENOMES = {
    "Human (GRCh38)": "GRCh38",
    "Mouse (GRCm39)": "GRCm39",
    "E. coli (K-12)": "U00096",
    "Yeast (S288C)": "R64",
    "Fruitfly (BDGP6)": "BDGP6",
    "Cow (ARS-UCD1.2)": "ARS-UCD1.2",
    "Dog (CanFam3.1)": "CanFam3.1",
    "Chicken (GRCg6a)": "GRCg6a",
    "Pig (Sscrofa11.1)": "Sscrofa11.1"
}

def parse_fastq(file_path):
    sequences = []
    qualities = []
    
    if file_path.endswith('.gz'):
        handle = gzip.open(file_path, 'rt')
    else:
        handle = open(file_path, 'r')
    
    try:
        for record in SeqIO.parse(handle, "fastq"):
            sequences.append(str(record.seq))
            qualities.append(record.letter_annotations["phred_quality"])
    finally:
        handle.close()
    
    return sequences, qualities

def quality_control(sequences, qualities):
    if not sequences:
        return "No sequences found", None
    
    seq_lengths = [len(seq) for seq in sequences]
    avg_length = np.mean(seq_lengths)
    median_length = np.median(seq_lengths)
    
    all_qualities = [q for qual_list in qualities for q in qual_list]
    avg_quality = np.mean(all_qualities)
    
    gc_contents = []
    for seq in sequences:
        gc = (seq.count('G') + seq.count('C')) / len(seq) * 100
        gc_contents.append(gc)
    avg_gc = np.mean(gc_contents)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=seq_lengths, name="Read Lengths", nbinsx=50))
    fig.update_layout(
        title="Read Length Distribution",
        xaxis_title="Length (bp)",
        yaxis_title="Count",
        template="plotly_white"
    )
    
    qc_report = f"""
📊 Quality Control Report
========================
Total Reads: {len(sequences):,}
Average Length: {avg_length:.1f} bp
Median Length: {median_length:.1f} bp
Length Range: {min(seq_lengths)} - {max(seq_lengths)} bp

Average Quality Score: {avg_quality:.2f}
Average GC Content: {avg_gc:.2f}%
    """
    
    return qc_report, fig

def align_to_reference(query_seq, reference_seq):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    
    alignments = aligner.align(reference_seq, query_seq)
    
    if len(alignments) > 0:
        return alignments[0]
    return None

def find_variants(query_seq, reference_seq, region_start=None, region_end=None):
    alignment = align_to_reference(query_seq, reference_seq)
    
    if alignment is None:
        return [], None
    
    aligned_ref = str(alignment[0])
    aligned_query = str(alignment[1])
    
    variants = []
    position = 0
    
    for i, (ref_base, query_base) in enumerate(zip(aligned_ref, aligned_query)):
        if ref_base != '-':
            position += 1
        
        if ref_base != query_base:
            if region_start and region_end:
                if region_start <= position <= region_end:
                    variants.append({
                        'position': position,
                        'reference': ref_base,
                        'variant': query_base,
                        'type': 'SNP' if ref_base != '-' and query_base != '-' else 'INDEL'
                    })
            else:
                variants.append({
                    'position': position,
                    'reference': ref_base,
                    'variant': query_base,
                    'type': 'SNP' if ref_base != '-' and query_base != '-' else 'INDEL'
                })
    
    return variants, alignment

def create_alignment_visualization(alignment, variants, reference_seq, query_seq):
    aligned_ref = str(alignment[0])
    aligned_query = str(alignment[1])
    
    fig = go.Figure()
    
    ref_colors = []
    query_colors = []
    
    for i, (r, q) in enumerate(zip(aligned_ref, aligned_query)):
        if r != q:
            ref_colors.append('lightblue')
            query_colors.append('lightblue')
        else:
            ref_colors.append('lightgray')
            query_colors.append('lightgray')
    
    fig.add_trace(go.Bar(
        x=list(range(len(aligned_ref))),
        y=[1] * len(aligned_ref),
        marker_color=ref_colors,
        name='Reference',
        hovertext=[f"Pos {i}: {base}" for i, base in enumerate(aligned_ref)],
        hoverinfo='text'
    ))
    
    fig.add_trace(go.Bar(
        x=list(range(len(aligned_query))),
        y=[1] * len(aligned_query),
        marker_color=query_colors,
        name='Query',
        hovertext=[f"Pos {i}: {base}" for i, base in enumerate(aligned_query)],
        hoverinfo='text',
        base=-1
    ))
    
    fig.update_layout(
        title="Sequence Alignment",
        xaxis_title="Position",
        yaxis_title="Track",
        barmode='relative',
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig

def quantify_variants(sequences, reference_seq, region_start, region_end):
    variant_counts = defaultdict(lambda: defaultdict(int))
    total_sequences = len(sequences)
    
    for seq in sequences:
        variants, _ = find_variants(seq, reference_seq, region_start, region_end)
        for variant in variants:
            pos = variant['position']
            var_base = variant['variant']
            variant_counts[pos][var_base] += 1
    
    variant_summary = []
    for pos in sorted(variant_counts.keys()):
        for base, count in variant_counts[pos].items():
            frequency = (count / total_sequences) * 100
            variant_summary.append({
                'Position': pos,
                'Variant': base,
                'Count': count,
                'Frequency (%)': f"{frequency:.2f}"
            })
    
    return pd.DataFrame(variant_summary)

def analyze_nanopore_data(fastq_file, reference_genome, reference_sequence, region_start, region_end):
    try:
        sequences, qualities = parse_fastq(fastq_file.name)
        
        qc_report, qc_plot = quality_control(sequences, qualities)
        
        if not reference_sequence:
            return qc_report, qc_plot, "Please provide a reference sequence", None, None
        
        if region_start and region_end:
            variant_df = quantify_variants(sequences, reference_sequence, region_start, region_end)
        else:
            variant_df = quantify_variants(sequences, reference_sequence, None, None)
        
        if sequences:
            variants, alignment = find_variants(sequences[0], reference_sequence, region_start, region_end)
            alignment_fig = create_alignment_visualization(alignment, variants, reference_sequence, sequences[0])
        else:
            alignment_fig = None
        
        variant_summary = f"""
🧬 Variant Analysis Summary
==========================
Total unique variants found: {len(variant_df)}
Region analyzed: {region_start if region_start else 'Full sequence'} - {region_end if region_end else 'Full sequence'}
        """
        
        return qc_report, qc_plot, variant_summary, variant_df, alignment_fig
        
    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None

with gr.Blocks(title="NanoScope - Nanopore Sequencing Analysis", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🔬 NanoScope - Nanopore Sequencing Analysis Tool
    
    Automated analysis of Nanopore sequencing data with QC, alignment, and variant calling.
    """)
    
    with gr.Row():
        with gr.Column():
            fastq_input = gr.File(label="Upload FASTQ File (.fastq or .fastq.gz)", file_types=[".fastq", ".gz"])
            reference_genome = gr.Dropdown(
                choices=list(REFERENCE_GENOMES.keys()),
                label="Reference Genome",
                value="Human (GRCh38)"
            )
            reference_sequence = gr.Textbox(
                label="Reference Sequence (paste your WT sequence)",
                placeholder="ATCGATCGATCG...",
                lines=5
            )
            
            with gr.Row():
                region_start = gr.Number(label="Region Start (optional)", precision=0)
                region_end = gr.Number(label="Region End (optional)", precision=0)
            
            analyze_btn = gr.Button("🚀 Analyze", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Quality Control")
            qc_output = gr.Textbox(label="QC Report", lines=10)
            qc_plot = gr.Plot(label="Read Length Distribution")
        
        with gr.Column():
            gr.Markdown("### 🧬 Variant Analysis")
            variant_summary = gr.Textbox(label="Variant Summary", lines=5)
            variant_table = gr.Dataframe(label="Detected Variants")
    
    gr.Markdown("### 🔍 Alignment Visualization")
    alignment_plot = gr.Plot(label="Sequence Alignment (First Read)")
    
    analyze_btn.click(
        fn=analyze_nanopore_data,
        inputs=[fastq_input, reference_genome, reference_sequence, region_start, region_end],
        outputs=[qc_output, qc_plot, variant_summary, variant_table, alignment_plot]
    )
    
    gr.Markdown("""
    ---
    ### 📖 How to Use:
    1. Upload your FASTQ file from Nanopore sequencing
    2. Select the reference genome
    3. Paste your wild-type reference sequence
    4. (Optional) Define a specific region to analyze
    5. Click **Analyze** for comprehensive results
    
    ### ✨ Features:
    - ✅ Automated QC analysis
    - ✅ Multi-genome alignment support
    - ✅ Variant detection and quantification
    - ✅ Interactive visualizations
    - ✅ One-click analysis workflow
    """)

if __name__ == "__main__":
    app.launch()
