"""
Chart style constants shared across all modules.
Weather-themed palette with cool blues and greens.
"""

# ── Palette ────────────────────────────────────────────────────────────────────
PALETTE = [
    "#3D5A3C",  # forest
    "#B87D4B",  # copper
    "#8B9D83",  # sage
    "#D4AF37",  # gold
    "#4A4A4A",  # slate
    "#2C2C2C",  # charcoal
    "#5e7855",  # alternate green
    "#c29063",  # alternate copper
    "#A3B19D",  # light sage
    "#1A1A1A",  # dark charcoal
]

BLUE     = "#3D5A3C"
INDIGO   = "#B87D4B"
VIOLET   = "#8B9D83"
EMERALD  = "#8B9D83"
AMBER    = "#D4AF37"
CYAN     = "#4A4A4A"
RED      = "#2C2C2C"


def base_layout(title: str = "", height: int = 380) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, family="'DM Sans', sans-serif", color="#2C2C2C")),
        height=height,
        paper_bgcolor="#F8F6F1",
        plot_bgcolor="white",
        font=dict(family="'DM Sans', sans-serif", size=12, color="#4A4A4A"),
        margin=dict(t=50, b=40, l=20, r=20),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.04)",
            linecolor="rgba(0,0,0,0.08)",
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.04)",
            linecolor="rgba(0,0,0,0.08)",
            showgrid=True,
        ),
    )


def heatmap_layout(title: str = "", height: int = 500) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, family="'DM Sans', sans-serif", color="#2C2C2C")),
        height=height,
        paper_bgcolor="#F8F6F1",
        plot_bgcolor="white",
        font=dict(family="'DM Sans', sans-serif", size=11, color="#4A4A4A"),
        margin=dict(t=50, b=20, l=20, r=20),
    )
